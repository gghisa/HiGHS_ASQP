/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Highs.h"
#include "qpsolver/g_active_set.hpp"

ActiveSetData::ActiveSetData(const HighsBasis& basis, // we will discard this, or can we make use of it?
                             const HighsLp& lp,
                             HighsSolution& solution,
                             HighsHessian& Q)
                            : lp_(lp),
                            solution_(solution),
                            Q_(Q){
    if (Q.format_ == HessianFormat::kTriangular) Q = Q.toSquare(); // make Hessian square to improve columns/rows accessing speed
    std::vector<HighsInt> basis_indices = setupActiveVarCon(basis.col_status, basis.row_status); // basis indices can be local because they'll be stored in HFactor
    setupBasisMat(basis, basis_indices);
    setupReducedHessian();
};
// given a vector of statuses, extract whether active or not and the corresponding location index
void ActiveSetData::setupActive(const std::vector<HighsBasisStatus>& status,
                                std::vector<HighsInt>& index,
                                std::vector<HighsBasisStatus>& active_status,
                                const HighsInt offset){
    std::vector<double> emptyvec;
    for (size_t i {0}; i<status.size(); i++){
        if (status[i] != HighsBasisStatus::kBasic){
            // when the constraints and bounds in the problem are less than the number of variables, then there may be kNonbasic elements
            // they are not in the simplex basis, yet we need them to construct the invertible matrix B = [A:V]
            index.push_back(i + offset);
            active_status.push_back(status[i]);
            if (status[i] == HighsBasisStatus::kZero) this->ZT_.push_back(emptyvec); // kZero means one available nullspace dimension
            // kNonbasic is ignored
        }
    }
}
// define function to run after phase1 to extract varumns and cons that are active
std::vector<HighsInt> ActiveSetData::setupActiveVarCon(const std::vector<HighsBasisStatus>& var_status,
                                                       const std::vector<HighsBasisStatus>& con_status){
    // variables' indexes start counting from m, which is the number of constraints. Constraint count starts from 0, as required by HFactor
    setupActive(var_status, this->active_var_, this->status_var_, this->lp_.num_row_);
    setupActive(con_status, this->active_con_, this->status_con_, 0);
    std::vector<HighsInt> basis_indices; // create vector for the basis required by HFactor
    basis_indices.insert(basis_indices.end(), this->active_con_.begin(), this->active_con_.end());
    basis_indices.insert(basis_indices.end(), this->active_var_.begin(), this->active_var_.end());
    return basis_indices;
}
// setup basis matrix
void ActiveSetData::setupBasisMat(const HighsBasis& basis, std::vector<HighsInt>& basis_indices){
            HighsSparseMatrix constraint_mat = this->lp_.a_matrix_; // create a copy of the constraint matrix
            printsparse(constraint_mat);
            constraint_mat.ensureRowwise(); // flip the way in which it is stored
            constraint_mat.format_ = MatrixFormat::kColwise; // but "trick it" into thinking it is still stored columnwise
            HighsInt temp_old_num_row = constraint_mat.num_row_; // flip the number of rows and columns
            constraint_mat.num_row_ = constraint_mat.num_col_; // so that when the matrix is used by HFactor
            constraint_mat.num_col_ = temp_old_num_row; // it received the constraint matrix "column wise"
            this->B_.setup(constraint_mat, basis_indices); // where each column is a constraint. its inverse transpose will have as columns the nullspace basis
            // once the basis matrix is set up, extract the null spaces basis
            this->B_.build();
            setupBasisNullSpace();
}
void ActiveSetData::setupBasisNullSpace(){
            HighsInt n_active = this->lp_.num_col_ - (HighsInt)this->ZT_.size(); // wouldn't it be better to store this as a member of the class?
            for (size_t i {0}; i < this->ZT_.size() ; i++){
                std::vector<double> z_col(lp_.num_col_ ); // create unit vector
                z_col.assign(this->lp_.num_col_,0.);
                z_col[i + n_active] = 1.; // set unit entry at the index for the desired column of B^{-T}
                this->B_.btranCall(z_col); // solve B^T\cdot e_i = z_col
                this->ZT_[i] = z_col; // add column vector to transpose of Z, effectively adding a new row
                // do I have to check if the size of the matrix is correct? should I check if it is empty?
            }
            printmatrix(this->ZT_);
}

void ActiveSetData::setupReducedHessian(){
    assert (this->Q_.format_ == HessianFormat::kSquare);
    assert (this->redhes_.empty());
    for (size_t i {0}; i < this->ZT_.size(); i++){// loop over the rows of the reduced hessian
        std:vector<double> empty_row_red_hessian(this->ZT_.size());
        this->redhes_.push_back(empty_row_red_hessian); // initialise new row of reduced hessian
    }
    for (size_t i {0}; i < this->ZT_.size(); i++){// loop over the rows of Z^T
        std::vector<double> row(this->lp_.num_col_); // row of Z^T Q
        this->Q_.product(this->ZT_[i], row); // compute it
        double sum {0.};
        for (size_t j {0}; j <= i; j++){ // loop through columns of Z, up to the current row of Z^T, to only compute lower triangle of red_hessian_
            for (HighsInt k {0}; k < this->lp_.num_col_; k++){ // inner produce of row of Z^T Q with column of Z
                sum += row[k] * this->ZT_[j][k];
            }
            this->redhes_[i][j] = sum;
            this->redhes_[j][i] = sum; // off-diagonal symmetric element, could i make this a pointer instead?
            // i could check that second assignment only happens when i != j, but the check would run for nothing most of the time
        }
    }

    for (size_t i {0}; i < this->ZT_.size(); i++){// loop over the rows of the reduced hessian
        assert(this->redhes_[i].size() == this->redhes_.size()); // check that every row is as long as it needs to be
    }
    printmatrix(this->redhes_);
}

size_t ActiveSetData::getSizeNullSpace(){
    return this->ZT_.size();
}

void ActiveSetData::printActive(){
    std::cout<<"\nActive constraints:\n";
    for (size_t i {0}; i<this->active_con_.size(); i++){
        std::cout<< active_con_[i] << " -- ";
    }
    std::cout<<"\nActive variables:\n";
    for (size_t i {0}; i<this->active_var_.size(); i++){
        std::cout<< active_var_[i] << " -- ";
    }
}

void ActiveSetData::printvector(const std::vector<double>& vec){
    // from Claude.ai
    for (double val : vec) {
        std::cout << val << " ";
    }
    std::cout.flush();
}
void ActiveSetData::printmatrix(const std::vector<std::vector<double>>& mat){
    // from Claude.ai
    for (const std::vector<double>& row : mat) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout <<"\n";
    }
    std::cout << "-----\n";
    std::cout.flush();
}
void ActiveSetData::printsparse(const HighsSparseMatrix& mat){
    std::vector<std::vector<double>> dense;
    for (HighsInt i {0}; i < mat.num_row_; i++){
        std::vector<double> col;
        std::vector<double> x;
        x.assign(mat.num_row_, 0.);
        x[i] = 1.;
        mat.productTranspose(col, x);
        dense.push_back(col);
    }
    printmatrix(dense);
}
void ActiveSetData::computeLocGrad(){// g + Q x_k
    this->Q_.product(this->solution_.col_value, this->loc_grad_);
    for (HighsInt i {0}; i < this->lp_.num_col_; i++){ // add g to Q x_k
        this->loc_grad_[i] += this->lp_.col_cost_[i];
    }
}

void ActiveSetData::computeRedGrad(){
    this->red_grad_.clear();
    for (size_t i {0}; i < this->ZT_.size(); i++){
        double sum {0};
        for (HighsInt j {0}; j < this->lp_.num_col_; j++){
            sum += this->ZT_[i][j] * this->loc_grad_[j];
        }
        this->red_grad_.push_back(sum);// Z^T (g + Q x_k)
    }
}
void ActiveSetData::price(){
    // compute pricing for each active constraint: Y^T (g + Qx)
    computeRedGrad();
}