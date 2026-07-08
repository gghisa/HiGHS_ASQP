/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Highs.h"
#include "qpsolver/g_active_set.hpp"

ActiveSetData::ActiveSetData(const HighsLp& lp,
                             HighsBasis& basis,
                             HighsSolution& solution,
                             HighsHessian& Q)
                            : lp_(lp),
                            basis_(basis),
                            solution_(solution),
                            Q_(Q){
    if (Q.format_ == HessianFormat::kTriangular) Q = Q.toSquare(); // make Hessian square to improve columns/rows accessing speed
    std::vector<HighsInt> basis_indices = countActiveConVar(); // basis indices can be local because they'll be stored in HFactor
    setupBasisMat(basis_indices);
    setupReducedHessian();
};
// define function to run after phase1 to extract varumns and cons that are active
std::vector<HighsInt> ActiveSetData::countActiveConVar(){
    // if a variable is kBasic, it is not in the ASM basis
    // if it is kLower or kUpper, it is in the ASM basis
    // kNonbasic is never returned by simplex phase 1 (though here it would become a basis)
    // so each dimension in the null space corresponds to a kZero variable (can constraints be kZero? if not, remove option below)
    std::vector<HighsInt> basis_indices; // create vector for the basis required by HFactor
    for (size_t i {0}; i<basis_.row_status.size(); i++){ // loop through variables
        std::vector<double> emptyvec;
        if (basis_.row_status[i] != HighsBasisStatus::kBasic){
            basis_indices.push_back(i); // Constraint count starts from 0
            if (basis_.row_status[i] == HighsBasisStatus::kZero) this->ZT_.push_back(emptyvec); // kZero means one available nullspace dimension
        }
    }
    for (size_t i {0}; i<basis_.col_status.size(); i++){ // loop through constraints
        std::vector<double> emptyvec;
        if (basis_.col_status[i] != HighsBasisStatus::kBasic){
            basis_indices.push_back(i + this->lp_.num_row_); // Variable count starts from m (nr of constraints)
            if (basis_.col_status[i] == HighsBasisStatus::kZero) this->ZT_.push_back(emptyvec); // kZero means one available nullspace dimension
        }
    }
    return basis_indices;
}
// setup basis matrix
void ActiveSetData::setupBasisMat(std::vector<HighsInt>& basis_indices){
            HighsSparseMatrix constraint_mat = this->lp_.a_matrix_; // create a copy of the constraint matrix
            printsparse(constraint_mat);
            constraint_mat.ensureRowwise(); // flip the way in which it is stored
            constraint_mat.format_ = MatrixFormat::kColwise; // but "trick it" into thinking it is still stored columnwise
            HighsInt temp_old_num_row = constraint_mat.num_row_; // flip the number of rows and columns
            constraint_mat.num_row_ = constraint_mat.num_col_; // so that when the matrix is used by HFactor
            constraint_mat.num_col_ = temp_old_num_row; // it received the constraint matrix "column wise"
            this->B_.setup(constraint_mat, basis_indices); // where each column is a constraint. its inverse transpose will have as columns the nullspace basis
            this->B_.build();
            setupBasisNullSpace();// once the basis matrix is set up, extract the null spaces basis
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
    for (size_t i {0}; i<this->basis_.row_status.size(); i++){
        if (basis_.row_status[i] == HighsBasisStatus::kLower ||
            basis_.row_status[i] == HighsBasisStatus::kUpper)
        std::cout<< i << " -- ";
    }
    std::cout<<"\nActive variables:\n";
    for (size_t i {0}; i<this->basis_.col_status.size(); i++){
        if (basis_.col_status[i] == HighsBasisStatus::kLower ||
            basis_.col_status[i] == HighsBasisStatus::kUpper)
        std::cout<< i << " -- ";
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