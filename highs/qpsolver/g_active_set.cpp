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
                             const HighsBasis& basis,
                             HighsSolution& solution,
                             HighsHessian& Q)
                            : lp_(lp),
                            solution_(solution),
                            Q_(Q){
    if (Q.format_ == HessianFormat::kTriangular) Q = Q.toSquare(); // make Hessian square to improve columns/rows accessing speed
    initAsmBasis(basis); // basis indices can be local because they'll be stored in HFactor
    setupBasisMat();
    setupReducedHessian();
};
// convert Asm Status to Highs
HighsBasisStatus ActiveSetData::AsmStatusToHighs(const AsmBasisStatus& status){
    if (status==AsmBasisStatus::kLower || status==AsmBasisStatus::kEquality) return HighsBasisStatus::kLower;
    else if (status==AsmBasisStatus::kFreeInBasis) return HighsBasisStatus::kZero;// should not happen with constraints but how do I check it?
    else if (status==AsmBasisStatus::kInactive) return HighsBasisStatus::kBasic;
    else return HighsBasisStatus::kNonbasic; // should never happen
};
// convert Highs Status to Asm
AsmBasisStatus ActiveSetData::HighsStatusToAsm(const HighsBasisStatus& status, HighsInt index){
    if (index >= this->lp_.num_row_){ // it is a variable
        index -= this->lp_.num_row_; // if equality this should be a fixed variable and needs presolve
        if (this->lp_.col_lower_[index] == this->lp_.col_upper_[index]) return AsmBasisStatus::kEquality;
    // row_ check should only happen if it is indeed a row we are working with
    } else if (this->lp_.row_lower_[index] == this->lp_.row_upper_[index]) return AsmBasisStatus::kEquality;
    // if no equality is found:
    if(status == HighsBasisStatus::kLower) return AsmBasisStatus::kLower;
    else if(status == HighsBasisStatus::kUpper) return AsmBasisStatus::kUpper;
    else return AsmBasisStatus::kInactive;
};
//
void ActiveSetData::initAsmBasisLoop(const std::vector<HighsBasisStatus>& status, const bool isconstr){
    for (size_t i {0}; i<status.size(); i++){ // loop through variables
        if (status[i] != HighsBasisStatus::kBasic){// inside the loop we deal with simplex nonbasic variables only
            if (isconstr){
                this->basis_idxs_.push_back(i); // Constraint count starts from 0
                this->basis_status_.push_back(HighsStatusToAsm(status[i],i));
                assert(status[i] != HighsBasisStatus::kZero); // kZero should not occurr for constraints
            } else {
                HighsInt index = i + this->lp_.num_row_;
                this->basis_idxs_.push_back(index); // Variable count starts from number of constraints
                this->basis_status_.push_back(HighsStatusToAsm(status[i],index));
                if (status[i] == HighsBasisStatus::kZero){
                    std::vector<double> emptyvec;
                    this->ZT_.push_back(emptyvec); // kZero means one available nullspace dimension
                }
            }
        }
    }
}
// define function to run after phase1 to extract varumns and cons that are active
void ActiveSetData::initAsmBasis(const HighsBasis& basis){
    this->con_status_ = basis.row_status;
    this->var_status_ = basis.col_status;
    this->basis_idxs_.clear();// clear vector for the basis required by HFactor
    initAsmBasisLoop(this->con_status_, true);
    initAsmBasisLoop(this->var_status_, false);
}
// setup basis matrix
void ActiveSetData::setupBasisMat(){
            HighsSparseMatrix constraint_mat = this->lp_.a_matrix_; // create a copy of the constraint matrix
            printsparse(constraint_mat);
            constraint_mat.ensureRowwise(); // flip the way in which it is stored
            constraint_mat.format_ = MatrixFormat::kColwise; // but "trick it" into thinking it is still stored columnwise
            HighsInt temp_old_num_row = constraint_mat.num_row_; // flip the number of rows and columns
            constraint_mat.num_row_ = constraint_mat.num_col_; // so that when the matrix is used by HFactor
            constraint_mat.num_col_ = temp_old_num_row; // it received the constraint matrix "column wise"
            this->B_.setup(constraint_mat, this->basis_idxs_); // where each column is a constraint. its inverse transpose will have as columns the nullspace basis
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
    this->loc_grad_.clear(); // is this clearing memory efficient?
    this->Q_.product(this->solution_.col_value, this->loc_grad_);
    for (HighsInt i {0}; i < this->lp_.num_col_; i++){ // add g to Q x_k
        this->loc_grad_[i] += this->lp_.col_cost_[i];
    }
}

void ActiveSetData::computeRedGrad(){
    this->red_grad_.clear(); // is this clearing memory efficient?
    for (size_t i {0}; i < this->ZT_.size(); i++){
        double sum {0};
        for (HighsInt j {0}; j < this->lp_.num_col_; j++){
            sum += this->ZT_[i][j] * this->loc_grad_[j];
        }
        this->red_grad_.push_back(sum);// Z^T (g + Q x_k)
    }
}
void ActiveSetData::price(){
    // compute pricing for each active constraint: Y^T (g + Q x_k)
    computeLocGrad();
    this->pricing_.clear();
    // loop over all basis and only count values for the active ones
    HighsInt countActive {0};
    for (size_t i {0}; i < this->basis_status_.size(); i++){
        if (this->basis_status_[i]!=AsmBasisStatus::kFreeInBasis){ // we already are in the basis, we need to check it's active
            // Equality is expected less frequent, should this check be moved down?
            if (this->basis_status_[i]==AsmBasisStatus::kEquality) this->pricing_.push_back(0.);
            else {
            std::vector<double> yt_row(this->lp_.num_col_);
            yt_row[i] = 1.; // select column i of B^{-T}
            this->B_.btranCall(yt_row);
            double sum {0.};
            for (HighsInt j {0}; j < this->lp_.num_col_; j ++){
                sum += yt_row[j] * this->loc_grad_[j];
            }
            if (this->basis_status_[i]==AsmBasisStatus::kLower) this->pricing_.push_back(sum);
            else if (this->basis_status_[i]==AsmBasisStatus::kUpper) this->pricing_.push_back(-sum);
            }
        }
    }
    printvector(this->pricing_);
}
// deactivate a constraint
HighsModelStatus ActiveSetData::deactivate(){
    // should we check that there is at least one active constraint? or is it guaranteed here?
    price();
    // loop through basis elements
    HighsInt chosen {0};
    double value_chosen {this->pricing_[0]};
    for (size_t i {1}; i < this->basis_idxs_.size(); i++){
        if (this->pricing_[i] > value_chosen){
            chosen = i;
            value_chosen = this->pricing_[i];
        }
    }
    if (value_chosen >= 0.) return HighsModelStatus::kOptimal;// if all prices are non negative, nothing to deactivate
    // else
    this->basis_idxs_.erase(this->basis_idxs_.begin() + chosen); // remove index
    this->basis_idxs_.push_back(chosen); // and place it back at the end of it
    this->basis_status_.erase(this->basis_status_.begin() + chosen); // remove status
    this->basis_status_.push_back(AsmBasisStatus::kFreeInBasis); // add the free in basis status
    // update basis inverse by extracting relevant vector from Y and moving it to Z
    // TODO
    return HighsModelStatus::kNotset;
}