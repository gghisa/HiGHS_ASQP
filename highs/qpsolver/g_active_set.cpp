/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Highs.h"
#include "qpsolver/g_active_set.hpp"
#include "qpsolver/g_red_hessian.hpp"

ActiveSetData::ActiveSetData(const HighsLp& lp,
                             const HighsBasis& basis,
                             HighsSolution& solution,
                             HighsHessian& Q)
                            : lp_(lp),
                            solution_(solution),
                            Q_(Q),
                            redhes_(Q),
                            nullsp_dim_(0),
                            basis_idxs_(lp.num_col_),
                            basis_status_(lp.num_col_),
                            loc_grad_(lp.num_col_),
                            red_grad_(lp.num_col_),
                            pricing_(lp.num_col_){
    if (Q.format_ == HessianFormat::kTriangular) Q = Q.toSquare(); // make Hessian square to improve columns/rows accessing speed
    initAsmBasis(basis);
    setupBasisMat();
    this->redhes_.build();
};
// convert Asm Status to Highs
HighsBasisStatus ActiveSetData::AsmStatusToHighs(const AsmBasisStatus& status){
    if (status==AsmBasisStatus::kLower || status==AsmBasisStatus::kEquality) return HighsBasisStatus::kLower;
    else if (status==AsmBasisStatus::kFreeInBasis) return HighsBasisStatus::kZero;// should not happen with constraints but how do I check it?
    else if (status==AsmBasisStatus::kInactive) return HighsBasisStatus::kBasic;
    else return HighsBasisStatus::kNonbasic; // should never happen
};
// convert Highs Status to Asm Status
AsmBasisStatus ActiveSetData::HighsStatusToAsm(const HighsBasisStatus& status, HighsInt index){
    if (index >= this->lp_.num_row_){ // it is a variable
        index -= this->lp_.num_row_; // if equality this should be a fixed variable and needs presolve
        if (this->lp_.col_lower_[index] == this->lp_.col_upper_[index]) return AsmBasisStatus::kEquality;
    } else if (this->lp_.row_lower_[index] == this->lp_.row_upper_[index]) return AsmBasisStatus::kEquality;
    // if no equality is found:
    if(status == HighsBasisStatus::kLower) return AsmBasisStatus::kLower;
    else if(status == HighsBasisStatus::kUpper) return AsmBasisStatus::kUpper;
    else if(status == HighsBasisStatus::kZero) return AsmBasisStatus::kFreeInBasis;
    else return AsmBasisStatus::kInactive;
};
//
void ActiveSetData::initAsmBasisLoop(const std::vector<HighsBasisStatus>& status, const bool isconstr){
    for (size_t i {0}; i<status.size(); i++){ // loop through vector of statuses
        if (status[i] != HighsBasisStatus::kBasic){// inside the loop we deal with simplex nonbasic variables only
            HighsInt index {(HighsInt)i}; // declare index for HFactor basis_index
            if (!isconstr){ // constraints shouldnt be kZero
                index += this->lp_.num_row_; // variable count starts from number of constraints
                if (status[i] == HighsBasisStatus::kZero) this->nullsp_dim_ += 1;
            }
            this->basis_idxs_[i] = index; // add index
            this->basis_status_[i] = HighsStatusToAsm(status[i], index); // and add its status
        } // else basic variables are just inactive and don't come into play until ratio test
    }
}
// initialise basis data members
void ActiveSetData::initAsmBasis(const HighsBasis& basis){
    assert(basis.valid);
    // by looking at active constraints, initialise range and null spaces, and basis indices for HFactor
    initAsmBasisLoop(basis.row_status, true);
    initAsmBasisLoop(basis.col_status, false);
}
// setup basis matrix
void ActiveSetData::setupBasisMat(){
    HighsSparseMatrix constraint_mat = this->lp_.a_matrix_; // create a copy of the constraint matrix
    constraint_mat.ensureRowwise(); // flip the way in which it is stored
    constraint_mat.format_ = MatrixFormat::kColwise; // but "trick it" into thinking it is still stored columnwise
    HighsInt temp_old_num_row = constraint_mat.num_row_; // flip the number of rows and columns
    constraint_mat.num_row_ = constraint_mat.num_col_; // so that when the matrix is used by HFactor
    constraint_mat.num_col_ = temp_old_num_row; // it received the constraint matrix "column wise"
    this->redhes_.init(this->nullsp_dim_);
    this->redhes_.Hsetup(constraint_mat, this->basis_idxs_); // where each column is a constraint. its inverse transpose will have as columns the nullspace basis
    this->redhes_.Hbuild(); // factorize method
}

HighsInt ActiveSetData::getSizeNullSpace(){
    return this->nullsp_dim_;
}

HighsInt ActiveSetData::getSizeRangeSpace(){
    return this->lp_.num_col_ - this->nullsp_dim_;
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
    this->Q_.product(this->solution_.col_value, this->loc_grad_); // stores result in loc_grad_
    for (HighsInt i {0}; i < this->lp_.num_col_; i++){ // add g to Q x_k
        this->loc_grad_[i] += this->lp_.col_cost_[i];
    }
}

void ActiveSetData::computeRedGrad(){ // TODO change to Hftran
    this->red_grad_.assign(this->ZT_.size(), 0.);
    for (size_t i {0}; i < this->red_grad_.size(); i++){
        double sum {0};
        for (HighsInt j {0}; j < this->lp_.num_col_; j++){
            sum += this->ZT_[i][j] * this->loc_grad_[j];
        }
        this->red_grad_[i] = sum;// Z^T (g + Q x_k)
    }
}

void ActiveSetData::price(){
    // compute pricing for each basis element: B \lambda = (g + Q x_k)
    computeLocGrad();
    this->pricing_ = this->loc_grad_;
    this->redhes_.Hbtran(this->pricing_);
}
// deactivate a constraint
HighsModelStatus ActiveSetData::deactivate(){
    // should we check that there is at least one active constraint? or is it guaranteed here?
    price();
    // loop through basis elements
    assert(this->nullsp_dim_ != this->lp_.num_col_); // check that there is at least one active constraints (can be equality)
    HighsInt chosen {0};
    double value_chosen {this->pricing_[0]};
    for (HighsInt i {1}; i < this->lp_.num_col_; i++){ // loop through all elements of this->pricing_ or equivalently through all basis indices
        if (isActiveInequality(this->basis_status_[i]) && this->pricing_[i] > value_chosen){
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
    this->nullsp_dim_ += 1;
    // TODO how is this change communicated to the matrix this->B_? TODO
    // recompute reduced hessian
    this->redhes_.extend();
    return HighsModelStatus::kNotset;
}

bool ActiveSetData::isActiveInequality(const AsmBasisStatus& status){
    if (status == AsmBasisStatus::kLower ||
        status == AsmBasisStatus::kUpper) return true;
    else return false;
}