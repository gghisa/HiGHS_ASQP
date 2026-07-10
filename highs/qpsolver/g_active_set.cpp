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
                            Q_(Q),
                            nullsp_dim_(0),
                            basis_idxs_(lp.num_col_),
                            basis_status_(lp.num_col_),
                            loc_grad_(lp.num_col_),
                            red_grad_(lp.num_col_),
                            pricing_(lp.num_col_){
    if (Q.format_ == HessianFormat::kTriangular) Q = Q.toSquare(); // make Hessian square to improve columns/rows accessing speed
    initAsmBasis(basis);
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
            std::vector<double> emptyvec(this->lp_.num_col_); // initialise with correct lenght TODO remove when removing explicit Z representation
            if (status[i] == HighsBasisStatus::kZero){
                assert(!isconstr); // constraints shouldnt be kZero
                index += this->lp_.num_row_;
                this->ZT_.push_back(emptyvec); // kZero means one available nullspace dimension
                this->nullsp_dim_ += 1;
            }
            else { // non free variables contribute to range space
                if (!isconstr) index +=  this->lp_.num_row_; // variable count starts from number of constraints
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
    this->B_.setup(constraint_mat, this->basis_idxs_); // where each column is a constraint. its inverse transpose will have as columns the nullspace basis
    this->B_.build(); // factorize method
    setupInvBasisSpace();
}

void ActiveSetData::setupInvBasisSpace(){ // eventually remove this explicit representation of Z TODO
    HighsInt n_active = this->lp_.num_col_ - this->nullsp_dim_; // wouldn't it be better to store this as a member of the class?
    std::vector<double> col(this->lp_.num_col_); // declare vector (column of Z, row of Z^T)
    for (HighsInt i {0}; i < this->nullsp_dim_ ; i++){
        col.assign(this->lp_.num_col_,0.); // (re)start unit vector
        col[n_active + i] = 1.; // set unit entry at the index for the desired column of B^{-T}
        this->B_.btranCall(col); // solve B^T\cdot e_i = col
        this->ZT_[i] = col; // extract copy for Z^T
    }
}

void ActiveSetData::setupReducedHessian(){ // this function assumes explicit representation of Z. it has to change TODO
    assert (this->Q_.format_ == HessianFormat::kSquare);
    assert (this->redhes_.empty());
    for (size_t i {0}; i < this->ZT_.size(); i++){// loop over the rows of the reduced hessian
        std::vector<double> empty_row_red_hessian(this->ZT_.size());
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
            this->redhes_[j][i] = sum; // off-diagonal symmetric element
            // i could check that second assignment only happens when i != j, but the check would run for nothing most of the time
        }
    }
}

void ActiveSetData::extendReducedHessian(){ // algo from Feldmeier thesis 
    // TODO requires having a factorization of the reduced hessian
    // so do we need to compute it explicitly to start with?
    // maybe we should factorize it to start with
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

void ActiveSetData::computeRedGrad(){
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
    this->B_.btranCall(this->pricing_);
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
    extendReducedHessian();
    return HighsModelStatus::kNotset;
}

bool ActiveSetData::isActiveInequality(const AsmBasisStatus& status){
    if (status == AsmBasisStatus::kLower ||
        status == AsmBasisStatus::kUpper) return true;
    else return false;
}