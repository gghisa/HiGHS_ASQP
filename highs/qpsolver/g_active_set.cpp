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
                            redhes_(Q),
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
                if (status[i] == HighsBasisStatus::kZero) this->redhes_.addOneNullSpaceDim();
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
    this->redhes_.init();
}
// setup basis matrix
void ActiveSetData::setupBasisMat(){
    HighsSparseMatrix constraint_mat = this->lp_.a_matrix_; // create a copy of the constraint matrix
    constraint_mat.ensureRowwise(); // flip the way in which it is stored
    constraint_mat.format_ = MatrixFormat::kColwise; // but "trick it" into thinking it is still stored columnwise
    HighsInt temp_old_num_row = constraint_mat.num_row_; // flip the number of rows and columns
    constraint_mat.num_row_ = constraint_mat.num_col_; // so that when the matrix is used by HFactor
    constraint_mat.num_col_ = temp_old_num_row; // it received the constraint matrix "column wise"
    this->redhes_.Hsetup(constraint_mat, this->basis_idxs_); // where each column is a constraint. its inverse transpose will have as columns the nullspace basis
    this->redhes_.Hbuild(); // factorize method
}

HighsInt ActiveSetData::getSizeNullSpace(){
    return this->redhes_.getNullSpaceSize();
}

HighsInt ActiveSetData::getSizeRangeSpace(){
    return this->lp_.num_col_ - getSizeNullSpace();
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

void ActiveSetData::computeRedGrad(){ // Z^T (g + Q x_k)
    computeLocGrad();
    std::vector<double> vec = this->loc_grad_;
    this->redhes_.Hftran(vec); // compute B x = g_k
    this->red_grad_.assign(vec.end() - getSizeNullSpace(), vec.end()); // extract last z elements of the result, i.e. Z^T (g + Q x_k)
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
    assert(getSizeNullSpace() != this->lp_.num_col_); // check that there is at least one active constraints (can be equality)
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
    this->redhes_.addOneNullSpaceDim();
    // TODO how is this change communicated to the matrix this->B_? TODO
    // recompute reduced hessian
    this->redhes_.extend();
    return HighsModelStatus::kNotset;
}
// activate a constraint
HighsModelStatus ActiveSetData::activate(){
    // compute step in full space
    return HighsModelStatus::kNotset;
}
// solve reduced equality problem
void ActiveSetData::solveEQ(){ // M d = red_grad_
    computeRedGrad();
    this->delta_ = this->red_grad_; // store info for solve
    this->redhes_.solve(this->delta_); // solve for delta (reduced step)
    this->step_.assign(this->Q_.dim_, 0.); // (re)initialise full space step
    // and copy information to it
    std::copy(this->delta_.begin(), this->delta_.end(), this->step_.begin() + (this->Q_.dim_ - getSizeNullSpace()));
    this->redhes_.Hbtran(this->step_); // extract product Z\delta
};

void ActiveSetData::compute_new_loc(const double& alpha, std::vector<double>& newloc){
    for (HighsInt i {0}; i < this->lp_.num_col_; i++){
        this->step_[i] *= alpha;
        this->alpha_ *= alpha;
        newloc[i] = this->solution_.col_value[i] + this->step_[i];
    } // compute x_{k+1}
}

void ActiveSetData::ratiotest(){
    // we keep a copy of location, whereas a temporary alpha will just be local to a broken constraint
    // at each constraint, if it is broken, we update alpha
    // with the final alpha being the product of all the alphas (so it is also update as we go)
    std::vector<double> newloc(this->lp_.num_col_);
    compute_new_loc(1., newloc);
    HighsInt newactive_idx = -1; // recall: numbering variables starts from nr of constraints
    // it may be more efficient to check bounds first (assume feasibility ofc), so we do that here
    for (HighsInt i {0}; i < this->lp_.num_col_; i++){ // loop through constraints
        HighsInt index_here = i + this->lp_.num_row_;
        if (this->lp_.col_lower_[i] > newloc[i]){
            // compute new alpha
            double alpha_here = ( newloc[i] - this->lp_.col_lower_[i] ) / this->step_[i];
            compute_new_loc(alpha_here, newloc); // update alpha and temporary location
            newactive_idx = index_here; // store new index
        } else if (this->lp_.col_upper_[i] < newloc[i]) {
            double alpha_here = ( this->lp_.col_upper_[i] - newloc[i] ) / this->step_[i];
            compute_new_loc(alpha_here, newloc);
            newactive_idx = index_here;
        }
    }
    // loop through inactive (inequality) constraints
    std::vector<double> convals(this->lp_.num_col_); // vector for constraint values
    this->lp_.a_matrix_.productTranspose(convals, this->solution_.col_value);
    std::vector<double> denoms(this->lp_.num_col_); // vectors for denominators of ratio test formula
    this->lp_.a_matrix_.productTranspose(denoms, this->step_);
    for (HighsInt i {0}; i < this->lp_.num_col_; i++){
        if (this->lp_.row_lower_[i] > convals[i]){
        } else if (this->lp_.row_upper_[i] < convals[i]) {}
    }
    // check degeneracy? TODO
};

bool ActiveSetData::isActiveInequality(const AsmBasisStatus& status){
    if (status == AsmBasisStatus::kLower ||
        status == AsmBasisStatus::kUpper) return true;
    else return false;
}

bool ActiveSetData::isInactive(const AsmBasisStatus& status){
    if (status == AsmBasisStatus::kFreeInBasis ||
        status == AsmBasisStatus::kInactive) return true;
    else return false;
}

double ActiveSetData::vec2norm(const std::vector<double> vector){ // useless?
    double sum {0.};
    for (size_t i {0}; i < vector.size(); i++){
        sum += vector[i] * vector[i];
    }
    return std::sqrt( sum );
};