/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "qpsolver/g_solver.hpp"

void AsmSolver::ratio1(const double tol, const double denom, const double lower,
                       const double upper, const double oldval, const double newval, double& alpha){
    double bound; // checking bounds arent inf may avoid FLOPs below
    if ( denom < - tol && lower > -kHighsInf ) bound = lower;
    else if ( denom > tol && upper < kHighsInf ) bound = upper;
    else return;
    alpha = std::min( alpha, ( bound - oldval ) / denom );
    return;
}

bool AsmSolver::ratiotest_pass1(){
    double alphatemp = this->alpha_relaxed_;
    const double tol = 10 * this->options_.factor_pivot_tolerance;
    for (HighsInt i {0}; i < this->Q_.dim_; i++) // loop through variables
        this->ratio1(tol, this->step_[i], this->lp_relaxed_.col_lower_[i], this->lp_relaxed_.col_upper_[i],
                     this->solution_.col_value[i], this->newvarvals_[i], this->alpha_relaxed_);
    for (HighsInt i {0}; i < this->lp_.num_row_; i++) // loop through constraints
        this->ratio1(tol, this->newconpivots_[i], this->lp_relaxed_.row_lower_[i], this->lp_relaxed_.row_upper_[i],
                     this->solution_.row_value[i], this->newconvals_[i], this->alpha_relaxed_);
    if ( alpha_relaxed_ != alphatemp ){
        return true; // a constraint has been hit
    } else if ( this->alpha_relaxed_ == kHighsInf ) this->model_status_ = HighsModelStatus::kUnbounded;
    // case in which alpha_relaxed == 1 after ratio test is here
    return false;
}

void AsmSolver::ratio2(double& max_pivot, const double denom, const double lower, const double upper,
                       const double oldval, const double alphamax, double& alpha,
                       const HighsInt idx, HighsInt& newactive_idx, AsmBasisStatus& newactive_status){
    double bound;
    if ( denom < - std::abs( max_pivot ) ) bound = lower;
    else if ( denom > std::abs( max_pivot ) ) bound = upper;
    else return;
    double alpha_here = ( bound - oldval ) / denom;
    if ( alpha_here <= alphamax ){
        newactive_idx = idx;
        max_pivot = std::abs( denom );
        alpha = alpha_here;
        if ( lower == upper ) newactive_status = AsmBasisStatus::kEquality;
        else if ( denom < 0 ) newactive_status = AsmBasisStatus::kLower;
        else newactive_status = AsmBasisStatus::kUpper;
    }
}

void AsmSolver::ratiotest_pass2(){
    double max_pivot = 0;
    for (HighsInt i {0}; i < this->Q_.dim_; i++) // loop through variables
        this->ratio2(max_pivot, this->step_[i], this->lp_.col_lower_[i], this->lp_.col_upper_[i],
                     this->solution_.col_value[i], this->alpha_relaxed_, this->alpha_,
                     i + this->lp_.num_row_, this->newactive_idx_, this->newactive_status_);
    for (HighsInt i {0}; i < this->lp_.num_row_; i++) // loop through constraints
        this->ratio2(max_pivot, this->newconpivots_[i], this->lp_.row_lower_[i], this->lp_.row_upper_[i],
                     this->solution_.row_value[i], this->alpha_relaxed_, this->alpha_,
                     i, this->newactive_idx_, this->newactive_status_);
    assert(max_pivot > this->options_.factor_pivot_tolerance);
    this->alpha_ = std::max( this->alpha_, 0. );
    return;
}

void AsmSolver::cleanOutActive(){ // set zero movement for those active constraints or bounds
    for (HighsInt i {0}; i < this->rangsp_dim_; i++){
        if ( this->relaxed_iloc_ != i ){ // if we are not at the relaxed point
            if ( this->basis_idxs_[i] < this->lp_.num_row_ ) this->newconpivots_[ this->basis_idxs_[i] ] = 0.;
            else this->step_[ this->basis_idxs_[i] - this->lp_.num_row_ ] = 0.;
        }
    }
}

void AsmSolver::ratiotest(){
    // assumes this->newvarvals are already computed (due to differences between major and minor loop)
    this->lp_.a_matrix_.product(this->newconvals_, this->newvarvals_); // a_i^T x_{k+1}
    this->lp_.a_matrix_.product(this->newconpivots_, this->step_); // a_i^T \s
    this->cleanOutActive();
    if ( this->ratiotest_pass1() ) this->ratiotest_pass2(); // if we meet a constraint
    else this->alpha_ = this->alpha_relaxed_;
    assert(this->getAsmBasisStatus(this->newactive_idx_) != this->newactive_status_); // cannot activate constraint at where it's already active
    // TODO unnecessary computations are done here if we are in minor loop
    // TODO is updating step_ to be done?
    // TODO what if alpha is null?
    this->compute_varvals(this->alpha_, this->solution_.col_value);
    this->lp_.a_matrix_.product(this->solution_.row_value, this->solution_.col_value); // a_i^T x_{k+1}
    return;
}