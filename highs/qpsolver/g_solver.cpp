/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "qpsolver/g_solver.hpp"

HighsStatus AsmSolver::run(){
    this->feasibility();
    if ( this->model_status_ == HighsModelStatus::kOptimal ){
        this->model_status_ = HighsModelStatus::kNotset;
        this->minorloop(); // in case nullspace is non-empty to start with
        while ( this->maximalStepNotTaken() && !( this->iterlimit() || this->timelimit() || this->nullsizelimit() ) ) { // major iterations
            this->relaxAndSearch();
            if ( this->isoptimal() ) break;
            // if ( this->num_basis_updates_ > this->reinversion_freq_) reinvertBasis();
            if (this->alpha_relaxed_ < 1.) this->minorloop();
            std::cout<<this->objective_<<" - "<< this->nullsp_dim_<<"\n"<<std::flush;
        }
        // outside loop but run only if feasibility is successful:
        std::cout<<this->objective_<<" iterations: "<<this->info_.qp_iteration_count<<" time: "<<this->timer_.read()<<"\n";
    }
    return this->getHighsStatus();
}

void AsmSolver::relaxAndSearch(){ // loop through prices to find a constraint to deactivate
    signPrices();
    HighsInt bestidx {-1}, bestloc {-1};
    double bestprice = - this->options_.dual_feasibility_tolerance;
    for (HighsInt i {0}; i < this->rangsp_dim_; i++){ // loop through active constraints only
        if ( this->pricing_[i] < bestprice ){
            bestprice = this->pricing_[i];
            bestidx = this->basis_idxs_[i];
            bestloc = i;
        }
    }
    if ( bestidx == -1 ) this->model_status_ = HighsModelStatus::kOptimal; // set to optimal to break the major loop
    else {
        // this->extend( this->basis_perm_[bestloc], bestidx ); // update factorization(s)
        // send deactivated constraint to the end of free-in-basis constraints
        //std::vector<HighsInt>::iterator it = this->basis_idxs_.begin() + bestloc;
        //std::rotate(it, it + 1, this->basis_idxs_.end());
        //it = this->basis_perm_.begin() + bestloc;
        //std::rotate(it, it + 1, this->basis_perm_.end());
        //this->addNullSpaceDim();
        double alpha_min = 0.; // use first as buffer for y_p^T Q y_p
        // extract yp
        this->step_.assign(0, this->Q_.dim_);
        this->step_[ this->basis_perm_[bestloc] ] = 1.;
        this->B_.btranCall(this->step_);
        // compute direction
        if ( this->nullsp_dim_ > 0 ){
            std::vector<double> vec(this->Q_.dim_);
            this->Q_.product(this->step_, vec); // Q y_p
            for (HighsInt i {0}; i < this->Q_.dim_; i++) alpha_min += this->step_[i] * vec[i]; // compute y_p^T Q y_p for later
            this->HFtran(vec); // B^{-1} Q y_p
            std::vector<double> redbuffer(vec.end() - this->nullsp_dim_, vec.end()); // Z^T Q y_p
            LLTsolve(redbuffer); // M^{-1} Z^T Q y_p
            std::fill(vec.begin(), vec.end() - this->nullsp_dim_, 0.); // [ 0 | ? ]
            std::copy(redbuffer.begin(), redbuffer.end(), vec.end() - this->nullsp_dim_); // [ 0 | M^{-1} Z^T Q y_p ]
            this->HBtran(vec); // B^{-T} [ 0 | M^{-1} Z^T Q y_p ] = M^{-1} Z^T Q y_p
            for (HighsInt i {0}; i < this->Q_.dim_; i++) this->step_[i] -= vec[i]; // y_p ( I - M^{-1} Z^T Q y_p )
        }
        // compute step
        if ( std::abs(alpha_min) < this->options_.factor_pivot_tolerance ) std::cout<<"Zero search direction"; // TODO
        if ( alpha_min < - this->options_.factor_pivot_tolerance ) std::cout<<"Negative curvature search direction"; // TODO
        alpha_min = - bestprice / alpha_min;
        for (HighsInt i {0}; i < this->Q_.dim_; i++){
            this->newvarvals_[i] = this->solution_.col_value[i] + alpha_min * this->step_[i];
        }
        // finally return price to original value to update reduced gradient, then update status
        if (bestidx < this->lp_.num_row_){ 
            bestprice *= static_cast<double>( this->con_status_[bestidx] );
            this->con_status_[bestidx] = AsmBasisStatus::kFreeInBasis;
        } else {
            HighsInt var_idx = bestidx - this->lp_.num_row_;
            bestprice *= static_cast<double>( this->var_status_[var_idx] );
            this->var_status_[var_idx] = AsmBasisStatus::kFreeInBasis;
        }
        // make redgrad and pricing ready for basis factorization and potential update
        this->red_grad_.push_back(bestprice);
        this->pricing_.erase(this->pricing_.begin() + bestloc);
        //
        takeStep();
    }
    return;
}

void AsmSolver::ratio1(const double tol, const double denom, const double lower,
                       const double upper, const double oldval, const double newval, double& alpha){
    double bound;
    if (denom < - tol && lower > newval ) bound = lower;
    else if ( denom > tol && upper < newval ) bound = upper;
    else return;
    alpha = std::min( alpha, ( bound - oldval ) / denom );
    return;
}

void AsmSolver::ratiotest_pass1(){
    this->alpha_relaxed_ = 1.; // we want to minimise it
    const double tol = 10 * this->options_.factor_pivot_tolerance;
    for (HighsInt i {0}; i < this->Q_.dim_; i++) // loop through variables
        this->ratio1(tol, this->step_[i], this->lp_relaxed_.col_lower_[i], this->lp_relaxed_.col_upper_[i],
                     this->solution_.col_value[i], this->newvarvals_[i], this->alpha_relaxed_);
    for (HighsInt i {0}; i < this->lp_.num_row_; i++) // loop through constraints
        this->ratio1(tol, this->newconpivots_[i], this->lp_relaxed_.row_lower_[i], this->lp_relaxed_.row_upper_[i],
                     this->solution_.row_value[i], this->newconvals_[i], this->alpha_relaxed_);
    return;
}

void AsmSolver::ratio2(double& max_pivot, const double denom, const double lower, const double upper,
                       const double oldval, const double newval, const double alphamax, double& alpha,
                       const HighsInt idx, HighsInt& newactive_idx, AsmBasisStatus& newactive_status){
    double bound;
    if ( denom < - max_pivot ) bound = lower;
    else if ( denom > max_pivot ) bound = upper;
    else return;
    double alpha_here = ( bound - oldval ) / denom;
    if ( denom < - max_pivot && alpha_here < alphamax ){
        newactive_idx = idx;
        newactive_status = AsmBasisStatus::kLower;
        max_pivot = - denom;
        alpha = alpha_here;
    } else if ( denom > max_pivot && alpha_here < alphamax ){
        newactive_idx = idx;
        newactive_status = AsmBasisStatus::kUpper;
        max_pivot = denom;
        alpha = alpha_here;
    }
}

void AsmSolver::ratiotest_pass2(HighsInt& newactive_idx, AsmBasisStatus& newactive_status){
    double max_pivot = 0; // to ensure no division by 0 in ratio2
    for (HighsInt i {0}; i < this->Q_.dim_; i++) // loop through variables
        this->ratio2(max_pivot, this->step_[i], this->lp_.col_lower_[i], this->lp_.col_upper_[i],
                     this->solution_.col_value[i], this->newvarvals_[i], this->alpha_relaxed_, this->alpha_,
                     i + this->lp_.num_row_, newactive_idx, newactive_status);
    for (HighsInt i {0}; i < this->lp_.num_row_; i++) // loop through constraints
        this->ratio2(max_pivot, this->newconpivots_[i], this->lp_.row_lower_[i], this->lp_.row_upper_[i],
                     this->solution_.row_value[i], this->newconvals_[i], this->alpha_relaxed_, this->alpha_,
                     i, newactive_idx, newactive_status);
    if ( max_pivot <= this->options_.factor_pivot_tolerance){
        std::cout<<"Second pass not activating any constraint!"<<std::flush;
        throw std::logic_error("Second pass not activating any constraint!");
    }
    if ( this->alpha_ < 0 ) this->alpha_ = 0.;
    return;
}

void AsmSolver::solveEP(){ // solve Equality Problem
    this->delta_.resize(this->red_grad_.size());
    for (size_t i {0}; i < this->red_grad_.size(); i++){
        this->delta_[i] = - this->red_grad_[i]; // TODO, is there a better place to flip sign?
    }
    this->LLTsolve(this->delta_);
    // then compute full space step
    std::fill(this->step_.begin(), this->step_.end() - this->delta_.size(), 0.);
    std::copy(this->delta_.begin(), this->delta_.end(), this->buffer_.end() - this->delta_.size());
    this->HBtran(this->step_);
    // and update newvarvals
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
            this->newvarvals_[i] = this->solution_.col_value[i] + this->step_[i];
    }
}

void AsmSolver::takeStep(){
    // ratio test vectors
    // assumes this->newvarvals are already computed (due to differences between major and minor loop)
    this->lp_.a_matrix_.product(this->newconvals_, this->newvarvals_); // a_i^T x_{k+1}
    this->lp_.a_matrix_.product(this->newconpivots_, this->step_); // a_i^T \s
    ratiotest_pass1(); // ratio test on relaxed instance
    if ( this->alpha_relaxed_ < 1.){ // TODO time saving if step is null?
        // if we meet a constraint
        HighsInt newactive_idx;
        AsmBasisStatus newactive_status;
        ratiotest_pass2(newactive_idx, newactive_status);
        this->compute_varvals(this->alpha_, this->solution_.col_value);
        this->lp_.a_matrix_.product(this->solution_.row_value, this->solution_.col_value); // a_i^T x_{k+1}
        this->activate(newactive_idx, newactive_status);
    } else { // if no constraint activated and we take the full step
        this->solution_.row_value = this->newconvals_; // don't recompute new constraint values
        this->solution_.col_value = this->newvarvals_; // nor variables' values either
    }
    this->updateObjective();
    this->computeReducedVecs(); // red grad needs updating with new position
    this->info_.qp_iteration_count++;
    return;
}

void AsmSolver::minorloop(){
    double OPCS { false };
    while ( !OPCS ){
        solveEP();
        takeStep();
        if ( this->alpha_relaxed_ < 1 ) extend(900000, 900000); // TODO
        else OPCS = true;
    }
    return;
}

void AsmSolver::activate(const HighsInt& idx, const AsmBasisStatus& status){
    // we keep track of constraints free in basis for book-keeping purposes, even though
    // the V part of B is made up of arbitrary unit vectors
    HighsInt loc_remove {-1};
    HighsInt varidx = idx - this->lp_.num_row_; // possibly unused, otherwise reused many times
    // handle status update
    if (idx < this->lp_.num_row_){
        if ( this->lp_.row_lower_[idx] == this->lp_.row_upper_[idx] ) this->con_status_[idx] = AsmBasisStatus::kEquality;
        else this->con_status_[idx] = status;
    } else {
        if ( this->lp_.col_lower_[varidx] == this->lp_.col_upper_[varidx] ) this->var_status_[varidx] = AsmBasisStatus::kEquality;
        else this->var_status_[varidx] = status;
    }
    // now that statuses have been taken care of, we have to choose what to do
    // 1. if we are activating a unit vector, we check if it is already in V. If yes, just update perm and idxs, else update factorisations
    // 2. if we are activating a constraint, update factorisations
    // by update factorisations we mean updating B and L, the latter by choosing wisely which element to drop.
    if ( idx >= this->lp_.num_row_){ // if we are activating a variable's bound
        for (loc_remove = 0; loc_remove < this->nullsp_dim_; loc_remove++){
            if ( this->Vi_[loc_remove] == varidx ){ // unit vector already in basis
                HighsInt loc_actual = this->rangsp_dim_ + loc_remove;
                this->basis_idxs_[loc_actual] = idx; // update index
                auto it = this->basis_idxs_.begin();
                std::rotate(it + this->rangsp_dim_, it + loc_actual, it + loc_actual + 1);
                it = this->basis_perm_.begin();
                std::rotate(it + this->rangsp_dim_, it + loc_actual, it + loc_actual + 1);
                // update factorization if the already-in-basis
                this->reduceInBasis(loc_remove);
                this->Vi_.erase( this->Vi_.begin() + loc_remove ); // remove reference to element in padding
                return;
            }
        }
    }
    // if we are activating a constraint or the variable bound we are activating is not already in V
    this->reduceOutsideBasis(idx);
    return;
}
