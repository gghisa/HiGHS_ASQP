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
        if ( norm(this->red_grad_) > this->options_.primal_feasibility_tolerance ) this->minorloop(); // in case nullspace is non-empty to start with
        while ( this->maximalStepNotTaken() && !( this->iterlimit() || this->timelimit() || this->nullsizelimit() ) ) { // major iterations
            this->relaxAndSearch();
            if ( this->isoptimal() ) break;
            this->minorloop();
            // if ( this->num_basis_updates_ > this->reinversion_freq_) reinvertBasis();
        }
        // outside loop but run only if feasibility is successful:
        std::cout<<this->objective_<<" iterations: "<<this->info_.qp_iteration_count<<" time: "<<this->timer_.read()<<"\n";
    }
    return this->getHighsStatus();
}

void AsmSolver::findBestPrice(HighsInt& bestidx, double& bestmultiplier, HighsInt& bestloc){
    double bestprice = - this->options_.dual_feasibility_tolerance;
    double price {0.}, sign {0.}, bestpricesign {0.}; // TODO messy initialisations and temp variables
    for (HighsInt i {0}; i < this->rangsp_dim_; i++){ // loop through active constraints only
        HighsInt idx = this->basis_idxs_[i];
        // sign Dantzig prices on the go
        if (idx < this->lp_.num_row_) sign = static_cast<double>( this->con_status_[idx] );
        else sign = static_cast<double>( this->var_status_[idx - this->lp_.num_row_] );
        price = sign * this->pricing_[i];
        if ( price < bestprice ){
            bestpricesign = sign;
            bestprice = price;
            bestidx = idx; // TODO use idx
            bestloc = i;
        }
    }
    bestmultiplier = bestpricesign * bestprice;
    return;
}

void AsmSolver::computeSearchDir(const HighsInt& bestloc, const double& bestmultiplier){
    double alpha_min = 0.; // use first as buffer for y_p^T Q y_p
    // extract yp
    this->step_.assign(this->Q_.dim_, 0.);
    this->step_[ this->basis_perm_[bestloc] ] = 1.;
    this->B_.btranCall(this->step_); // y_p
    std::vector<double> vec(this->Q_.dim_);
    this->Q_.product(this->step_, vec); // Q y_p
    for (HighsInt i {0}; i < this->Q_.dim_; i++) alpha_min += this->step_[i] * vec[i]; // compute y_p^T Q y_p for later
    // compute direction
    if ( this->nullsp_dim_ > 0 ){
        this->HFtran(vec); // B^{-1} Q y_p
        std::vector<double> redbuffer(vec.end() - this->nullsp_dim_, vec.end()); // Z^T Q y_p
        LLTsolve(redbuffer); // M^{-1} Z^T Q y_p
        std::fill(vec.begin(), vec.end() - this->nullsp_dim_, 0.); // [ 0 | ? ]
        std::copy(redbuffer.begin(), redbuffer.end(), vec.end() - this->nullsp_dim_); // [ 0 | M^{-1} Z^T Q y_p ]
        this->HBtran(vec); // B^{-T} [ 0 | M^{-1} Z^T Q y_p ] = Z M^{-1} Z^T Q y_p
        for (HighsInt i {0}; i < this->Q_.dim_; i++) this->step_[i] -= vec[i]; // y_p ( I - Z M^{-1} Z^T Q y_p )
    }
    // compute step
    if ( std::abs(alpha_min) < this->options_.factor_pivot_tolerance ) std::cout<<"Zero search direction"; // TODO
    if ( alpha_min < - this->options_.factor_pivot_tolerance ) std::cout<<"Negative curvature search direction"; // TODO
    alpha_min = - bestmultiplier / alpha_min;
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        this->step_[i] *= alpha_min;
        this->newvarvals_[i] = this->solution_.col_value[i] + this->step_[i];
    }
}

void AsmSolver::relaxAndSearch(){ // loop through prices to find a constraint to deactivate
    HighsInt bestidx {-1}, bestloc;
    double bestmultiplier;
    this->findBestPrice(bestidx, bestmultiplier, bestloc);
    if ( bestidx == -1 ) this->model_status_ = HighsModelStatus::kOptimal; // set to optimal to break the major loop
    else {
        // update status of relaxed constraint
        if (bestidx < this->lp_.num_row_) this->con_status_[bestidx] = AsmBasisStatus::kFreeInBasis;
        else this->var_status_[bestidx - this->lp_.num_row_] = AsmBasisStatus::kFreeInBasis;
        //
        this->computeSearchDir(bestloc, bestmultiplier);
        HighsInt newactiveidx {-1};
        AsmBasisStatus newactivestatus;
        this->ratiotest(newactiveidx, newactivestatus);
        if ( newactiveidx > - 1){
            // replace relaxed with new one, update B factors only if we are going from vertex to vertex
            if ( this->nullsp_dim_ == 0){
                this->replace( this->basis_perm_[bestloc], bestidx, newactiveidx );
                this->basis_idxs_[ bestloc ] = newactiveidx; // update basis indices
            } else { // update factorisations of both B and M
                this->extend( bestloc, bestidx ); // takes care of basis_idxs_ too
                this->activate( newactiveidx, newactivestatus ); // takes care of basis_idxs_ too
            }
            // update status of new active constraint
            if (newactiveidx < this->lp_.num_row_) this->con_status_[newactiveidx] = newactivestatus;
            else this->var_status_[newactiveidx - this->lp_.num_row_] = newactivestatus;
        } else {
            this->extend( bestloc, bestidx ); // update factorization(s)
        }
        this->updateObjective();
        this->computeReducedVecs(); // TODO recomputing local gradient may not be necessary
        this->info_.qp_iteration_count++;
        std::cout<<this->objective_<<" - "<< this->nullsp_dim_<<"\n"<<std::flush;
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
    std::copy(this->delta_.begin(), this->delta_.end(), this->step_.end() - this->delta_.size());
    this->HBtran(this->step_);
    // and update newvarvals
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        this->newvarvals_[i] = this->solution_.col_value[i] + this->step_[i];
    }
}

void AsmSolver::ratiotest(HighsInt& newactive_idx, AsmBasisStatus& newactive_status){
    // assumes this->newvarvals are already computed (due to differences between major and minor loop)
    this->lp_.a_matrix_.product(this->newconvals_, this->newvarvals_); // a_i^T x_{k+1}
    this->lp_.a_matrix_.product(this->newconpivots_, this->step_); // a_i^T \s
    ratiotest_pass1(); // ratio test on relaxed instance
    if ( this->alpha_relaxed_ < 1.){ // TODO time saving if step is null?
        // if we meet a constraint
        ratiotest_pass2(newactive_idx, newactive_status);
        this->compute_varvals(this->alpha_, this->solution_.col_value);
        this->lp_.a_matrix_.product(this->solution_.row_value, this->solution_.col_value); // a_i^T x_{k+1}
    } else {
        this->solution_.row_value = this->newconvals_; // don't recompute new constraint values
        this->solution_.col_value = this->newvarvals_; // nor variables' values either
    }
    return;
}

void AsmSolver::minorloop(){
    while ( norm(this->red_grad_) > this->options_.primal_feasibility_tolerance ){ // if nullsp is empty then norm returns 0
        this->solveEP();
        HighsInt newactiveidx {-1};
        AsmBasisStatus newactivestatus;
        this->ratiotest(newactiveidx, newactivestatus);
        if ( newactiveidx > -1 ) this->activate(newactiveidx, newactivestatus);
        this->updateObjective();
        this->computeReducedVecs(); // red grad needs updating with new position
        this->info_.qp_iteration_count++;
        std::cout<<this->objective_<<" - "<< this->nullsp_dim_<<"\n"<<std::flush;
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