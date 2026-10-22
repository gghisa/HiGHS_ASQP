/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "qpsolver/g_solver.hpp"

HighsStatus AsmSolver::run(){
    if ( this->feasibility() != HighsModelStatus::kNotset ) return this->status_;
    while ( this->keepLooping() ){
        // compute search direction
        if ( this->atFSEP_ ){
            this->computeRelaxedDirection();
            if ( this->isOptimal() ) break;
        } else this->computeReducedDirection();
        // if step is too small, we consider to be at FSEP
        //if ( this->norm(step_) < this->options_.primal_feasibility_tolerance ){
        //    this->atFSEP_ = true;
        //    continue;
        //}
        // perform ratio test
        this->ratiotest();
        if ( this->isUnbounded() ) break;
        // (potentially) update factorisations
        this->doUpdates();
    }
    // outside loop but run only if feasibility is successful:
    this->AsmToHighs();
    return this->getHighsStatus();
}

void AsmSolver::doUpdates(){
    if ( this->atFSEP_ ) {
        this->stepAlreadyTaken_ = false;
        if ( this->newactive_idx_ > - 1){ // if a constraint is activated
            if ( this->nullsp_dim_ == 0) this->replace(); // stays at FSEP
            else { // update factorisations of both B and M
                this->extend();
                this->activate();
                this->atFSEP_ = false;
            }
        } else {
            this->extend();
            this->atFSEP_ = false;
        }
        this->computeReducedVecs();
    } else {
        if ( this->newactive_idx_ > -1){
            this->activate();
            this->stepAlreadyTaken_ = false;
            this->updateRedGrad();
        } else { // no need to update reduced gradient, only pricing
            this->atFSEP_ = true;
            this->stepAlreadyTaken_ = true;
            this->computeReducedVecs();
        }
    }
    //if ( this->num_basis_updates_ == this->reinversion_freq_ ) this->reinvertBasis();
    this->updateObjective();
    this->logging();
    // book keeping
    this->relaxed_idx_ = -1;
    this->relaxed_iloc_ = -1;
    this->newactive_idx_ = -1;
    this->newactive_status_ = AsmBasisStatus::kInactive;
    this->info_.qp_iteration_count++;
}

void AsmSolver::findBestPrice(double& bestmultiplier, double& bestpricesign){
    double bestprice = - this->options_.dual_feasibility_tolerance;
    double price {0.}, sign {0.};
    bestpricesign = 0.; // TODO messy initialisations and temp variables
    for (HighsInt i {0}; i < this->rangsp_dim_; i++){ // loop through active constraints only
        HighsInt idx = this->basis_idxs_[i];
        // sign Dantzig prices on the go
        if (idx < this->lp_.num_row_) sign = static_cast<double>( this->con_status_[idx] );
        else sign = static_cast<double>( this->var_status_[idx - this->lp_.num_row_] );
        assert(sign < 1.1); // either upper, equality, or lower
        price = sign * this->pricing_[i];
        if ( price < bestprice ){
            bestpricesign = sign;
            bestprice = price;
            this->relaxed_idx_ = idx;
            this->relaxed_iloc_ = i;
        }
    }
    bestmultiplier = bestpricesign * bestprice;
    return;
}

void AsmSolver::computeRelaxedDirection(){
    double bestmultiplier, bestpricesign;
    this->findBestPrice(bestmultiplier, bestpricesign);
    if ( this->relaxed_idx_ == -1 ){
        this->model_status_ = HighsModelStatus::kOptimal; // set to optimal to break the major loop
        return;
    } else { 
        this->yQytemp_ = 0.; // buffer for y_p^T Q y_p
        this->Rtemp_.assign(this->Q_.dim_, 0.); // buffer for the solution of L x = Z^T Q y_p
        // extract yp
        this->buildZtemp( this->basis_perm_[ this->relaxed_iloc_ ] ); // y_p in this->Ztemp_
        this->step_ = this->Ztemp_.array; // Permutation happens before btran called in buildZtemp()
        this->Q_.product(this->step_, this->Rtemp_); // Q y_p
        // compute direction
        if ( this->nullsp_dim_ > 0 ){
            this->HFtran(this->Rtemp_); // B^{-1} Q y_p ~ Z^T Q y_p (in memory)
            this->yQytemp_ = this->Rtemp_[ this->relaxed_iloc_ ]; // extract y_p^T Q y_p for later
            std::fill(this->Rtemp_.begin(), this->Rtemp_.end() - this->nullsp_dim_, 0.); // [ 0 | M^{-1} Z^T Q y_p ]
            this->Lsolve(this->Rtemp_); // L^{-1} Z^T Q y_p
            std::vector<double> vec = this->Rtemp_;
            this->LTsolve(vec); // M^{-1} Z^T Q y_p
            this->HBtran(vec); // B^{-T} [ 0 | M^{-1} Z^T Q y_p ] = Z M^{-1} Z^T Q y_p
            for (HighsInt i {0}; i < this->Q_.dim_; i++) this->step_[i] -= vec[i]; // ( I - Z M^{-1} Z^T Q ) y_p
        } else // compute y_p^T Q y_p for later
            for (HighsInt i {0}; i < this->Q_.dim_; i++) this->yQytemp_ += this->step_[i] * this->Rtemp_[i];
        // compute step
        double tol = 10 * this->options_.factor_pivot_tolerance; // TODO use a proper number
        if ( std::abs(this->yQytemp_) <= tol){
            this->yQytemp_ = 0;
            this->alpha_relaxed_ = kHighsInf;
            for (HighsInt i {0}; i < this->Q_.dim_; i++){
                this->step_[i] *= bestpricesign;
                this->newvarvals_[i] = this->solution_.col_value[i] + this->step_[i];
            }
        } else if ( this->yQytemp_ < - tol ) std::cout<<"Negative curvature search direction\n"; // TODO
        else {
            double alpha_min = - bestmultiplier / this->yQytemp_;
            for (HighsInt i {0}; i < this->Q_.dim_; i++){
                this->step_[i] *= alpha_min;
                this->newvarvals_[i] = this->solution_.col_value[i] + this->step_[i];
            }
            this->alpha_relaxed_ = 1.;
        }
    }
    // logging
    highsLogDev(this->options_.log_options, HighsLogType::kDetailed,
                "    Relaxed index %d active at %s\n", this->relaxed_idx_, AsmStatusToString(this->relaxed_idx_).c_str());
    return;
}

void AsmSolver::computeReducedDirection(){ // solve Equality Problem
    if ( this->stepAlreadyTaken_ ) this->recomputeRedHessian(); // TODO keep?
    for (HighsInt i {0}; i < this->nullsp_dim_; i++)
        this->step_[this->rangsp_dim_ + i] = - this->red_grad_[i]; // TODO, is there a better place to flip sign?
    this->Lsolve(this->step_);
    this->LTsolve(this->step_);
    // then compute full space step
    std::fill(this->step_.begin(), this->step_.begin() + this->rangsp_dim_, 0.);
    this->HBtran(this->step_);
    // and update newvarvals
    for (HighsInt i {0}; i < this->Q_.dim_; i++)
        this->newvarvals_[i] = this->solution_.col_value[i] + this->step_[i];
    this->alpha_relaxed_ = 1.; // will look for lower alphas than unity only, since step is properly scaled
    return;
}

void AsmSolver::activate(){
    HighsInt loc_remove {-1};
    assert(this->getAsmBasisStatus(this->newactive_idx_) == AsmBasisStatus::kFreeInBasis ||
           this->getAsmBasisStatus(this->newactive_idx_) == AsmBasisStatus::kInactive);
    // whether the constraint is padding or inactive, check whether the existing padding already has the correct vector
    if ( this->getAsmBasisStatus(this->newactive_idx_) == AsmBasisStatus::kFreeInBasis ){
        assert( this->newactive_idx_ >= this->lp_.num_row_ ); // it should be a unit vector
        for ( loc_remove = this->rangsp_dim_; loc_remove < this->Q_.dim_; loc_remove++){
            if ( this->basis_idxs_[ loc_remove ] == this->newactive_idx_ ){ // unit vector already in basis
                // update factorization if the unit vector that is activated is already-in-basis
                this->reduceInBasis( loc_remove - this->rangsp_dim_ );
                break;
            }
        }
    } else { // if we are activating a constraint or the variable bound we are activating is not already in V
        this->reducePadding(this->newactive_idx_, loc_remove); // yields location of padding vector to be removed from basis_idxs_
        this->changeStatus( this->basis_idxs_[loc_remove], AsmBasisStatus::kInactive );
        this->basis_idxs_[loc_remove] = this->newactive_idx_; // update index
    }
    this->changeStatus(this->newactive_idx_, this->newactive_status_);
    if ( loc_remove > this->rangsp_dim_ ) // perform move if element isnt already neighbouring active indices
        this->fromPaddingToActive(loc_remove);
    this->removeNullSpaceDim();
    // logging
    highsLogDev(this->options_.log_options, HighsLogType::kDetailed,
                "    Activated index %d at %s\n", this->newactive_idx_, AsmStatusToString(this->newactive_status_).c_str());
    return;
}