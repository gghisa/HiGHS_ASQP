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
            if ( this->isOptimal() ) break; // TODO change to return
        } else this->computeReducedDirection();
        // if step is too small, we consider to be at FSEP
        //if ( this->norm(step_) < this->options_.primal_feasibility_tolerance ){
        //    this->atFSEP_ = true;
        //    continue;
        //}
        // perform ratio test
        this->ratiotest();
        // (potentially) update factorisations
        this->doUpdates();
        std::cout<<this->objective_<<" - "<<this->nullsp_dim_<<"\n";
    }
    // outside loop but run only if feasibility is successful:
    std::cout<<this->objective_<<" iterations: "<<this->info_.qp_iteration_count<<" time: "<<this->timer_.read()<<"\n";
    return this->getHighsStatus();
}

void AsmSolver::doUpdates(){
    if ( this->atFSEP_ ) {
        this->stepAlreadyTaken_ = false;
        if ( this->newactive_idx_ > - 1){ // if a constraint is activated
            if ( this->nullsp_dim_ == 0) this->replace();
            else { // update factorisations of both B and M
                this->extend();
                this->activate();
                this->atFSEP_ = false;
            }
        } else {
            this->extend();
            this->atFSEP_ = false;
        }
    } else {
        if ( this->newactive_idx_ > -1){
            this->activate();
            this->stepAlreadyTaken_ = false;
        } else {
            this->atFSEP_ = true;
            this->stepAlreadyTaken_ = true;
        }
    }
    this->updateObjective();
    this->computeReducedVecs(); // red grad needs updating with new position
    this->info_.qp_iteration_count++;
}

void AsmSolver::findBestPrice(HighsInt& bestidx, double& bestmultiplier){
    double bestprice = - this->options_.dual_feasibility_tolerance;
    double price {0.}, sign {0.}, bestpricesign {0.}; // TODO messy initialisations and temp variables
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
            bestidx = idx;
            this->relaxed_iloc_ = i;
        }
    }
    bestmultiplier = bestpricesign * bestprice;
    return;
}

void AsmSolver::computeRelaxedDirection(){
    HighsInt bestidx {-1};
    double bestmultiplier;
    this->findBestPrice(bestidx, bestmultiplier);
    if ( bestidx == -1 ) this->model_status_ = HighsModelStatus::kOptimal; // set to optimal to break the major loop
    else { 
        double alpha_min = 0.; // use first as buffer for y_p^T Q y_p
        // extract yp
        this->step_.assign(this->Q_.dim_, 0.);
        this->step_[ this->basis_perm_[this->relaxed_iloc_] ] = 1.;
        this->B_.btranCall(this->step_); // y_p
        std::vector<double> vec(this->Q_.dim_);
        this->Q_.product(this->step_, vec); // Q y_p
        for (HighsInt i {0}; i < this->Q_.dim_; i++) alpha_min += this->step_[i] * vec[i]; // compute y_p^T Q y_p for later
        // compute direction
        if ( this->nullsp_dim_ > 0 ){
            this->HFtran(vec); // B^{-1} Q y_p ~ Z^T Q y_p (in memory)
            LLTsolve(vec); // M^{-1} Z^T Q y_p
            std::fill(vec.begin(), vec.end() - this->nullsp_dim_, 0.); // [ 0 | M^{-1} Z^T Q y_p ]
            this->HBtran(vec); // B^{-T} [ 0 | M^{-1} Z^T Q y_p ] = Z M^{-1} Z^T Q y_p
            for (HighsInt i {0}; i < this->Q_.dim_; i++) this->step_[i] -= vec[i]; // ( I - Z M^{-1} Z^T Q ) y_p
        }
        // compute step
        if ( std::abs(alpha_min) <= this->options_.factor_pivot_tolerance ) std::cout<<"Zero search direction"; // TODO
        if ( alpha_min < - this->options_.factor_pivot_tolerance ) std::cout<<"Negative curvature search direction"; // TODO
        alpha_min = - bestmultiplier / alpha_min;
        for (HighsInt i {0}; i < this->Q_.dim_; i++){
            this->step_[i] *= alpha_min;
            this->newvarvals_[i] = this->solution_.col_value[i] + this->step_[i];
        }
    }
    return;
}


void AsmSolver::computeReducedDirection(){ // solve Equality Problem
    if ( this->stepAlreadyTaken_ ) this->recomputeRedHessian(); // TODO keep?
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){
        this->step_[this->rangsp_dim_ + i] = - this->red_grad_[i]; // TODO, is there a better place to flip sign?
    }
    this->LLTsolve(this->step_);
    // then compute full space step
    std::fill(this->step_.begin(), this->step_.begin() + this->rangsp_dim_, 0.);
    this->HBtran(this->step_);
    // and update newvarvals
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        this->newvarvals_[i] = this->solution_.col_value[i] + this->step_[i];
    }
    return;
}

void AsmSolver::activate(){
    assert(getAsmBasisStatus(this->newactive_idx_) == AsmBasisStatus::kFreeInBasis ||
           getAsmBasisStatus(this->newactive_idx_) == AsmBasisStatus::kInactive);
    // whether the constraint is padding or inactive, check whether the existing padding already has the correct vector
    if ( this->getAsmBasisStatus(this->newactive_idx_) == AsmBasisStatus::kFreeInBasis ){
        assert( this->newactive_idx_ >= this->lp_.num_row_ ); // it should be a unit vector
        for (HighsInt loc_remove {0}; loc_remove < this->nullsp_dim_; loc_remove++){
            if ( this->basis_idxs_[this->rangsp_dim_ + loc_remove] == this->newactive_idx_ ){ // unit vector already in basis
                this->changeStatus(this->newactive_idx_, this->newactive_status_); // handle status update for activation
                if ( loc_remove >  0) // memory operations needed if we are activating a constraint that is not the first in padding
                    this->fromPaddingToActive(loc_remove + this->rangsp_dim_); // handle index update
                // update factorization if the unit vector that is activated is already-in-basis
                this->reduceInBasis(loc_remove);
                break;
            }
        }
    } else { // if we are activating a constraint or the variable bound we are activating is not already in V
        HighsInt loc_remove {-1};
        this->reducePadding(this->newactive_idx_, loc_remove); // yields location of padding vector to be removed from basis_idxs_
        this->changeStatus( this->basis_idxs_[loc_remove], AsmBasisStatus::kInactive );
        this->basis_idxs_[loc_remove] = this->newactive_idx_; // update index
        if ( loc_remove > this->rangsp_dim_ ) this->fromPaddingToActive(loc_remove);
        this->changeStatus(this->newactive_idx_, this->newactive_status_);
        removeNullSpaceDim();
    }
    this->newactive_idx_ = -1; // book keeping
    this->newactive_status_ = AsmBasisStatus::kInactive;
    return;
}