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
        while ( this->keepLooping() ) { // major iterations
            this->relaxAndSearch(); // increase size of nullspace
            if ( this->isOptimal() ) break;
            this->minorloop(); // find optimum in the nullspace
            //if ( this->num_basis_updates_ > this->reinversion_freq_) reinvertBasis();
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
        assert(sign < 1.1); // either upper, equality, or lower
        price = sign * this->pricing_[i];
        if ( price < bestprice ){
            bestpricesign = sign;
            bestprice = price;
            bestidx = idx;
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

void AsmSolver::relaxAndSearch(){ // loop through prices to find a constraint to deactivate
    HighsInt bestidx {-1}, bestloc;
    double bestmultiplier;
    this->findBestPrice(bestidx, bestmultiplier, bestloc);
    if ( bestidx == -1 ) this->model_status_ = HighsModelStatus::kOptimal; // set to optimal to break the major loop
    else {
        this->computeSearchDir(bestloc, bestmultiplier);
        HighsInt newactiveidx {-1};
        AsmBasisStatus newactivestatus;
        this->ratiotest(newactiveidx, newactivestatus);
        if ( newactiveidx > - 1){ // if a constraint is activated
            if ( this->nullsp_dim_ == 0){ // replace relaxed with new one, update B factors only if we are going from vertex to vertex
                this->replace( bestloc, newactiveidx, newactivestatus ); // statuses updated and indexes swapped inside replace()
            } else { // update factorisations of both B and M
                this->extend( bestloc );
                this->activate( newactiveidx, newactivestatus );
            }
        } else this->extend( bestloc );
        this->updateObjective();
        this->computeReducedVecs(); // TODO recomputing local gradient may not be necessary
        this->info_.qp_iteration_count++;
        std::cout<<this->objective_<<" - "<< this->nullsp_dim_<<"\n"<<std::flush;
    }
    return;
}

void AsmSolver::solveEP(){ // solve Equality Problem
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
}

void AsmSolver::minorloop(){
    this->stepAlreadyTaken_ = false;
    while ( norm(this->red_grad_) > this->options_.primal_feasibility_tolerance ){ // if nullsp is empty then norm returns 0
        if ( this->stepAlreadyTaken_ ) recomputeRedHessian(); // if the last step was unconstrained but somehow we are not at reduced optimum yet
        this->solveEP();
        //if ( norm(this->step_) < this->options_.primal_feasibility_tolerance ) break;
        HighsInt newactiveidx {-1};
        AsmBasisStatus newactivestatus;
        this->ratiotest(newactiveidx, newactivestatus);
        if ( newactiveidx > -1 ) this->activate(newactiveidx, newactivestatus);
        else this->stepAlreadyTaken_ = true;
        this->updateObjective();
        this->computeReducedVecs(); // red grad needs updating with new position
        this->info_.qp_iteration_count++;
        std::cout<<this->objective_<<" - "<< this->nullsp_dim_<<"\n"<<std::flush;
    }
    return;
}

void AsmSolver::activate(const HighsInt& idx, const AsmBasisStatus& status){
    AsmBasisStatus oldstatus = getAsmBasisStatus(idx);
    // old status cannot be active since activation requires movement in nullspace
    assert (oldstatus == AsmBasisStatus::kFreeInBasis || oldstatus == AsmBasisStatus::kInactive);
    // whether the constraint is padding or inactive, check whether the existing padding already has the correct vector
    if ( idx >= this->lp_.num_row_){ // if we are activating a variable's bound
        HighsInt loc_remove {-1};
        HighsInt varidx = idx - this->lp_.num_row_;
        for (loc_remove = 0; loc_remove < this->nullsp_dim_; loc_remove++){
            if ( this->Vi_[loc_remove] == varidx ){ // unit vector already in basis
                HighsInt loc_actual = this->rangsp_dim_ + loc_remove;
                // the unit vector in the padding took the place of some other constraint, whose status needs to be updated
                if ( this->basis_idxs_[loc_actual] != idx ){ // if the unit vector was a replacement for some constraint that will leave the basis
                    this->changeStatus(this->basis_idxs_[loc_actual], AsmBasisStatus::kInactive); // deactivate old constraint (book-keeping)
                    this->basis_idxs_[loc_actual] = idx; // update index
                }
                // or if the unit vector is an actual bound that is being activated
                this->changeStatus(idx, status); // handle status update for activation
                this->fromPaddingToActive(loc_actual);
                this->Vi_.erase( this->Vi_.begin() + loc_remove ); // remove reference to element in padding
                // update factorization if the unit vector that is activated is already-in-basis
                this->reduceInBasis(loc_remove);
                return;
            }
        }
    }
    // if we are activating a constraint or the variable bound we are activating is not already in V
    HighsInt loc_remove {-1};
    if ( oldstatus == AsmBasisStatus::kInactive ){
        this->reducePadding(idx, loc_remove); // returns location of padding vector to be removed from basis_idxs_
        this->changeStatus( this->basis_idxs_[loc_remove], AsmBasisStatus::kInactive );
        this->basis_idxs_[loc_remove] = idx;
    } else if ( oldstatus == AsmBasisStatus::kFreeInBasis ){
        for ( HighsInt i {this->rangsp_dim_}; i < this->Q_.dim_; i++){
            if ( this->basis_idxs_[i] == idx ){
                loc_remove = i;
                this->reducePadding(idx, loc_remove);
                break;
            }
        }
    }
    assert(loc_remove>-1);
    this->fromPaddingToActive(loc_remove);
    this->changeStatus(idx, status);
    removeNullSpaceDim();
    return;
}