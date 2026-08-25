/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "qpsolver/g_solver.hpp"

AsmSolver::AsmSolver(const HighsOptions& options,
                     HighsTimer& timer,
                     HighsLp lp,
                     HighsHessian hessian,
                     HighsBasis& basis,
                     HighsSolution& solution,
                     HighsModelStatus& model_status,
                     HighsInfo& info,
                     HighsCallback& callback)
                     : options_(options),
                     timer_(timer),
                     lp_(lp),
                     Q_(hessian),
                     lp_basis_(basis),
                     solution_(solution),
                     model_status_(model_status),
                     info_(info),
                     callback_(callback),
                     feasibility_lp_(lp),
                     buffer_(hessian.dim_),
                     loc_grad_(hessian.dim_),
                     step_(hessian.dim_),
                     basis_perm_(hessian.dim_), // no init of basis_idxs_ as it is built with push_back()
                     var_status_(hessian.dim_),
                     con_status_(lp.num_row_){}

HighsStatus AsmSolver::getHighsStatus(){ // public function
    return this->status_; // private attribute
}

HighsModelStatus AsmSolver::getHighsModelStatus(){ // public function
    return this->model_status_; // private attribute
}

void AsmSolver::HSetup(const HighsSparseMatrix& constraint_mat){
    this->B_.setup(constraint_mat, this->basis_idxs_); //shuffles basis indices
    return;
}

void AsmSolver::HBuild(){ // also for refactorization
    this->B_.build();
    return;
}

void AsmSolver::HBtran(std::vector<double>& vec){
    if ((HighsInt)this->buffer_.size() != this->Q_.dim_) throw std::length_error("Wrong buffer_ size!");
    // first apply P
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        this->buffer_[ this->basis_perm_[i] ] = vec[i];
    } // then solve
    this->B_.btranCall(this->buffer_); // B^{-T}
    vec = this->buffer_;
    return;
}

void AsmSolver::HFtran(std::vector<double>& vec){
    if ((HighsInt)this->buffer_.size() != this->Q_.dim_) throw std::length_error("Wrong buffer_ size!");
    this->B_.ftranCall(vec); // first solve for B^{-1}
    // then apply P^T = P^{-1}
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        this->buffer_[ i ] = vec[ this->basis_perm_[i] ];
    }
    vec = this->buffer_;
    return;
}

void AsmSolver::HBtran(HVector& vec, const double expected_density){
    this->B_.btranCall(vec, expected_density); // no permutations here, assumed to be taken care of when forming inputs
    return;
}

void AsmSolver::HFtran(HVector& vec, const double expected_density){
    this->B_.ftranCall(vec, expected_density); // no permutations here, assumed to be taken care of when forming inputs
    return;
}

void AsmSolver::HUpdate(HighsInt loc_idxdrop, HighsInt idx_new){
    HighsInt hint { 99999 }; // same number as Micheal in Basis::updatebasis
    // build HVector to add to basis
    HVector newcol;
    if (idx_new < this->lp_.num_row_){ // extract constraint and build HVec
        std::vector<double> ep(this->lp_.num_row_);
        ep[idx_new] = 1.;
        this->lp_.a_matrix_.productTranspose(this->buffer_, ep);
        newcol = stdvec2hvec(this->buffer_);
    }
    else newcol = unit_hvec(idx_new - this->lp_.num_row_); // for a new bound becoming active
    this->HFtran(newcol, 1.);
    // build HVector to point to constraint exiting basis
    HVector ep = unit_hvec(loc_idxdrop);
    this->HBtran(ep, 1.);
    // update basis matrix
    this->B_.update(&newcol, &ep, &loc_idxdrop, &hint);
    return;
}

HVector AsmSolver::stdvec2hvec(const std::vector<double>& vec){
    HVector hvec;
    hvec.setup(vec.size());
    HighsInt count_nz {0};
    for (size_t i {0}; i < vec.size(); i++){
        if (std::abs(vec[i]) < this->options_.primal_feasibility_tolerance){
            hvec.index[count_nz] = i;
            count_nz += 1;
        }
    }
    hvec.array = vec;
    hvec.count = count_nz;
    hvec.packFlag = true;
    return hvec;
}

HVector AsmSolver::unit_hvec(const HighsInt& p){
    HVector hvec;
    hvec.setup(this->Q_.dim_);
    hvec.packFlag = true; // what is this?
    hvec.index[0] = p;
    hvec.array[p] = 1.;
    hvec.count = 1;
    return hvec;
}

void AsmSolver::feasibility(){
    // TODO hotstart if basis is provided
    if (this->options_.qp_allow_hot_start &&
        this->lp_basis_.valid &&
        this->solution_.value_valid){
        // TODO add check to make sure basis checks out with solution
        this->status_ = HighsStatus::kError; // return false to not run the active set solver
    } else {
        this->setupFeasibilityLp();
        Highs highs_feasibility;
        highs_feasibility.passModel(this->feasibility_lp_);
        highs_feasibility.passOptions(this->options_);
        //feasibility_lp.setOptionValue("presolve", kHighsOnString); // presolving phase1 makes it faster, im guessing the postsolve is included
        highs_feasibility.setOptionValue("output_flag", false); // don't print anything
        highs_feasibility.setOptionValue("simplex_strategy", kSimplexStrategyDual); // specifying what solver to use in case a basis is set that is known to be either primal or dual feasible
        // use dual simplex if the objective value is all zeros, beacuse that means dual feasibility is guaranteed
        this->status_ = highs_feasibility.run();
        this->model_status_ = highs_feasibility.getModelStatus();
        // TODO deal with timer? report it up
        if ( this->model_status_ == HighsModelStatus::kOptimal ){ // note Optimal in Phase1 is Feasible for ASM
            this->info_.simplex_iteration_count = highs_feasibility.getSimplexIterationCount();
            this->lp_basis_ = highs_feasibility.getBasis();
            this->solution_ = highs_feasibility.getSolution();
            this->updateObjective();
            this->setupQpBasis();
            // compute relaxed bounds for two pass ratio test
            this->computeRelaxedBounds(this->lp_.row_lower_, this->lp_.row_upper_,
                                       this->lp_relaxed_.row_lower_, this->lp_relaxed_.row_upper_);
            this->computeRelaxedBounds(this->lp_.col_lower_, this->lp_.col_upper_,
                                       this->lp_relaxed_.col_lower_, this->lp_relaxed_.col_upper_);
        }
    }
}

void AsmSolver::setupFeasibilityLp(){
    // build feasibility_lp_
    this->feasibility_lp_.col_cost_.assign(this->Q_.dim_, 0.); // zero out objective
    return;
    // TODO minimize slacks in this first phase
    // we want to have as small of a nullspace as possible;
    // do we also want to have as many bounds, rather than constraints,
    // active, to maximise HFactor's sparsity?
}

void AsmSolver::setupQpBasis(){
    // init active and free temporary index vectors
    std::vector<HighsInt> active_idxs;
    std::vector<HighsInt> free_idxs;
    for (HighsInt i {0}; i < this->lp_.num_row_; i++){ // loop through constraints
        this->con_status_[i] = this->HighsStatusToAsm(this->lp_basis_.row_status[i], i, false);
        if ( this->isInBasis(this->con_status_[i]) ) active_idxs.push_back(i); // add index to list of indices
        // constraints shouldn't be free in basis, ignore HighsBasisStatus::kZero and HighsBasisStatus::kNonbasic
    }
    for (HighsInt i {0}; i < this->Q_.dim_; i++){ // loop through variables
        HighsInt idx = i + this->lp_.num_row_;
        this->var_status_[i] = this->HighsStatusToAsm(this->lp_basis_.col_status[i], i, true);
        // ignore HighsBasisStatus::kNonbasic
        if ( this->isInBasis(this->var_status_[i]) ){
            if ( this->isFreeInBasis(this->var_status_[i]) ) free_idxs.push_back(idx); // add index to list of indices
            else active_idxs.push_back(idx); // if not free then it is active in the basis
        }
    }
    // set nullspace and range dimensions
    this->nullsp_dim_ = (HighsInt) free_idxs.size();
    this->rangsp_dim_ = (HighsInt) active_idxs.size();
    if (this->rangsp_dim_ + this->nullsp_dim_ != this->Q_.dim_) throw std::logic_error("Active and Free constraints should add up to number of columns!");
    // merge indices
    this->basis_idxs_ = active_idxs;
    this->basis_idxs_.insert(this->basis_idxs_.end(),
                             free_idxs.begin(), free_idxs.end());
    std::vector<HighsInt> ordered_basis = this->basis_idxs_; // store buffer
    this->setupBasisMat(); // setup HFactor
    // since basis indices may have been shuffled so that free indices may not trail active ones anymore,
    // set permutation order to match the index sets (A,V) structure
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        for (HighsInt j {0}; j < this->Q_.dim_; j++){
            if (ordered_basis[i] == this->basis_idxs_[j]){
                this->basis_perm_[i] = j;
                break;
            }
        }
    }
    this->basis_idxs_ = ordered_basis; // then restore order in basis indices
    this->setupReducedHessian(); // build Reduced Hessian
    this->computeReducedVecs(); // compute initial reduced gradient and pricing
    return;
}

void AsmSolver::setupBasisMat(){ // TODO do not create constraint mat copy
    HighsSparseMatrix constraint_mat = this->lp_.a_matrix_; // create a copy of the constraint matrix
    constraint_mat.ensureRowwise(); // flip the way in which it is stored
    constraint_mat.format_ = MatrixFormat::kColwise; // but "trick it" into thinking it is still stored columnwise
    HighsInt temp_old_num_row = constraint_mat.num_row_; // flip the number of rows and columns
    constraint_mat.num_row_ = constraint_mat.num_col_; // so that when HFactor uses the matrix
    constraint_mat.num_col_ = temp_old_num_row; // it receives the constraint matrix stored "column wise"
    this->HSetup(constraint_mat); // where each column is a constraint. its inverse transpose will have as columns the nullspace basis
    this->HBuild();
    return;
}

void AsmSolver::setupReducedHessian(){
    // change hessian to square for future
    if (this->Q_.format_ == HessianFormat::kTriangular) this->Q_ = this->Q_.toSquare();
    HighsInt chol_size = this->nullsp_dim_ * (this->nullsp_dim_ + 1) / 2; // number of elements in lower triangular matrix
    this->chol_.assign(chol_size, 0.);
    this->recomputeExplicit();
    this->refactorize();
    return;
}

HighsStatus AsmSolver::run(){
    this->feasibility();
    this->lp_.a_matrix_.ensureRowwise(); // for copying row-by-row in case of degeneracy
    if ( this->model_status_ == HighsModelStatus::kOptimal ){
        this->model_status_ = HighsModelStatus::kNotset;
        while ( true ) { // ASM iterations
            if (this->norm(this->red_grad_) < this->options_.primal_feasibility_tolerance){ // TODO primal residual tolerance?
                if ( this->maximalsteptaken() ) break;
                this->deactivate();
                if ( this->isoptimal() ) break;
                if ( this->iterlimit() ) break;
                if ( this->timelimit() ) break;
                if ( this->nullsizelimit() ) break;
            } else {
                this->takeStep();
                std::cout<<this->objective_<<"\n";
            }
        }
        // outside loop but run only if feasibility is successful:
        std::cout<<this->objective_<<" "<<this->info_.qp_iteration_count<<"\n";
    }
    // TODO record runtime?
    return this->getHighsStatus();
}

void AsmSolver::deactivate(){ // loop through prices to find a constraint to deactivate
    HighsInt bestloc {0};
    double bestprice = this->pricing_[bestloc];
    HighsInt bestidx = this->basis_idxs_[bestloc];
    for (HighsInt i {1}; i < this->rangsp_dim_; i++){ // loop through active constraints only
        HighsInt idx = this->basis_idxs_[i];
        if (idx < this->lp_.num_row_) { // it is a constraint
            if ( this->isNotEquality( this->con_status_[idx] ) &&
                this->pricing_[i] < bestprice && 
                this->pricing_[i] < - this->options_.factor_pivot_tolerance){
                bestprice = this->pricing_[i];
                bestidx = idx;
                bestloc = i;
            }
        } else { // it is a variable bound
            idx -= this->lp_.num_row_; // get variable index
            if ( this->isNotEquality( this->var_status_[idx]) &&
                this->pricing_[i] < bestprice && 
                this->pricing_[i] < - this->options_.factor_pivot_tolerance){
                bestprice = this->pricing_[i];
                bestidx = this->basis_idxs_[i];
                bestloc = i;
            }
        }
    }
    // check that bestprice is indeed negative, in case first price is best but non-negative
    if ( bestprice < - this->options_.dual_feasibility_tolerance ){ // TODO check negativity of tolerance
        // first return price to original value to update reduced gradient, then update status
        if (bestidx < this->lp_.num_row_){ 
            bestprice *= static_cast<double>( this->con_status_[bestidx] );
            this->con_status_[bestidx] = AsmBasisStatus::kFreeInBasis;
        }
        else {
            HighsInt var_idx = bestidx - this->lp_.num_row_;
            bestprice *= static_cast<double>( this->var_status_[var_idx] );
            this->var_status_[var_idx] = AsmBasisStatus::kFreeInBasis;
        }
        // extend the reduced hessian, no need to change HFactor
        this->extend(bestloc); // then extend the basis factorization
        // send deactivated constraint to the end
        std::vector<HighsInt>::iterator it = this->basis_idxs_.begin() + bestloc;
        std::rotate(it, it + 1, this->basis_idxs_.end());
        it = this->basis_perm_.begin() + bestloc;
        std::rotate(it, it + 1, this->basis_perm_.end());
        this->addNullSpaceDim();
        // update reduced gradient with element of pricing corresponding to deactivated constraint
        this->red_grad_.push_back(bestprice);
        this->pricing_.erase(this->pricing_.begin() + bestloc);
        this->step_taken_ = false; // since problem has been modified
    } else this->model_status_ = HighsModelStatus::kOptimal; // set to optimal to break the major loop
    return;
}

void AsmSolver::ratiotest_pass1(const std::vector<double>& newloc,
                                const std::vector<double>& newconvals,
                                const std::vector<double>& denoms){
    this->alpha_relaxed_ = 1.; // we want to minimise it
    double alpha_here;
    for (HighsInt i {0}; i < this->Q_.dim_; i++){ // loop through variables
        if (this->step_[i] < - this->options_.factor_pivot_tolerance && this->lp_relaxed_.col_lower_[i] > newloc[i] ){
            // lower bound break
            alpha_here = ( this->lp_relaxed_.col_lower_[i] - this->solution_.col_value[i] ) / this->step_[i];
            if ( alpha_here < this->alpha_relaxed_ ) this->alpha_relaxed_ = alpha_here;
        } else if ( this->step_[i] > this->options_.factor_pivot_tolerance && this->lp_relaxed_.col_upper_[i] < newloc[i] ){
            // upper bound break
            alpha_here = ( this->lp_relaxed_.col_upper_[i] - this->solution_.col_value[i] ) / this->step_[i];
            if ( alpha_here < this->alpha_relaxed_ ) this->alpha_relaxed_ = alpha_here;
        }
    }
    for (HighsInt i {0}; i < this->lp_.num_row_; i++){ // loop through inactive constraints
        if ( denoms[i] < - this->options_.factor_pivot_tolerance && this->lp_relaxed_.row_lower_[i] > newconvals[i] ){
            alpha_here = ( this->lp_relaxed_.row_lower_[i] - this->solution_.row_value[i] ) / denoms[i];
            if ( alpha_here < this->alpha_relaxed_ ) this->alpha_relaxed_ = alpha_here;
        } else if ( denoms[i] > this->options_.factor_pivot_tolerance && this->lp_relaxed_.row_upper_[i] < newconvals[i] ) {
            alpha_here = ( this->lp_relaxed_.row_upper_[i] - this->solution_.row_value[i] ) / denoms[i];
            if ( alpha_here < this->alpha_relaxed_ ) this->alpha_relaxed_ = alpha_here;
        }
    }
    return;
}
void AsmSolver::ratiotest_pass2(const std::vector<double>& newloc,
                                const std::vector<double>& newconvals,
                                const std::vector<double>& denoms,
                                HighsInt& newactive_idx,
                                AsmBasisStatus& newactive_status){
    double alpha_here;
    double max_pivot = this->options_.factor_pivot_tolerance; // here select constraint to activate based on denominator, not alpha
    for (HighsInt i {0}; i < this->Q_.dim_; i++){ // loop through variables
        if ( this->step_[i] < - max_pivot ){ // potential lower bound break
            alpha_here = ( this->lp_.col_lower_[i] - this->solution_.col_value[i] ) / this->step_[i];
            if ( alpha_here < this->alpha_relaxed_ ){
                newactive_idx = i + this->lp_.num_row_;
                newactive_status = AsmBasisStatus::kLower;
                this->alpha_ = alpha_here;
                max_pivot = - this->step_[i];
            }
        } else if ( this->step_[i] > max_pivot ){ // potential upper bound break
            alpha_here = ( this->lp_.col_upper_[i] - this->solution_.col_value[i] ) / this->step_[i];
            if ( alpha_here < this->alpha_relaxed_ ){
                newactive_idx = i + this->lp_.num_row_;
                newactive_status = AsmBasisStatus::kUpper;
                this->alpha_ = alpha_here;
                max_pivot = this->step_[i];
            }
        }
    }
    for (HighsInt i {0}; i < this->lp_.num_row_; i++){ // loop through inactive constraints
        if ( denoms[i] < - max_pivot ){
            alpha_here = ( this->lp_.row_lower_[i] - this->solution_.row_value[i] ) / denoms[i];
            if ( alpha_here < this->alpha_relaxed_ ){
                newactive_idx = i;
                newactive_status = AsmBasisStatus::kLower;
                this->alpha_ = alpha_here;
                max_pivot = - denoms[i];
            }
        } else if ( denoms[i] > max_pivot ) {
            alpha_here = ( this->lp_.row_upper_[i] - this->solution_.row_value[i] ) / denoms[i];
            if ( alpha_here < this->alpha_relaxed_ ){
                newactive_idx = i;
                newactive_status = AsmBasisStatus::kUpper;
                this->alpha_ = alpha_here;
                max_pivot = denoms[i];
            }
        }
    }
    return;
}

void AsmSolver::takeStep(){
    // solve Equality Problem first
    this->delta_ = this->red_grad_; // TODO remove and flip the sign when more convenient?
    for (size_t i {0}; i < this->delta_.size(); i++){
        this->delta_[i] *= -1.; // flip sign to solve reduced system
    }
    this->stepSanity(); // make sure the same identical problem has not been solved yet
    this->LLTsolve(this->delta_);
    this->computeFullStep(this->delta_, this->step_); // then compute full space step
    // ratio test vectors
    std::vector<double> newloc(this->Q_.dim_);
    this->compute_varvals(1., newloc); // compute (potential) x_{k+1}
    std::vector<double> newconvals(this->lp_.num_row_); // vector for new constraint values
    std::vector<double> denoms(this->lp_.num_row_); // vectors for denominators of ratio test formula
    this->lp_.a_matrix_.product(newconvals, newloc); // a_i^T x_{k+1}
    this->lp_.a_matrix_.product(denoms, this->step_); // a_i^T \s
    ratiotest_pass1(newloc, newconvals, denoms); // ratio test on relaxed instance
    if (this->alpha_relaxed_ < 1.){
        HighsInt newactive_idx {-1};
        AsmBasisStatus newactive_status;
        ratiotest_pass2(newloc, newconvals, denoms, newactive_idx, newactive_status);
        this->compute_varvals(this->alpha_, this->solution_.col_value);
        this->updateObjective();
        this->lp_.a_matrix_.product(this->solution_.row_value, this->solution_.col_value); // a_i^T x_{k+1}
        this->activate(newactive_idx, newactive_status);
    } else { // if no constraint activated
        this->solution_.row_value = newconvals; // don't recompute new constraint values
        this->solution_.col_value = newloc; // nor variables' values either
        this->updateObjective();
    }
    this->computeReducedVecs(); // red grad needs updating with new position
    this->step_taken_ = true;
    this->info_.qp_iteration_count++;
    return;
}

void AsmSolver::activate(const HighsInt& idx, const AsmBasisStatus& status){
    // handle status update
    bool alreadyinbasis {false};
    if (idx < this->lp_.num_row_){
        if ( this->isFreeInBasis(this->con_status_[idx]) ) alreadyinbasis = true;
        if ( this->lp_.row_lower_[idx] == this->lp_.row_upper_[idx] ) this->con_status_[idx] = AsmBasisStatus::kEquality;
        else this->con_status_[idx] = status;
    }
    else {
        HighsInt var_idx = idx - this->lp_.num_row_;
        if ( this->isFreeInBasis(this->var_status_[var_idx]) ) alreadyinbasis = true;
        if ( this->lp_.col_lower_[var_idx] == this->lp_.col_upper_[var_idx] ) this->var_status_[var_idx] = AsmBasisStatus::kEquality;
        else this->var_status_[var_idx] = status;
    }
    // TODO find good rationale to select which constraint to drop
    if (alreadyinbasis){
        // send element in location i to the end of active constraints
        // and shift all free in basis down by one until the old position of the constraint in activation
        HighsInt i;
        for (i = this->rangsp_dim_; i < this->Q_.dim_; i++){
            if (this->basis_idxs_[i] == idx){
                auto it = this->basis_idxs_.begin();
                std::rotate(it + this->rangsp_dim_, it + i, it + i + 1);
                it = this->basis_perm_.begin();
                std::rotate(it + this->rangsp_dim_, it + i, it + i + 1);
                break;
            }
        }
        if (i < this->Q_.dim_ - 1){
            this->remove(i - this->rangsp_dim_);
            return;
        } // else we just need to drop the last row of the lower triangular factor
    } else { // update HFactor if constraint not already in basis
        // drop the last column in V and substitute it with new index,
        // which then moves to the end of the current active set while all other free indices are shifted by 1 to the end
        this->HUpdate(this->basis_perm_.back(), idx);
        // change dropped constraint to inactive
        if (this->basis_idxs_.back() < this->lp_.num_row_) this->con_status_[this->basis_idxs_.back()] = AsmBasisStatus::kInactive;
        else this->var_status_[this->basis_idxs_.back() - this->lp_.num_row_] = AsmBasisStatus::kInactive;
        this->basis_idxs_.back() = idx;
        std::rotate(this->basis_idxs_.begin() + this->rangsp_dim_, this->basis_idxs_.end() - 1, this->basis_idxs_.end());
        std::rotate(this->basis_perm_.begin() + this->rangsp_dim_, this->basis_perm_.end() - 1, this->basis_perm_.end());   
    }
    // update factorization if the already-in-basis row was the last one or if we arbitrarily chose to drop the last one
    this->chol_.resize(this->chol_.size() - this->nullsp_dim_); // drop last row of L
    this->removeNullSpaceDim();
    this->ZT_.resize(this->nullsp_dim_); // drop last row of Z^T (last column of Z)
    return;
}

void AsmSolver::computeRelaxedBounds(const std::vector<double>& old_lower,
                                     const std::vector<double>& old_upper,
                                     std::vector<double>& new_lower,
                                     std::vector<double>& new_upper){
    new_lower = old_lower;
    new_upper = old_upper;
    for (size_t i {0}; i < new_lower.size(); i++){
        new_lower[i] -= this->options_.factor_pivot_tolerance;
        new_upper[i] += this->options_.factor_pivot_tolerance;
    }
}

void AsmSolver::computeLocGrad(){ // g + Q x_k
    this->Q_.product(this->solution_.col_value, this->loc_grad_); // stores result in loc_grad_
    for (HighsInt i {0}; i < this->Q_.dim_; i++){ // add g to Q x_k
        this->loc_grad_[i] += this->lp_.col_cost_[i];
    }
    return;
}

double AsmSolver::computeReducedVecs(){ // solve B x = (g + Q x_k) to compute Dantzig prices and reduced gradient
    this->computeLocGrad();
    std::vector<double> vec = this->loc_grad_;
    this->HFtran(vec); // compute B x = g_k
    this->pricing_.assign(vec.begin(), vec.end() - this->nullsp_dim_); // TODO, other types of pricing
    this->signPrices();
    this->red_grad_.assign(vec.end() - this->nullsp_dim_, vec.end()); // extract last z elements of the result, i.e. Z^T (g + Q x_k)
    // returns the magnitude of the reduced gradient (0 if of null dimension)
    return norm(this->red_grad_);
}

void AsmSolver::compute_varvals(const double& alpha, std::vector<double>& loc){ // compute x_{k+1}
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        loc[i] = this->solution_.col_value[i] + alpha * this->step_[i];
    }
    return;
}

void AsmSolver::computeFullStep(const std::vector<double>& delta, std::vector<double>& step){
    step.assign(this->rangsp_dim_, 0.);
    step.insert(step.end(), delta.begin(), delta.end());
    this->HBtran(step); // is this cheaper than holding the explicit Z^T and using that one?
    return;
}

double AsmSolver::computeQuadObjective(const std::vector<double>& vec){
    double sum {0.};
    // TODO take advantage of symmetry
    for (HighsInt iCol = 0; iCol < this->Q_.dim_; iCol++) { // from claude.ai
        for (HighsInt iEl = this->Q_.start_[iCol]; iEl < this->Q_.start_[iCol + 1]; iEl++) {
            sum += 0.5 * vec[iCol] * this->Q_.value_[iEl] * vec[this->Q_.index_[iEl]];
        }
    }
    return sum;
}

void AsmSolver::updateObjective(){
    this->objective_ = this->lp_.objectiveValue(this->solution_.col_value);
    this->objective_ += computeQuadObjective(this->solution_.col_value);
    return;
}

void AsmSolver::signPrices(){
    for (HighsInt i {0}; i < this->rangsp_dim_; i++){
        HighsInt idx { this->basis_idxs_[i] };
        if (idx < this->lp_.num_row_) this->pricing_[i] *= static_cast<double>( this->con_status_[idx] );
        else{
            idx -= this->lp_.num_row_;
            this->pricing_[i] *= static_cast<double>( this->var_status_[idx] );
        }
    }
    return;
}

bool AsmSolver::iterlimit(){// iteration limit
    if (this->info_.qp_iteration_count >= this->options_.qp_iteration_limit){
        this->model_status_ = HighsModelStatus::kIterationLimit;
        this->status_ = HighsStatus::kWarning; // TODO ok?
        return true;
    }
    return false;
};

bool AsmSolver::timelimit(){// time limit
    if (this->timer_.read() >= this->options_.time_limit){
        this->model_status_ = HighsModelStatus::kTimeLimit;
        this->status_ = HighsStatus::kWarning; // TODO ok?
        return true;
    }
        return false;
};

bool AsmSolver::maximalsteptaken(){// optimality condition
        if (this->nullsp_dim_ == this->Q_.dim_ && this->step_taken_){ // cannot deactivate anything anymore, nullspace is maximal already
            this->model_status_ = HighsModelStatus::kOptimal;
            this->status_ = HighsStatus::kOk;
        return true;
        }
    return false;
};

bool AsmSolver::nullsizelimit(){// nullspace size limit
        if ( this->nullsp_dim_ > this->options_.qp_nullspace_limit){
            this->model_status_ = HighsModelStatus::kSolveError;
            this->status_ = HighsStatus::kError;
        return true;
        }
    return false;
};

bool AsmSolver::isoptimal(){ // break loop if optimality check is positive during deactivation
        if ( this->model_status_ == HighsModelStatus::kOptimal ){
            this->status_ = HighsStatus::kOk;
        return true;
    }
    return false;
}

void AsmSolver::stepSanity(){
    if (this->step_taken_){
        this->recomputeExplicit();
        this->refactorize();
        this->step_taken_ = false;
    }
    return;
}

void AsmSolver::addNullSpaceDim(){
    this->nullsp_dim_++;
    this->rangsp_dim_--;
    return;
}

void AsmSolver::removeNullSpaceDim(){
    this->nullsp_dim_--;
    this->rangsp_dim_++;
    return;
}
// convert Highs Status to Asm Status
AsmBasisStatus AsmSolver::HighsStatusToAsm(const HighsBasisStatus& status, const HighsInt i, const bool variable){
    if(status == HighsBasisStatus::kLower){
        if (variable){ // it is a variable this should be a fixed variable and needs presolve
            if (this->lp_.col_lower_[i] == this->lp_.col_upper_[i]) return AsmBasisStatus::kEquality;
        } else if (this->lp_.row_lower_[i] == this->lp_.row_upper_[i]) return AsmBasisStatus::kEquality;
        return AsmBasisStatus::kLower;
    }
    else if(status == HighsBasisStatus::kUpper) return AsmBasisStatus::kUpper;
    else if(status == HighsBasisStatus::kZero) return AsmBasisStatus::kFreeInBasis;
    else return AsmBasisStatus::kInactive;
}

bool AsmSolver::isNotEquality(const AsmBasisStatus& status){
    if (status != AsmBasisStatus::kEquality) return true;
    else return false;
}

bool AsmSolver::isInBasis(const AsmBasisStatus& status){
    if (status == AsmBasisStatus::kInactive) return false; // note false return here
    else return true;
}

bool AsmSolver::isFreeInBasis(const AsmBasisStatus& status){
    if (status == AsmBasisStatus::kFreeInBasis) return true;
    else return false;
}

bool AsmSolver::isActive(const AsmBasisStatus& status){ // TODO unused, remove
    if (status == AsmBasisStatus::kLower ||
        status == AsmBasisStatus::kUpper ||
        status == AsmBasisStatus::kEquality) return true;
    else return false;
}

bool AsmSolver::isActiveInequality(const AsmBasisStatus& status){
    if (status == AsmBasisStatus::kLower ||
        status == AsmBasisStatus::kUpper) return true;
    else return false;
}

bool AsmSolver::isInactive(const AsmBasisStatus& status){ // TODO unused, remove
    if (status == AsmBasisStatus::kInactive ||
        status == AsmBasisStatus::kFreeInBasis) return true;
    else return false;
}

double AsmSolver::norm(const std::vector<double>& vec){
    double sum {0.}; // returns zero if size is null
    for (size_t i {0}; i < vec.size(); i++){
        sum += vec[i] * vec[i];
    }
    return std::sqrt(sum);
}