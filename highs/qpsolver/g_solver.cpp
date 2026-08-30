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
                     newvarvals_(hessian.dim_),
                     newconvals_(lp.num_row_),
                     newconpivots_(lp.num_row_),
                     basis_perm_(hessian.dim_), // no init of basis_idxs_ as it is built with push_back()
                     var_status_(hessian.dim_),
                     con_status_(lp.num_row_){
    // change hessian to square for better memory access
    if (this->Q_.format_ == HessianFormat::kTriangular) this->Q_ = this->Q_.toSquare();
}

HighsStatus AsmSolver::getHighsStatus(){ // public function
    return this->status_; // private attribute
}

HighsModelStatus AsmSolver::getHighsModelStatus(){ // public function
    return this->model_status_; // private attribute
}

void AsmSolver::HBtran(std::vector<double>& vec){
    if ((HighsInt)this->buffer_.size() != this->Q_.dim_) throw std::length_error("Wrong buffer_ size!");
    // first apply P
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        this->buffer_[ this->basis_perm_[i] ] = vec[ i ];
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

void AsmSolver::HUpdate(HighsInt loc_idxdrop, HighsInt idx_new){
    HighsInt hint { 99999 }; // same number as Micheal in Basis::updatebasis
    // build HVector to add to basis
    if (idx_new < this->lp_.num_row_){ // extract constraint and build HVec
        std::vector<double> select(this->lp_.num_row_);
        select[idx_new] = 1.;
        this->lp_.a_matrix_.productTranspose(this->buffer_, select);
    } else {
        this->buffer_.assign(this->Q_.dim_, 0.);
        this->buffer_[idx_new - this->lp_.num_row_] = 1.;
    }
    HVector newcol;
    stdvec2hvec(this->buffer_, newcol);
    this->B_.ftranCall(newcol, 1.);
    // update basis matrix
    this->B_.update(&newcol, &this->ZT_.back(), &loc_idxdrop, &hint);
    return;
}

HVector AsmSolver::stdvec2hvec(const std::vector<double>& vec, HVector& hvec){
    hvec.setup(vec.size());
    for (size_t i {0}; i < vec.size(); i++){
        if (vec[i] != 0){
            hvec.index[hvec.count] = i;
            hvec.count += 1;
        }
    }
    hvec.array = vec;
    hvec.packFlag = true;
    return hvec;
}

void AsmSolver::feasibility(){
    // TODO hotstart if basis is provided
    if (this->options_.qp_allow_hot_start &&
        this->lp_basis_.valid &&
        this->solution_.value_valid){
        // TODO add check to make sure basis checks out with solution
        this->status_ = HighsStatus::kError; // TODO, for now do not run the active set solver
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
    std::vector<HighsInt> free_idxs;
    for (HighsInt i {0}; i < this->lp_.num_row_; i++){ // loop through constraints
        this->con_status_[i] = this->HighsStatusToAsm(this->lp_basis_.row_status[i], i, false);
        if ( this->isInBasis(this->con_status_[i]) ) this->basis_idxs_.push_back(i); // add index to list of indices
        // constraints shouldn't be free in basis, ignore HighsBasisStatus::kZero and HighsBasisStatus::kNonbasic
    }
    for (HighsInt i {0}; i < this->Q_.dim_; i++){ // loop through variables
        this->var_status_[i] = this->HighsStatusToAsm(this->lp_basis_.col_status[i], i, true);
        // ignore HighsBasisStatus::kNonbasic
        if ( this->isInBasis(this->var_status_[i]) ){
            if ( this->isFreeInBasis(this->var_status_[i]) ) free_idxs.push_back(i + this->lp_.num_row_); // add index to list of indices
            else this->basis_idxs_.push_back(i + this->lp_.num_row_); // if not free then it is active in the basis
        }
    }
    // set nullspace and range dimensions
    this->nullsp_dim_ = (HighsInt) free_idxs.size();
    this->rangsp_dim_ = (HighsInt) this->basis_idxs_.size();
    if (this->rangsp_dim_ + this->nullsp_dim_ != this->Q_.dim_) throw std::logic_error("Active and Free constraints should add up to number of columns!");
    // merge indices
    this->basis_idxs_.insert(this->basis_idxs_.end(),
                             free_idxs.begin(), free_idxs.end());
    std::vector<HighsInt> ordered_basis = this->basis_idxs_; // store buffer
    this->setupBasisMat(ordered_basis); // setup HFactor
    // since basis indices may have been shuffled so that free indices may not trail active ones anymore,
    // set permutation order to match the index sets (A,V) structure
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        for (HighsInt j {0}; j < this->Q_.dim_; j++){
            if ( this->basis_idxs_[i] == ordered_basis[j] ){
                this->basis_perm_[i] = j;
                break;
            }
        }
    }
    // build Reduced Hessian
    this->recomputeExplicit();
    this->refactorize();
    this->computeReducedVecs(); // compute initial reduced gradient and pricing
    return;
}

void AsmSolver::setupBasisMat(std::vector<HighsInt>& basis_idxs){ // TODO do not create constraint mat copy
    HighsSparseMatrix constraint_mat = this->lp_.a_matrix_; // create a copy of the constraint matrix
    constraint_mat.ensureRowwise(); // flip the way in which it is stored
    constraint_mat.format_ = MatrixFormat::kColwise; // but "trick it" into thinking it is still stored columnwise
    HighsInt temp_old_num_row = constraint_mat.num_row_; // flip the number of rows and columns
    constraint_mat.num_row_ = constraint_mat.num_col_; // so that when HFactor uses the matrix
    constraint_mat.num_col_ = temp_old_num_row; // it receives the constraint matrix stored "column wise"
    // where each column is a constraint. its inverse transpose will have as columns the nullspace basis
    this->B_.setup(constraint_mat, basis_idxs); // shuffles basis indices
    this->B_.build();
    return;
}

HighsStatus AsmSolver::run(){
    this->feasibility();
    if ( this->model_status_ == HighsModelStatus::kOptimal ){
        this->model_status_ = HighsModelStatus::kNotset;
        while ( true ) { // ASM iterations
            if ( this->norm(this->red_grad_) < this->options_.primal_feasibility_tolerance ){ // TODO primal residual tolerance?
                if ( this->maximalsteptaken() ) break;
                this->deactivate();
                if ( this->isoptimal() || this->iterlimit() || this->timelimit() || this->nullsizelimit()) break;
            } else {
                this->takeStep();
                std::cout<<this->objective_<<" - "<< this->nullsp_dim_<<"\n"<<std::flush;
            }
        }
        // outside loop but run only if feasibility is successful:
        std::cout<<this->objective_<<" iterations: "<<this->info_.qp_iteration_count<<"\n";
    }
    // TODO record runtime?
    return this->getHighsStatus();
}

void AsmSolver::deactivate(){ // loop through prices to find a constraint to deactivate
    // prices are signed already
    HighsInt bestidx {-1}, bestloc {-1};
    double bestprice = - this->options_.dual_feasibility_tolerance;
    for (HighsInt i {0}; i < this->rangsp_dim_; i++){ // loop through active constraints only
        if ( this->pricing_[i] < bestprice ){
            bestprice = this->pricing_[i];
            bestidx = this->basis_idxs_[i];
            bestloc = i;
        }
    }
    if ( bestidx > -1 ){
        // first return price to original value to update reduced gradient, then update status
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
        this->extend( this->basis_perm_[bestloc] ); // update factorization(s)
        // send deactivated constraint to the end of free-in-basis constraints
        std::vector<HighsInt>::iterator it = this->basis_idxs_.begin() + bestloc;
        std::rotate(it, it + 1, this->basis_idxs_.end());
        it = this->basis_perm_.begin() + bestloc;
        std::rotate(it, it + 1, this->basis_perm_.end());
        this->addNullSpaceDim();
        this->step_taken_ = false; // since problem has been modified TODO remove
    } else this->model_status_ = HighsModelStatus::kOptimal; // set to optimal to break the major loop
    return;
}

void AsmSolver::ratio1(const double tol, const double denom, const double lower,
                       const double upper, const double oldval, const double newval, double& alpha){
    double bound;
    if (denom < - tol && lower > newval ) bound = lower;
    else if ( denom > tol && upper < newval ) bound = upper;
    else return;
    alpha = std::min( alpha, ( bound - oldval ) / denom );
}

void AsmSolver::ratiotest_pass1(const std::vector<double>& newloc,
                                const std::vector<double>& newconvals,
                                const std::vector<double>& denoms){
    this->alpha_relaxed_ = 1.; // we want to minimise it
    this->alpha_ = 1.;
    const double tol = this->options_.factor_pivot_tolerance;
    for (HighsInt i {0}; i < this->Q_.dim_; i++) // loop through variables
        ratio1(tol, this->step_[i], this->lp_relaxed_.col_lower_[i], this->lp_relaxed_.col_upper_[i],
              this->solution_.col_value[i], newloc[i], this->alpha_relaxed_);
    for (HighsInt i {0}; i < this->lp_.num_row_; i++) // loop through constraints
        ratio1(tol, denoms[i], this->lp_relaxed_.row_lower_[i], this->lp_relaxed_.row_upper_[i],
              this->solution_.row_value[i], newconvals[i], this->alpha_relaxed_);
    return;
}

void AsmSolver::ratio2(double& max_pivot, const double denom, const double lower, const double upper,
                       const double oldval, const double newval, const double alpha,
                       const HighsInt idx, HighsInt& newactive_idx, AsmBasisStatus& newactive_status){
    if ( denom < - max_pivot && ( lower - oldval ) / denom < alpha ){
            newactive_idx = idx;
            newactive_status = AsmBasisStatus::kLower;
            max_pivot = - denom;
    } else if ( denom > max_pivot && ( upper - oldval ) / denom < alpha ){
            newactive_idx = idx;
            newactive_status = AsmBasisStatus::kUpper;
            max_pivot = denom;
    }
}

void AsmSolver::ratiotest_pass2(const std::vector<double>& newloc,
                                const std::vector<double>& newconvals,
                                const std::vector<double>& denoms,
                                HighsInt& newactive_idx, AsmBasisStatus& newactive_status){
    double max_pivot = std::max( this->options_.factor_pivot_tolerance, 0. ); // to ensure no division by 0 in ratio2
    for (HighsInt i {0}; i < this->Q_.dim_; i++) // loop through variables
        ratio2(max_pivot, this->step_[i], this->lp_.col_lower_[i], this->lp_.col_upper_[i],
               this->solution_.col_value[i], newloc[i], this->alpha_relaxed_,
               i + this->lp_.num_row_, newactive_idx, newactive_status);
    for (HighsInt i {0}; i < this->lp_.num_row_; i++) // loop through constraints
        ratio2(max_pivot, denoms[i], this->lp_.row_lower_[i], this->lp_.row_upper_[i],
               this->solution_.row_value[i], newconvals[i], this->alpha_relaxed_,
               i, newactive_idx, newactive_status);
    if ( max_pivot <= this->options_.factor_pivot_tolerance) throw std::logic_error("Second pass not activating any constraint!");
    return;
}

void AsmSolver::takeStep(){
    // solve Equality Problem first
    this->delta_.resize(this->red_grad_.size());
    for (size_t i {0}; i < this->red_grad_.size(); i++){
        this->delta_[i] = - this->red_grad_[i]; // TODO, is there a better place to flip sign?
    }
    //this->stepSanity(); // make sure the same identical problem has not been solved yet
    this->LLTsolve(this->delta_);
    this->computeFullStep(this->delta_, this->step_); // then compute full space step
    // ratio test vectors
    this->compute_varvals(1., this->newvarvals_); // compute (potential) x_{k+1}
    this->lp_.a_matrix_.product(this->newconvals_, this->newvarvals_); // a_i^T x_{k+1}
    this->lp_.a_matrix_.product(this->newconpivots_, this->step_); // a_i^T \s
    ratiotest_pass1(this->newvarvals_, this->newconvals_, this->newconpivots_); // ratio test on relaxed instance
    if (this->alpha_relaxed_ < 1.){ //implies there is an activation the relaxed test yielded a step smaller than unity
        HighsInt newactive_idx;
        AsmBasisStatus newactive_status;
        ratiotest_pass2(this->newvarvals_, this->newconvals_, this->newconpivots_, newactive_idx, newactive_status);
        if ( this->alpha_relaxed_ < 0 ) this->alpha_ = 0.; // if we are not moving at all
        else { // TODO these checks can be removed
            this->alpha_ = this->alpha_relaxed_;
            this->compute_varvals(this->alpha_, this->solution_.col_value);
            this->updateObjective();
            this->lp_.a_matrix_.product(this->solution_.row_value, this->solution_.col_value); // a_i^T x_{k+1}
        }
        this->activate(newactive_idx, newactive_status);
    } else { // if no constraint activated and we take the full step
        this->solution_.row_value = this->newconvals_; // don't recompute new constraint values
        this->solution_.col_value = this->newvarvals_; // nor variables' values either
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
    HighsInt loc_remove {-1};
    if (idx < this->lp_.num_row_){
        if ( this->isFreeInBasis(this->con_status_[idx]) ) alreadyinbasis = true;
        //
        if ( this->lp_.row_lower_[idx] == this->lp_.row_upper_[idx] ) this->con_status_[idx] = AsmBasisStatus::kEquality;
        else this->con_status_[idx] = status;
    }
    else {
        HighsInt var_idx = idx - this->lp_.num_row_;
        if ( this->isFreeInBasis(this->var_status_[var_idx]) ) alreadyinbasis = true;
        //
        if ( this->lp_.col_lower_[var_idx] == this->lp_.col_upper_[var_idx] ) this->var_status_[var_idx] = AsmBasisStatus::kEquality;
        else this->var_status_[var_idx] = status;
    }
    // TODO find good rationale to select which constraint to drop
    if (alreadyinbasis){
        std::cout<<"Already in basis! Not happening often...\n";
        // send element in location i to the end of active constraints
        // and shift all free in basis down by one until the old position of the constraint in activation
        for (loc_remove = this->rangsp_dim_; loc_remove < this->Q_.dim_; loc_remove++){
            if (this->basis_idxs_[loc_remove] == idx){
                auto it = this->basis_idxs_.begin();
                std::rotate(it + this->rangsp_dim_, it + loc_remove, it + loc_remove + 1);
                it = this->basis_perm_.begin();
                std::rotate(it + this->rangsp_dim_, it + loc_remove, it + loc_remove + 1);
                break;
            }
        }
        loc_remove -= this->rangsp_dim_;
    } else { 
        loc_remove = this->nullsp_dim_; // drop last, TODO select which to drop
        // change dropped constraint to inactive
        if (this->basis_idxs_.back() < this->lp_.num_row_) this->con_status_[this->basis_idxs_.back()] = AsmBasisStatus::kInactive;
        else this->var_status_[this->basis_idxs_.back() - this->lp_.num_row_] = AsmBasisStatus::kInactive;
        // update HFactor if constraint not already in basis
        // drop the last column in V and substitute it with new index,
        // which then moves to the end of the current active set while all other free indices are shifted by 1 to the end
        this->HUpdate(this->basis_perm_.back(), idx);
        this->basis_idxs_.back() = idx;
        std::rotate(this->basis_idxs_.begin() + this->rangsp_dim_, this->basis_idxs_.end() - 1, this->basis_idxs_.end());
        std::rotate(this->basis_perm_.begin() + this->rangsp_dim_, this->basis_perm_.end() - 1, this->basis_perm_.end());   
    }
    // update factorization if the already-in-basis row was the last one or if we arbitrarily chose to drop the last one
    this->reduce(loc_remove);
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

void AsmSolver::computeReducedVecs(){ // solve B x = (g + Q x_k) to compute Dantzig prices and reduced gradient
    this->computeLocGrad();
    this->pricing_ = this->loc_grad_;
    this->HFtran(this->pricing_); // compute B x = g_k, TODO other types of pricing
    this->red_grad_.assign( std::make_move_iterator(this->pricing_.begin() + this->rangsp_dim_),
                            std::make_move_iterator(this->pricing_.end()));
    this->pricing_.resize(this->rangsp_dim_);    
    this->signPrices();
    return;
}

void AsmSolver::compute_varvals(const double& alpha, std::vector<double>& loc){ // compute x_{k+1}
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        loc[i] = this->solution_.col_value[i] + alpha * this->step_[i];
    }
    return;
}

void AsmSolver::computeFullStep(const std::vector<double>& delta, std::vector<double>& step){ // TODO don't use function arguments
    step.assign(this->rangsp_dim_, 0.);
    step.insert(step.end(), delta.begin(), delta.end());
    this->HBtran(step);
    return;
}

double AsmSolver::computeQuadObjective(const std::vector<double>& vec){
    double sum {0.};
    // matrix is stored in full, but it is symmetric
    for (HighsInt iCol = 0; iCol < this->Q_.dim_; iCol++) {
        for (HighsInt iEl = this->Q_.start_[iCol]; iEl < this->Q_.start_[iCol + 1]; iEl++) {
            if ( this->Q_.index_[iEl] < iCol ) sum += vec[iCol] * this->Q_.value_[iEl] * vec[this->Q_.index_[iEl]];
            else if ( this->Q_.index_[iEl] == iCol ) sum += 0.5 * vec[iCol] * vec[iCol] * this->Q_.value_[iEl];
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
        HighsInt idx = this->basis_idxs_[i];
        if (idx < this->lp_.num_row_) this->pricing_[i] *= static_cast<double>( this->con_status_[idx] );
        else {
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

bool AsmSolver::isInBasis(const AsmBasisStatus& status){
    if (status == AsmBasisStatus::kInactive) return false; // note false return here
    else return true;
}

bool AsmSolver::isFreeInBasis(const AsmBasisStatus& status){
    if (status == AsmBasisStatus::kFreeInBasis) return true;
    else return false;
}

double AsmSolver::norm(const std::vector<double>& vec){
    double sum {0.}; // returns zero if size is null
    for (size_t i {0}; i < vec.size(); i++){
        sum += vec[i] * vec[i];
    }
    return std::sqrt(sum);
}