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
                     HFactor_basis_(hessian.dim_),
                     var_status_(hessian.dim_),
                     con_status_(lp.num_row_){
    // change hessian to square for better memory access
    if (this->Q_.format_ == HessianFormat::kTriangular) this->Q_ = this->Q_.toSquare();
    this->diagonalQ_ = isQdiagonal();
    if ( this->diagonalQ_ ) std::cout<<"Diagonal Hessian\n"<<std::flush;
}

HighsStatus AsmSolver::getHighsStatus(){ // public function
    return this->status_; // private attribute
}

HighsModelStatus AsmSolver::getHighsModelStatus(){ // public function
    return this->model_status_; // private attribute
}

void AsmSolver::HBtran(std::vector<double>& vec){
    assert( (HighsInt) this->buffer_.size() == this->Q_.dim_);
    // first apply P
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        this->buffer_[ this->basis_perm_[i] ] = vec[ i ];
    } // then solve
    this->B_.btranCall(this->buffer_); // B^{-T}
    vec = this->buffer_;
    return;
}

void AsmSolver::HFtran(std::vector<double>& vec){
    assert( (HighsInt) this->buffer_.size() == this->Q_.dim_);
    this->B_.ftranCall(vec); // first solve for B^{-1}
    // then apply P^T = P^{-1}
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        this->buffer_[ i ] = vec[ this->basis_perm_[i] ];
    }
    vec = this->buffer_;
    return;
}

HighsModelStatus AsmSolver::feasibility(){
    // TODO hotstart if basis is provided
    if (this->options_.qp_allow_hot_start &&
        this->lp_basis_.valid &&
        this->solution_.value_valid){
        // TODO add check to make sure basis checks out with solution
        this->status_ = HighsStatus::kError; // TODO, for now do not run the active set solver
        this->model_status_ = HighsModelStatus::kUnknown;
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
            this->model_status_ = HighsModelStatus::kNotset;
            this->info_.simplex_iteration_count = highs_feasibility.getSimplexIterationCount();
            this->lp_basis_ = highs_feasibility.getBasis();
            this->solution_ = highs_feasibility.getSolution();
            this->updateObjective();
            this->setupQpBasis();
            this->buildRelaxedLp();
        }
    }
    return this->model_status_;
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
        if ( this->con_status_[i] != AsmBasisStatus::kInactive ) this->basis_idxs_.push_back(i); // add index to list of indices
        // constraints shouldn't be free in basis, ignore HighsBasisStatus::kZero and HighsBasisStatus::kNonbasic
    }
    for (HighsInt i {0}; i < this->Q_.dim_; i++){ // loop through variables
        this->var_status_[i] = this->HighsStatusToAsm(this->lp_basis_.col_status[i], i, true);
        // ignore HighsBasisStatus::kNonbasic
        if ( this->var_status_[i] != AsmBasisStatus::kInactive ){
            if ( this->var_status_[i] == AsmBasisStatus::kFreeInBasis ) free_idxs.push_back(i + this->lp_.num_row_); // add index to list of indices
            else this->basis_idxs_.push_back(i + this->lp_.num_row_); // if not free then it is active in the basis
        }
    }
    // set nullspace and range dimensions
    this->nullsp_dim_ = (HighsInt) free_idxs.size();
    this->rangsp_dim_ = (HighsInt) this->basis_idxs_.size();
    assert(this->rangsp_dim_ + this->nullsp_dim_ == this->Q_.dim_);
    // merge indices
    this->basis_idxs_.insert(this->basis_idxs_.end(),
                             free_idxs.begin(), free_idxs.end());
    this->HFactor_basis_ = this->basis_idxs_; // store buffer
    // free indices at the start are necessarily variables, so they are unit vectors for sure
    this->Vi_.assign(this->nullsp_dim_, -1);
    for (HighsInt i {0}; i < this->nullsp_dim_; i++) this->Vi_[i] = free_idxs[i] - this->lp_.num_row_;
    this->setupBasisMat(this->HFactor_basis_); // setup HFactor
    // since basis indices may have been shuffled so that free indices may not trail active ones anymore,
    // set permutation order to match the index sets (A,V) structure
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        for (HighsInt j {0}; j < this->Q_.dim_; j++){
            if ( this->basis_idxs_[i] == this->HFactor_basis_[j] ){
                this->basis_perm_[i] = j;
                break;
            }
        }
    }
    // build Reduced Hessian
    if ( this->nullsp_dim_ > 0 ){
        this->recomputeRedHessian();
        this->atFSEP_ = false; // so that first minor loop is initiated
    }
    this->computeReducedVecs(); // compute initial reduced gradient and pricing
    return;
}

void AsmSolver::setupBasisMat(std::vector<HighsInt>& basis_idxs){ // TODO do not create constraint mat copy
    HighsSparseMatrix constraint_mat = this->lp_.a_matrix_; // create a copy of the constraint matrix
    constraint_mat.ensureRowwise(); // flip the way in which it is stored
    // but "trick it" into thinking it is still stored columnwise
    // where each column is a constraint. its inverse transpose will have as columns the nullspace basis
    // use same setup as Micheal, flip the number of rows and columns so that when HFactor uses the matrix
    // it receives the constraint matrix stored "column wise"
    this->B_.setup( constraint_mat.num_row_, constraint_mat.num_col_, constraint_mat.start_.data(),
                    constraint_mat.index_.data(), constraint_mat.value_.data(), basis_idxs.data() );
    this->B_.build();
    return;
}

void AsmSolver::buildRelaxedLp(){
    const double tol = 0.1 * this->options_.primal_feasibility_tolerance;
    this->lp_relaxed_.row_lower_.assign(this->lp_.num_row_, 0.);
    this->lp_relaxed_.row_upper_.assign(this->lp_.num_row_, 0.);
    for (HighsInt i {0}; i < this->lp_.num_row_; i++){ // relax all constraints (equalities too)
        this->lp_relaxed_.row_lower_[i] = this->lp_.row_lower_[i] - tol;
        this->lp_relaxed_.row_upper_[i] = this->lp_.row_upper_[i] + tol;
    }
    this->lp_relaxed_.col_lower_.assign(this->Q_.dim_, 0.);
    this->lp_relaxed_.col_upper_.assign(this->Q_.dim_, 0.);
    for (HighsInt i {0}; i < this->Q_.dim_; i++){ // relax all variables' bounds
        this->lp_relaxed_.col_lower_[i] = this->lp_.col_lower_[i] - tol;
        this->lp_relaxed_.col_upper_[i] = this->lp_.col_upper_[i] + tol;
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
    return;
}

void AsmSolver::compute_varvals(const double& alpha, std::vector<double>& loc){ // compute x_{k+1}
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        // this->step_[i] *= alpha; // TODO
        loc[i] = this->solution_.col_value[i] + alpha * this->step_[i];
    }
    return;
}

double AsmSolver::computeQuadObjective(const std::vector<double>& vec){ // TODO move to HighsHessian
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

bool AsmSolver::keepLooping(){
    if (this->info_.qp_iteration_count >= this->options_.qp_iteration_limit){ // iteration limit
        this->model_status_ = HighsModelStatus::kIterationLimit;
        this->status_ = HighsStatus::kWarning; // TODO ok?
        return false;
    }
    if (this->timer_.read() >= this->options_.time_limit){ // time limit
        this->model_status_ = HighsModelStatus::kTimeLimit;
        this->status_ = HighsStatus::kWarning; // TODO ok?
        return false;
    }
    if ( this->nullsp_dim_ > this->options_.qp_nullspace_limit){ // nullspace size limit
        this->model_status_ = HighsModelStatus::kSolveError;
        this->status_ = HighsStatus::kError;
        return false;
    }
    return true;
}

bool AsmSolver::isOptimal(){ // break loop if optimality check is positive during deactivation
    if ( this->model_status_ == HighsModelStatus::kOptimal ){
        this->status_ = HighsStatus::kOk;
        return true;
    }
    return false;
}

void AsmSolver::reinvertBasis(){
    this->HFactor_basis_.assign(this->Q_.dim_ + this->lp_.num_row_, -1);
    this->B_.build(); // TODO are indexes changed?
    this->recomputeRedHessian();
    this->num_basis_updates_ = 0;
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

double AsmSolver::norm(const std::vector<double>& vec){
    double sum {0.}; // returns zero if size is null
    for (size_t i {0}; i < vec.size(); i++){
        sum += vec[i] * vec[i];
    }
    return std::sqrt(sum);
}

void AsmSolver::changeStatus(const HighsInt& idx, const AsmBasisStatus& newstatus){
    if (idx < this->lp_.num_row_) this->con_status_[idx] = newstatus;
    else this->var_status_[idx - this->lp_.num_row_] = newstatus;
}


void AsmSolver::buildConstraint(const HighsInt& idx, HVector& hvec){
    if (idx < this->lp_.num_row_){
        std::vector<double> select(this->lp_.num_row_);
        select[idx] = 1.;
        this->lp_.a_matrix_.productTranspose(this->buffer_, select);
    } else {
        this->buffer_.assign(this->Q_.dim_, 0.);
        this->buffer_[idx - this->lp_.num_row_] = 1.;
    }
    stdvec2hvec(this->buffer_, hvec);
    return;
}

void AsmSolver::fromPaddingToActive(const HighsInt& loc){
    // send (activated) formerly free in basis index to end of active constraints
    std::vector<HighsInt>::iterator it = this->basis_idxs_.begin();
    std::rotate(it + this->rangsp_dim_, it + loc, it + loc + 1);
    it = this->basis_perm_.begin();
    std::rotate(it + this->rangsp_dim_, it + loc, it + loc + 1);
    return;
}

void AsmSolver::fromActiveToPadding(const HighsInt& loc){
    // send (deactivated) constraint index to the end of free-in-basis constraints
    // from the active set
    std::vector<HighsInt>::iterator it = this->basis_idxs_.begin() + loc;
    std::rotate(it, it + 1, this->basis_idxs_.end());
    it = this->basis_perm_.begin() + loc;
    std::rotate(it, it + 1, this->basis_perm_.end());
    return;
}

AsmBasisStatus AsmSolver::getAsmBasisStatus(const HighsInt& idx){
    if ( idx < this->lp_.num_row_ ) return con_status_[idx];
    else return var_status_[idx - this->lp_.num_row_];
}

bool AsmSolver::isQdiagonal(){
    // works with square or triangular HighsHessian, which are identical if diagonal
    HighsInt dim;
    for (HighsInt iCol = 0; iCol < this->Q_.dim_; iCol++) {
        dim = this->Q_.start_[iCol + 1] - this->Q_.start_[iCol];
        if ( dim == 0 || ( dim == 1 && this->Q_.index_[this->Q_.start_[iCol]] == iCol ) ) continue; // column is empty
        else return false;
    }
    return true;
}