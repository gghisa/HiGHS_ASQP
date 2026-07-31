/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Highs.h"
#include "qpsolver/g_solver.hpp"

HighsStatus gQP(HighsLp& lp,
                HighsBasis& basis,
                HighsSolution& solution,
                HighsModelStatus& model_status,
                HighsHessian hessian, // TODO make pass by reference when Micheal's code doesnt require it to be triangular anymore
                HighsTimer& timer){
    AsmSolver solver(lp, basis, solution, model_status, hessian, timer);
    solver.feasibility();
    if (solver.getHighsStatus() == HighsStatus::kError) return solver.getHighsStatus();
    solver.run();
    return solver.getHighsStatus();
}

AsmSolver::AsmSolver(HighsLp& lp,
                     HighsBasis& basis,
                     HighsSolution& solution,
                     HighsModelStatus& model_status,
                     HighsHessian& Q,
                     HighsTimer& timer)
                     : lp_(lp),
                     lp_basis_(basis),
                     solution_(solution),
                     model_status_(model_status),
                     Q_(Q),
                     timer_(timer),
                     buffer_(lp.num_col_),
                     loc_grad_(Q.dim_),
                     step_(Q.dim_),
                     var_status_(lp.num_col_),
                     con_status_(lp.num_row_){}

void AsmSolver::HSetup(HighsSparseMatrix& constraint_mat){
    this->basis_build_ = this->basis_idxs_; // because apparently HBuild changes the order of the vector
    this->B_.setup(constraint_mat, this->basis_build_);
}

void AsmSolver::HBuild(){
    this->B_.build();
}

void AsmSolver::HBtran(std::vector<double>& vec){
    // first apply P
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        this->buffer_[ this->basis_perm_[i] ] = vec[i];
    }
    this->B_.btranCall(this->buffer_); // B^{-T}
    vec = this->buffer_;
}

void AsmSolver::HBtran(HVector& vec, const double expected_density){ // TODO
    this->B_.btranCall(vec, expected_density);
}

void AsmSolver::HFtran(std::vector<double>& vec){
    this->B_.ftranCall(vec); // B^{-1}
    // then apply P^T = P^{-1}
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        this->buffer_[ i ] = vec[ this->basis_perm_[i] ];
    }
    vec = this->buffer_;
}

void AsmSolver::HFtran(HVector& vec, const double expected_density){ // TODO
    this->B_.ftranCall(vec, expected_density);
}

HVector AsmSolver::stdvec2hvec(const std::vector<double>& vec){
    HVector hvec;
    HighsInt count_nz {0};
    for (size_t i {0}; i < vec.size(); i++){
        if (vec[i] != 0) hvec.index.push_back(i);
    }
    hvec.array = vec;
    hvec.count = count_nz;
    hvec.packFlag = true;
    return hvec;
}

std::vector<double> AsmSolver::hvec2stdvec(const HVector& hvec){
    return hvec.array; // is this really it? TODO is HVector.array dense?
}

HVector AsmSolver::unit_hvec(const HighsInt& p){
    HVector hvec;
    hvec.index.push_back(p);
    hvec.array.push_back(1);
    hvec.count = 1;
    //hvec.packFlag = true; // TODO what is this?
    return hvec;
}

HVector AsmSolver::build_aq(const HighsInt& idx){
    // asume idx < this->lp_.num_row_
    std::vector<double> vec(this->Q_.dim_);
    std::vector<double> ep(this->Q_.dim_);
    ep[idx] = 1.;
    this->lp_.a_matrix_.product(vec, ep);
    return stdvec2hvec(vec);
}

void AsmSolver::HUpdate(HighsInt idx_drop, HighsInt idx_new){ // TODO
    HighsInt hint { 99999 }; // same number as Micheal in Basis::updatebasis
    // build HVector to add to basis
    HVector hvec_aq;
    if (idx_new < this->lp_.num_row_) hvec_aq = build_aq(idx_new);
    else hvec_aq = unit_hvec(idx_new - this->lp_.num_row_); // for a new bound becoming active
    HFtran(hvec_aq, 1.0); // compute B^{-1} a_q according to guidelines... why?
    // build HVector to point to constraint exiting basis
    HVector ep = unit_hvec(idx_drop);
    HBtran(ep, 1.0);
    // update basis matrix
    this->B_.update(&hvec_aq, &ep, &idx_drop, &hint);
}

HighsInt AsmSolver::loc(const HighsInt& i, const HighsInt& j) {
    // returns the index for the chol_ vector given the indices for the triangular matrix it represents, stored row-wise as lower triangular
    return i*(i+1)/2 + j;
}

void AsmSolver::recomputeExplicit(){
    // assume:
    // 1. number of nullspace dimensions known
    // 2. B_ factorization completed
    // then:
    // extracts explicit Z^T and computes explicitly the reduced Hessian
    // useful when starting point of ASM provides a non-empty null-space
    // computes the factorization row by row according to Cholesky—Banachiewicz
    // TODO pivoting
    // while Q is dense, we can't guarantee Z is too, so M is treated as dense, and likewise its factors
    // if we order M by the largest of its diagonal entries we need to first compute it all
    // BUILD RED HESSIAN FIRST
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){// get Z^T
        std::vector<double> z_col(this->Q_.dim_);
        z_col[this->rangsp_dim_ + i] = 1.;
        HBtran(z_col); // solves returning a column of Z, which we store as a row of Z^T
        this->ZT_.push_back(z_col);
    }
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){// loop over the rows of Z^T
        std::vector<double> row(this->Q_.dim_); // row of Z^T Q
        this->Q_.product(this->ZT_[i], row); // compute it
        for (HighsInt j {0}; j <= i; j++){ // loop through columns of Z, up to the current row of Z^T, to only compute lower triangle of red_hessian_
            double sum {0.};
            for (HighsInt k {0}; k < this->Q_.dim_; k++){ // inner produce of row of Z^T Q with column of Z
                sum += row[k] * this->ZT_[j][k];
            }
            this->chol_[ loc(i,j)] = sum; // should be ordered such that chol_ is row-wise of M
        }
    }
}

void AsmSolver::refactorize(){
    // perform cholesky factorization in place
    for (HighsInt i {0}; i<this->nullsp_dim_; i++){
        for (HighsInt j {0}; j <= i; j++){
            if (i == j){ // diagonal element
                for (HighsInt k {0}; k < j; k++){
                    double row_el = this->chol_[ loc(i,k)];
                    this->chol_[ loc(i,i)] -= row_el * row_el;
                }
                this->chol_[ loc(i,i)] = std::sqrt(this->chol_[ loc(i,i)]);
            }
            else { // off diagonal element
                for (HighsInt k {0}; k < j; k++){
                    this->chol_[ loc(i,j)] -= this->chol_[ loc(i,k)] * this->chol_[ loc(j,k)];
                }
                this->chol_[ loc(i,j)] /= this->chol_[ loc(j,j)];
            }
        }
    }
}

void AsmSolver::Lsolve(std::vector<double>& vec){
    if ( (HighsInt)vec.size() != this->nullsp_dim_) throw std::logic_error("Fw solve requires a vector the size of the nullspace!");
    // solve Ly = b with forward substitution
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){
        for (HighsInt j {0}; j < i; j++){
            vec[i] -= this->chol_[ loc(i,j) ] * vec[j]; 
        }
        vec[i] /= this->chol_[ loc(i,i) ];
    }
}

void AsmSolver::LTsolve(std::vector<double>& vec){
    if ( (HighsInt)vec.size() != this->nullsp_dim_) throw std::logic_error("Bw solve requires a vector the size of the nullspace!");
    // solve L^T z = y with backward substitution
    HighsInt limit = this->nullsp_dim_ - 1;
    for (HighsInt i {limit}; i > -1; i--){
        for (HighsInt j {limit}; j > i; j--){// perform operation in place
            vec[i] -= this->chol_[ loc(i,j) ] * vec[j];
        }
        vec[i] /= this->chol_[ loc(i,i) ];
    }
}

void AsmSolver::LLTsolve(std::vector<double>& vec){
    Lsolve(vec);
    LTsolve(vec);
}

void AsmSolver::extend(const HighsInt& loc_deactivated){
    // TODO extend by paying attention to numerical instabilities
    // TODO is explicit ZT_ necessary?
    // get new nullspace column
    std::vector<double> z_col(this->Q_.dim_);
    z_col[ this->basis_perm_[loc_deactivated] ] = 1.;
    HBtran(z_col);
    this->ZT_.push_back(z_col);
    double lambda {0.}; // new diagonal element for cholesky factor
    if (this->nullsp_dim_ > 0){
        // solve L l = Z^T ( Q z_col ) = Z^T vec
        std::vector<double> vec(this->Q_.dim_);
        this->Q_.product(z_col, vec);
        std::vector<double> sol(this->nullsp_dim_);
        for (HighsInt i {0}; i < this->nullsp_dim_; i++){
            for (HighsInt j {0}; j < this->Q_.dim_; j++){
                sol[i] += this->ZT_[i][j] * vec[j];
            }
        }
        Lsolve(sol);
        this->chol_.insert(this->chol_.end(),
                        sol.begin(),
                        sol.end());
        for (HighsInt i {0}; i < this->nullsp_dim_; i++){
            lambda -= sol[i] * sol[i];
        }
    }
    lambda += 2 * this->Q_.objectiveValue(z_col);
    if (lambda <= 0) throw std::domain_error("Reduced matrix is either semi- or indefinite!");
    this->chol_.push_back( std::sqrt(lambda) );
}

void AsmSolver::getFullStep(const std::vector<double>& delta, std::vector<double>& step){
    step.assign(this->rangsp_dim_, 0.);
    step.insert(step.end(), delta.begin(), delta.end());
    HBtran(step); // is this cheaper than holding the explicit Z^T and using that one?
}

void AsmSolver::addNullSpaceDim(){
    this->nullsp_dim_++;
    this->rangsp_dim_--;
}

void AsmSolver::removeNullSpaceDim(){
    this->nullsp_dim_--;
    this->rangsp_dim_++;
}

HighsInt AsmSolver::bperm(const HighsInt& idx_loc){
    // returns the permuted index given an index that matches the [ active | free ] basis matrix partitioning
    return this->basis_perm_[idx_loc];
}

void AsmSolver::setupBasisMat(){
    // TODO do not create constraint mat copy
    HighsSparseMatrix constraint_mat = this->lp_.a_matrix_; // create a copy of the constraint matrix
    constraint_mat.ensureRowwise(); // flip the way in which it is stored
    constraint_mat.format_ = MatrixFormat::kColwise; // but "trick it" into thinking it is still stored columnwise
    HighsInt temp_old_num_row = constraint_mat.num_row_; // flip the number of rows and columns
    constraint_mat.num_row_ = constraint_mat.num_col_; // so that when HFactor uses the matrix
    constraint_mat.num_col_ = temp_old_num_row; // it receives the constraint matrix stored "column wise"
    this->HSetup(constraint_mat); // where each column is a constraint. its inverse transpose will have as columns the nullspace basis
    this->HBuild(); // factorize method, TODO why does it change the order of the basis_idxs?
}

void AsmSolver::setupReducedHessian(){
    // change hessian to square for future
    if (this->Q_.format_ == HessianFormat::kTriangular) this->Q_ = this->Q_.toSquare();
    HighsInt chol_size = this->nullsp_dim_ * (this->nullsp_dim_ + 1) / 2; // number of elements in lower triangular matrix
    this->chol_.assign(chol_size, 0.);
    this->recomputeExplicit();
    this->refactorize();
}

void AsmSolver::setupQpBasis(){
    HighsInt count_basis {0};
    HighsInt count_nonbasis {0};
    // init active and free temporary index and permutation vectors
    std::vector<HighsInt> active_idxs;
    std::vector<HighsInt> active_blocs;
    std::vector<HighsInt> free_idxs;
    std::vector<HighsInt> free_blocs;
    // loop through constraints
    for (HighsInt i {0}; i < this->lp_.num_row_; i++){
        this->con_status_[i] = HighsStatusToAsm(this->lp_basis_.row_status[i], i, false);
        // ignore HighsBasisStatus::kNonbasic
        if ( isInBasis(this->con_status_[i]) ){
            active_idxs.push_back(i); // add index to list of indices
            active_blocs.push_back(count_basis); // add index location to list of indices locations
            count_basis++;
            // constraints shouldn't be free in basis, ignore HighsBasisStatus::kZero
        } else {
            this->inactive_idxs_.push_back(i);
            count_nonbasis++;
        }
    }
    // loop through variables
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        HighsInt idx = i + this->lp_.num_row_;
        this->var_status_[i] = HighsStatusToAsm(this->lp_basis_.col_status[i], i, true);
        // ignore HighsBasisStatus::kNonbasic
        if ( isInBasis(this->var_status_[i]) ){
            if ( isFreeInBasis(this->var_status_[i]) ){ // count nullspace dimension at the end with size of array
                free_idxs.push_back(idx); // add index to list of indices
                free_blocs.push_back(count_basis); // add index location to list of indices locations
            } else { // if not free then it is active in the basis
                active_idxs.push_back(idx); // add index to list of indices
                active_blocs.push_back(count_basis); // add index location to list of indices locations
            }
            count_basis++;
        } else {
            this->inactive_idxs_.push_back(idx);
            count_nonbasis++;
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
    // merge permutation indices
    this->basis_perm_ = active_blocs;
    this->basis_perm_.insert(this->basis_perm_.end(),
                             free_blocs.begin(), free_blocs.end());
    // setup HFactor
    setupBasisMat();
    // build Reduced Hessian
    setupReducedHessian();
}

void AsmSolver::feasibility(){
    // TODO hotstart if basis is provided
    // TODO minimize slacks in this first phase
    std::vector<double> col_cost_temp = this->lp_.col_cost_; // store linear costs
    this->lp_.col_cost_.assign(this->Q_.dim_, 0.); // zero out objective
    Highs feasibility_lp;
    feasibility_lp.passModel(this->lp_);
    feasibility_lp.setOptionValue("presolve", kHighsOnString); // presolving phase1 makes it faster, im guessing the postsolve is included
    feasibility_lp.setOptionValue("output_flag", false); // don't print anything
    feasibility_lp.setOptionValue("simplex_strategy", kSimplexStrategyDual); // specifying what solver to use in case a basis is set that is known to be either primal or dual feasible
    // use dual simplex if the objective value is all zeros, beacuse that means dual feasibility is guaranteed
    this->status_ = feasibility_lp.run();
    if (this->status_ != HighsStatus::kError){ // why not returning after extracting the model status too?
        this->model_status_ = HighsModelStatus::kNotset; // note Optimal in Phase1 is Feasible for ASM
        this->lp_basis_ = feasibility_lp.getBasis();
        this->solution_ = feasibility_lp.getSolution();
        setupQpBasis();
    }
    this->lp_.col_cost_ = col_cost_temp; // reset linear costs to original
    updateObjective();
}

HighsStatus AsmSolver::getHighsStatus(){ // public function
    return this->status_; // private attribute
}
// convert Highs Status to Asm Status
AsmBasisStatus AsmSolver::HighsStatusToAsm(const HighsBasisStatus& status, const HighsInt i, const bool variable){
    if (variable){ // it is a variable this should be a fixed variable and needs presolve
        if (this->lp_.col_lower_[i] == this->lp_.col_upper_[i]) return AsmBasisStatus::kEquality;
    } else if (this->lp_.row_lower_[i] == this->lp_.row_upper_[i]) return AsmBasisStatus::kEquality;
    // if no equality is found:
    if(status == HighsBasisStatus::kLower) return AsmBasisStatus::kLower;
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

bool AsmSolver::isActive(const AsmBasisStatus& status){
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

bool AsmSolver::isInactive(const AsmBasisStatus& status){
    if (status == AsmBasisStatus::kInactive ||
        status == AsmBasisStatus::kFreeInBasis) return true;
    else return false;
}

void AsmSolver::computeLocGrad(){// g + Q x_k
    this->Q_.product(this->solution_.col_value, this->loc_grad_); // stores result in loc_grad_
    for (HighsInt i {0}; i < this->Q_.dim_; i++){ // add g to Q x_k
        this->loc_grad_[i] += this->lp_.col_cost_[i];
    }
}

double AsmSolver::norm(const std::vector<double>& vec){
    double sum {0.}; // returns zero if size is null
    for (size_t i {0}; i < vec.size(); i++){
        sum += vec[i] * vec[i];
    }
    return std::sqrt(sum);
}

double AsmSolver::computeReducedVecs(){
    // solve B x = (g + Q x_k) to compute (Dantzig?) prices and reduced gradient
    computeLocGrad();
    std::vector<double> vec = this->loc_grad_; // TODO is loc_grad_ storing needed?
    this->HFtran(vec); // compute B x = g_k
    this->pricing_.assign(vec.begin(), vec.end() - this->nullsp_dim_); // TODO, other types of pricing
    this->red_grad_.assign(vec.end() - this->nullsp_dim_, vec.end()); // extract last z elements of the result, i.e. Z^T (g + Q x_k)
    // returns the magnitude of the reduced gradient (0 if of null dimension)
    return norm(this->red_grad_);
}

void AsmSolver::deactivate(){ // TODO reduce loops to loop over active constraints only
    // loop through prices to find a constraint to deactivate
    double bestprice = this->pricing_[0] * static_cast<double>( this->con_status_[0] );
    HighsInt bestidx = this->basis_idxs_[0];
    HighsInt bestloc = 0;
    for (HighsInt i {1}; i < this->rangsp_dim_; i++){ // loop through active constraints only
        HighsInt idx = this->basis_idxs_[i];
        if (idx < this->lp_.num_row_) { // it is a constraint
            double price = this->pricing_[i] * static_cast<double>( this->con_status_[idx] );
            if (isActiveInequality( this->con_status_[idx] ) && // TODO change order to improve speed
                price < 0 && 
                this->pricing_[i] < bestprice){
                bestprice = this->pricing_[i];
                bestidx = idx;
                bestloc = i;
            }
        } else { // it is a variable bound
            idx -= this->lp_.num_row_; // get variable index
            double price = this->pricing_[i] * static_cast<double>( this->var_status_[idx] );
            if (isActiveInequality( this->var_status_[idx]) && // TODO change order to improve speed
                price < 0 && 
                this->pricing_[i] < bestprice){
                bestprice = this->pricing_[i];
                bestidx = idx + this->lp_.num_row_;
                bestloc = i;
            }
        }
    }
    // check that bestprice is indeed negative, in case first price is best but non-negative
    if ( bestprice < 0. ){
        // swap of deactivated constraint with the last active one
        HighsInt newloc = this->rangsp_dim_ - 1;
        std::swap( this->basis_idxs_[bestloc], this->basis_idxs_[newloc] );
        std::swap( this->basis_perm_[bestloc], this->basis_perm_[newloc] );
        this->extend(bestloc); // extend the reduced hessian, no need to change HFactor
        addNullSpaceDim();
    } else this->model_status_ = HighsModelStatus::kOptimal;
}

void AsmSolver::compute_newloc(const double& alpha, std::vector<double>& loc){
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        this->step_[i] *= alpha;
        this->alpha_ *= alpha;
        loc[i] = this->solution_.col_value[i] + this->step_[i];
    } // compute x_{k+1}
}

void AsmSolver::activate(const HighsInt& idx, const AsmBasisStatus& status){
    // TODO make inactive_idxs_ ordered set to improve from O(n) to O(log n) deletion
    // TODO carefully select which constraint to drop
    HUpdate(this->basis_idxs_[ this->basis_perm_[this->Q_.dim_] ], // drop the last column in V
            idx);
    HighsInt end = this->Q_.dim_ - 1;
    this->basis_idxs_[end] = idx; // place new index at end of array, dropping the last column in V
    std::swap( this->basis_idxs_[ this->rangsp_dim_ ], this->basis_idxs_[ end ] ); // move the index before the start of V
    
}

void AsmSolver::ratiotest(){
    // we keep a copy of location, whereas a temporary alpha will just be local to a broken constraint
    // at each constraint, if it is broken, we update alpha
    // with the final alpha being the product of all the alphas (so it is also update as we go)
    std::vector<double> newloc(this->Q_.dim_);
    compute_newloc(1., newloc);
    HighsInt newactive_idx = -1; // recall: numbering variables starts from nr of constraints
    AsmBasisStatus newactive_status;
    // it may be more efficient to check bounds first (assume feasibility ofc), so we do that here
    for (HighsInt i {0}; i < this->Q_.dim_; i++){ // loop through variables
        if ( !isActive( this->var_status_[i] ) ){// if bound is inactive
            HighsInt idx = i + this->lp_.num_row_;
            if (this->lp_.col_lower_[i] > newloc[i]){
                // compute new alpha
                double alpha_here = ( this->lp_.col_lower_[i] - this->solution_.col_value[i] ) / this->step_[i];
                compute_newloc(alpha_here, newloc); // update alpha and temporary location
                newactive_idx = idx; // store new index
                newactive_status = AsmBasisStatus::kLower;
            } else if (this->lp_.col_upper_[i] < newloc[i]) {
                double alpha_here = ( this->lp_.col_upper_[i] - this->solution_.col_value[i] ) / this->step_[i];
                compute_newloc(alpha_here, newloc);
                newactive_idx = idx;
                newactive_status = AsmBasisStatus::kUpper;
            }
        }
    }
    // loop through inactive (inequality) constraints
    std::vector<double> convals(this->lp_.num_row_); // vector for constraint values
    this->lp_.a_matrix_.product(convals, this->solution_.col_value); // a_i^T x_{k+1}
    std::vector<double> denoms(this->lp_.num_row_); // vectors for denominators of ratio test formula
    this->lp_.a_matrix_.product(denoms, this->step_); // a_i^T \s
    for (HighsInt i {0}; i < this->lp_.num_row_; i++){
        if ( !isActive( this->con_status_[i] ) ){// if constraint is inactive
            if (this->lp_.row_lower_[i] > convals[i]){
                double alpha_here = ( this->lp_.row_lower_[i] - convals[i] ) / denoms[i];
                compute_newloc(alpha_here, newloc); // update alpha and temporary location
                newactive_idx = i; // store new index
                newactive_status = AsmBasisStatus::kLower;
            } else if (this->lp_.row_upper_[i] < convals[i]) {
                double alpha_here = ( this->lp_.row_lower_[i] - convals[i] ) / denoms[i];
                compute_newloc(alpha_here, newloc); // update alpha and temporary location
                newactive_idx = i; // store new index
                newactive_status = AsmBasisStatus::kUpper;
            }
        }
    }
    if (newactive_idx != -1) activate(newactive_idx, newactive_status);
    // other updates? TODO
    this->solution_.col_value = newloc;
    updateObjective();
}

void AsmSolver::updateObjective(){
    // assumes that objective is outdated compared to location
    this->objective_  = 0.;
    this->objective_ += this->Q_.objectiveValue(this->solution_.col_value);
    this->objective_ += this->lp_.objectiveValue(this->solution_.col_value);
}

void AsmSolver::solveREP(){
    // solve reduced equality problem
    // reduced gradient is assumed already up to date
    this->delta_ = this->red_grad_; // TODO is this efficient?
    for (size_t i {0}; i < this->delta_.size(); i++){
        this->delta_[i] *= -1.; // flip sign to solve reduced system
    }
    this->LLTsolve(this->delta_);
    // then compute full space step
    this->getFullStep(this->delta_, this->step_); // what if step is null? degeneracy TODO
}

void AsmSolver::run(){
    // TODO set iteration limit
    for (HighsInt i {0}; i < 2; i++){
        if (computeReducedVecs() < this->tol_){
            deactivate();
            if ( this->model_status_ == HighsModelStatus::kOptimal ) break;
        } else {
            solveREP();
            ratiotest();
        }
    }
}