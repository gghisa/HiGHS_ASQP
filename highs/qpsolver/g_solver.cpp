/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Highs.h"
#include "qpsolver/g_solver.hpp"

AsmSolver::AsmSolver(HighsLp& lp,
                     HighsBasis& basis,
                     HighsSolution& solution,
                     HighsModelStatus& model_status,
                     HighsHessian Q, // TODO pass by reference once Micheal's code removed
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

HighsStatus AsmSolver::getHighsStatus(){ // public function
    return this->status_; // private attribute
}

HighsStatus AsmSolver::run(){
    feasibility();
    if (getHighsStatus() == HighsStatus::kError) return getHighsStatus();
    solve();
    return getHighsStatus();
}

void AsmSolver::HSetup(const HighsSparseMatrix& constraint_mat){
    // basis indices are shuffled around. it is no problem for us that that happens
    this->B_.setup(constraint_mat, this->basis_idxs_);
}

void AsmSolver::HBuild(){
    this->B_.build(); // also for refactorization
}

void AsmSolver::HBtran(std::vector<double>& vec){
    // first apply P
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        this->buffer_[ this->basis_perm_[i] ] = vec[i];
    }
    this->B_.btranCall(this->buffer_); // B^{-T}
    vec = this->buffer_;
}

void AsmSolver::HFtran(std::vector<double>& vec){
    this->B_.ftranCall(vec); // B^{-1}
    // then apply P^T = P^{-1}
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        this->buffer_[ i ] = vec[ this->basis_perm_[i] ];
    }
    vec = this->buffer_;
}

void AsmSolver::HBtran(HVector& vec, const double expected_density){
    this->B_.btranCall(vec, expected_density); // no permutations here, assumed to be taken care of when forming inputs
}

void AsmSolver::HFtran(HVector& vec, const double expected_density){
    this->B_.ftranCall(vec, expected_density); // no permutations here, assumed to be taken care of when forming inputs
}

void AsmSolver::HUpdate(HighsInt loc_idxdrop, HighsInt idx_new){
    HighsInt hint { 99999 }; // same number as Micheal in Basis::updatebasis
    // build HVector to add to basis
    HVector newcol;
    if (idx_new < this->lp_.num_row_){ // extract constraint and build HVec
        std::vector<double> vec(this->Q_.dim_);
        std::vector<double> ep(this->lp_.num_row_);
        ep[idx_new] = 1.;
        this->lp_.a_matrix_.productTranspose(vec, ep);
        newcol = stdvec2hvec(vec);
    }
    else newcol = unit_hvec(idx_new - this->lp_.num_row_); // for a new bound becoming active
    HFtran(newcol, 1.);
    // build HVector to point to constraint exiting basis
    HVector ep = unit_hvec(loc_idxdrop);
    HBtran(ep, 1.);
    // update basis matrix
    this->B_.update(&newcol, &ep, &loc_idxdrop, &hint);
}

HVector AsmSolver::stdvec2hvec(const std::vector<double>& vec){
    HVector hvec;
    hvec.setup(vec.size());
    HighsInt count_nz {0};
    for (size_t i {0}; i < vec.size(); i++){
        if (std::abs(vec[i]) < this->tol_){
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
    this->ZT_.clear();
    HighsInt chol_size = this->nullsp_dim_ * (this->nullsp_dim_ + 1) / 2;
    this->chol_.assign(chol_size, 0.);
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
    // TODO is explicit ZT_ necessary?
    // get new nullspace column
    std::vector<double> z_col(this->Q_.dim_);
    z_col[ loc_deactivated ] = 1.; // permutation taken care of by HBtran
    HBtran(z_col);
    this->ZT_.push_back(z_col); // TODO remove when removing explicit ZT_
    // TODO extend by paying attention to numerical instabilities
    double lambda {0.}; // new diagonal element for cholesky factor
    if (this->nullsp_dim_ > 0){ // nullspace dimension updated just before calling extend()
        // solve L l = Z^T ( Q z_col ) = Z^T sol
        std::vector<double> sol(this->Q_.dim_);
        this->Q_.product(z_col, sol);
        HFtran(sol); // B^{-1} ( sol )
        // select Z^T ( sol ), which is the bottom part of the solution vector above
        sol.erase(sol.begin(), sol.end() - this->nullsp_dim_); // TODO is the other part useful?
        Lsolve(sol);
        this->chol_.insert(this->chol_.end(),
                        sol.begin(),
                        sol.end());
        for (HighsInt i {0}; i < this->nullsp_dim_; i++){
            lambda -= sol[i] * sol[i];
        }
    }
    lambda += 2 * computeQuadObjective(this->ZT_.back());
    if (lambda <= 0) throw std::domain_error("Reduced matrix is either semi- or indefinite!");
    this->chol_.push_back( std::sqrt(lambda) );
}

void AsmSolver::reduce(){
    // TODO update reduced Hessian instead of recomputing it
    recomputeExplicit();
    refactorize();
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

void AsmSolver::setupQpBasis(){
    HighsInt count_basis {0};
    HighsInt count_nonbasis {0};
    // init active and free temporary index and permutation vectors
    std::vector<HighsInt> active_idxs;
    std::vector<HighsInt> free_idxs;
    // loop through constraints
    for (HighsInt i {0}; i < this->lp_.num_row_; i++){
        this->con_status_[i] = HighsStatusToAsm(this->lp_basis_.row_status[i], i, false);
        if ( isInBasis(this->con_status_[i]) ){
            active_idxs.push_back(i); // add index to list of indices
            count_basis++; // constraints shouldn't be free in basis, ignore HighsBasisStatus::kZero and HighsBasisStatus::kNonbasic
        } else count_nonbasis++;
    }
    // loop through variables
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        HighsInt idx = i + this->lp_.num_row_;
        this->var_status_[i] = HighsStatusToAsm(this->lp_basis_.col_status[i], i, true);
        // ignore HighsBasisStatus::kNonbasic
        if ( isInBasis(this->var_status_[i]) ){
            if ( isFreeInBasis(this->var_status_[i]) ) free_idxs.push_back(idx); // add index to list of indices
            else active_idxs.push_back(idx); // if not free then it is active in the basis
            count_basis++;
        } else count_nonbasis++;
    }
    // set nullspace and range dimensions
    this->nullsp_dim_ = (HighsInt) free_idxs.size();
    this->rangsp_dim_ = (HighsInt) active_idxs.size();
    if (this->rangsp_dim_ + this->nullsp_dim_ != this->Q_.dim_) throw std::logic_error("Active and Free constraints should add up to number of columns!");
    // merge indices
    this->basis_idxs_ = active_idxs;
    this->basis_idxs_.insert(this->basis_idxs_.end(),
                             free_idxs.begin(), free_idxs.end());
    // setup HFactor
    setupBasisMat();
    // since basis indices may have been shuffled so that free indices may not trail active ones anymore,
    // set permutation order to match the index sets (A,V) structure
    std::vector<HighsInt> active_idxs_locs(this->rangsp_dim_);
    for (HighsInt i {0}; i < this->rangsp_dim_; i++){
        for (HighsInt j {0}; j < this->Q_.dim_; j++){
            if (active_idxs[i] == this->basis_idxs_[j]){
                active_idxs_locs[i] = j;
                break;
            }
        }
    }
    std::vector<HighsInt> free_idxs_locs(this->nullsp_dim_);
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){
        for (HighsInt j {0}; j < this->Q_.dim_; j++){
            if (free_idxs[i] == this->basis_idxs_[j]){
                free_idxs_locs[i] = j;
                break;
            }
        }
    }
    // merge locations
    this->basis_perm_ = active_idxs_locs;
    this->basis_perm_.insert(this->basis_perm_.end(),
                             free_idxs_locs.begin(), free_idxs_locs.end());
    // then restore order in basis indices
    this->basis_idxs_ = active_idxs;
    this->basis_idxs_.insert(this->basis_idxs_.end(),
                             free_idxs.begin(), free_idxs.end());
    // build Reduced Hessian
    setupReducedHessian();
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
    this->HBuild();
}

void AsmSolver::setupReducedHessian(){
    // change hessian to square for future
    if (this->Q_.format_ == HessianFormat::kTriangular) this->Q_ = this->Q_.toSquare();
    HighsInt chol_size = this->nullsp_dim_ * (this->nullsp_dim_ + 1) / 2; // number of elements in lower triangular matrix
    this->chol_.assign(chol_size, 0.);
    this->recomputeExplicit();
    this->refactorize();
}

void AsmSolver::solve(){
    // TODO set iteration limit
    for (HighsInt i {0}; i < 1000; i++){
        if (computeReducedVecs() < this->tol_){
            deactivate();
            if ( this->model_status_ == HighsModelStatus::kOptimal ) break;
        } else {
            solveREP();
            ratiotest();
        }
    }
    std::cout<<this->objective_<<"\n";
}

void AsmSolver::deactivate(){ // TODO reduce loops to loop over active constraints only
    // loop through prices to find a constraint to deactivate
    signPrices();
    HighsInt bestloc {0};
    double bestprice = this->pricing_[bestloc];
    HighsInt bestidx = this->basis_idxs_[bestloc];
    for (HighsInt i {1}; i < this->rangsp_dim_; i++){ // loop through active constraints only
        HighsInt idx = this->basis_idxs_[i];
        if (idx < this->lp_.num_row_) { // it is a constraint
            if (isActiveInequality( this->con_status_[idx] ) && // TODO change order to improve speed
                this->pricing_[i] < 0 && 
                this->pricing_[i] < bestprice){
                bestprice = this->pricing_[i];
                bestidx = idx;
                bestloc = i;
            }
        } else { // it is a variable bound
            idx -= this->lp_.num_row_; // get variable index
            if (isActiveInequality( this->var_status_[idx]) && // TODO change order to improve speed
                this->pricing_[i] < 0 && 
                this->pricing_[i] < bestprice){
                bestprice = this->pricing_[i];
                bestidx = this->basis_idxs_[i];
                bestloc = i;
            }
        }
    }
    // check that bestprice is indeed negative, in case first price is best but non-negative
    if ( bestprice < 0. ){
        // update status
        if (bestidx < this->lp_.num_row_) this->con_status_[bestidx] = AsmBasisStatus::kFreeInBasis;
        else this->var_status_[bestidx - this->lp_.num_row_] = AsmBasisStatus::kFreeInBasis;
        // extend the reduced hessian, no need to change HFactor
        extend(bestloc); // then extend the basis factorization
        // send deactivated constraint to the end
        std::vector<HighsInt>::iterator it = this->basis_idxs_.begin() + bestloc;
        std::rotate(it, it + 1, this->basis_idxs_.end());
        it = this->basis_perm_.begin() + bestloc;
        std::rotate(it, it + 1, this->basis_perm_.end());
        addNullSpaceDim();
    } else this->model_status_ = HighsModelStatus::kOptimal;
}

void AsmSolver::solveREP(){
    // solve reduced equality problem
    // reduced gradient is assumed already up to date
    this->delta_ = this->red_grad_; // TODO remove and flip the sign when computing new location with this->step_
    for (size_t i {0}; i < this->delta_.size(); i++){
        this->delta_[i] *= -1.; // flip sign to solve reduced system
    }
    this->LLTsolve(this->delta_);
    // then compute full space step
    this->computeFullStep(this->delta_, this->step_); // what if step is null? degeneracy TODO
}

void AsmSolver::ratiotest(){
    // we keep a copy of location, whereas a temporary alpha will just be local to a broken constraint
    // at each constraint, if it is broken, we update alpha
    // with the final alpha being the product of all the alphas (so it is also update as we go)
    this->alpha_ = 1.;
    std::vector<double> newloc(this->Q_.dim_);
    compute_newloc(1., newloc); // compute potential x_{k+1}
    HighsInt newactive_idx = -1; // recall: numbering variables starts from nr of constraints
    AsmBasisStatus newactive_status;
    // it may be more efficient to check bounds first (assume feasibility ofc), so we do that here
    for (HighsInt i {0}; i < this->Q_.dim_; i++){ // loop through variables
        if ( std::abs( this->step_[i] ) > this->tol_ ){ // if change is orthogonal to dimension, no chance of breaking its bounds
            if (this->lp_.col_lower_[i] > newloc[i]){
                double alpha_here = ( this->lp_.col_lower_[i] - this->solution_.col_value[i] ) / this->step_[i];
                if (alpha_here < this->alpha_){
                    newactive_idx = i + this->lp_.num_row_; // store new index
                    newactive_status = AsmBasisStatus::kLower;
                    this->alpha_ = alpha_here;
                }
            } else if (this->lp_.col_upper_[i] < newloc[i]) {
                double alpha_here = ( this->lp_.col_upper_[i] - this->solution_.col_value[i] ) / this->step_[i];
                if (alpha_here < this->alpha_){
                    newactive_idx = i + this->lp_.num_row_;
                    newactive_status = AsmBasisStatus::kUpper;
                    this->alpha_ = alpha_here;
                }
            }
        }
    }
    // loop through inactive (inequality) constraints
    // TODO is it more efficient to each time compute the products?
    std::vector<double> newconvals(this->lp_.num_row_); // vector for new constraint values
    std::vector<double> denoms(this->lp_.num_row_); // vectors for denominators of ratio test formula
    this->lp_.a_matrix_.product(newconvals, newloc); // a_i^T x_{k+1}
    this->lp_.a_matrix_.product(denoms, this->step_); // a_i^T \s
    for (HighsInt i {0}; i < this->lp_.num_row_; i++){
        if ( std::abs(denoms[i]) > this->tol_ ){ // if change is orthogonal to constraint, no chance of breaking its bounds
            if (this->lp_.row_lower_[i] > newconvals[i]){
                double alpha_here = ( this->lp_.row_lower_[i] - this->solution_.row_value[i] ) / denoms[i];
                if (alpha_here < this->alpha_){
                    newactive_idx = i; // store new index
                    newactive_status = AsmBasisStatus::kLower;
                    this->alpha_ = alpha_here;
                }
            } else if (this->lp_.row_upper_[i] < newconvals[i]) {
                double alpha_here = ( this->lp_.row_upper_[i] - this->solution_.row_value[i] ) / denoms[i];
                if (alpha_here < this->alpha_){
                    newactive_idx = i;
                    newactive_status = AsmBasisStatus::kUpper;
                    this->alpha_ = alpha_here;
                }
            }
        }
    }
    compute_newloc(this->alpha_, this->solution_.col_value);
    this->lp_.a_matrix_.product(this->solution_.row_value, this->solution_.col_value); // a_i^T x_{k+1}
    updateObjective();
    // other updates? TODO
    if (newactive_idx != -1){
        activate(newactive_idx, newactive_status);
    }
}

void AsmSolver::activate(const HighsInt& idx, const AsmBasisStatus& status){
    // handle status update
    bool alreadyinbasis {false};
    if (idx < this->lp_.num_row_){
        if ( isFreeInBasis(this->con_status_[idx]) ) alreadyinbasis = true;
        this->con_status_[idx] = status;
    }
    else {
        HighsInt var_idx = idx - this->lp_.num_row_;
        if ( isFreeInBasis(this->var_status_[var_idx]) ) alreadyinbasis = true;
        this->var_status_[var_idx] = status;
    }
    // TODO find good rationale to select which constraint to drop
    if (alreadyinbasis){
        // find location of index and swap it with the first one after the active set
        for (HighsInt i {this->rangsp_dim_}; i < this->Q_.dim_; i++){
            if (this->basis_idxs_[i] == idx){       
                std::swap( this->basis_idxs_[ this->rangsp_dim_ ], this->basis_idxs_[i] );
                std::swap( this->basis_perm_[ this->rangsp_dim_ ], this->basis_perm_[i] );
                break;
            }
        }
    } else { // update HFactor if constraint not already in basis
        // drop the last column in V and substitute it with new index,
        // which then moves to the end of the current active set
        HUpdate(this->basis_perm_.back(), idx);
        // change dropped constraint to inactive
        if (this->basis_idxs_.back() < this->lp_.num_row_) this->con_status_[this->basis_idxs_.back()] = AsmBasisStatus::kInactive;
        else this->var_status_[this->basis_idxs_.back() - this->lp_.num_row_] = AsmBasisStatus::kInactive;
        this->basis_idxs_.back() = idx; // place new index at end of array, dropping the last column in V
        std::swap( this->basis_idxs_[ this->rangsp_dim_ ], this->basis_idxs_.back() ); // move the index before the start of V
        std::swap( this->basis_perm_[ this->rangsp_dim_ ], this->basis_perm_.back() ); // so we need to do the same with perm, to match the location in basis_idxs_   
    }
    removeNullSpaceDim(); // TODO may need to be moved to after reduction
    reduce();
}

void AsmSolver::computeLocGrad(){// g + Q x_k
    this->Q_.product(this->solution_.col_value, this->loc_grad_); // stores result in loc_grad_
    for (HighsInt i {0}; i < this->Q_.dim_; i++){ // add g to Q x_k
        this->loc_grad_[i] += this->lp_.col_cost_[i];
    }
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

void AsmSolver::compute_newloc(const double& alpha, std::vector<double>& loc){
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        this->step_[i] *= alpha;
        loc[i] = this->solution_.col_value[i] + this->step_[i];
    } // compute x_{k+1}
}

void AsmSolver::computeFullStep(const std::vector<double>& delta, std::vector<double>& step){
    step.assign(this->rangsp_dim_, 0.);
    step.insert(step.end(), delta.begin(), delta.end());
    HBtran(step); // is this cheaper than holding the explicit Z^T and using that one?
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
    // assumes that objective is outdated compared to location
    this->objective_ = this->lp_.objectiveValue(this->solution_.col_value);
    this->objective_ += computeQuadObjective(this->solution_.col_value);
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
}

void AsmSolver::addNullSpaceDim(){
    this->nullsp_dim_++;
    this->rangsp_dim_--;
}

void AsmSolver::removeNullSpaceDim(){
    this->nullsp_dim_--;
    this->rangsp_dim_++;
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