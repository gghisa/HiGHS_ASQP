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
                HighsHessian& hessian,
                HighsTimer& timer){
    // initialiser solver object
    AsmSolver solver(lp, basis, solution, model_status, hessian, timer);
    solver.feasibility();
    if (solver.getHighsStatus() == HighsStatus::kError) return solver.getHighsStatus();
    solver.run();
    return solver.getHighsStatus();
};

AsmBasis::AsmBasis(const HighsInt& num_var,
                   const HighsInt& num_con)
                   // initialise basis by allocating memory
                   : var_status_(num_var),
                   con_status_(num_con){};

ReducedHessian::ReducedHessian(HighsHessian& Q)
                            : Q_(Q) {};

void ReducedHessian::HSetup(HighsSparseMatrix& constraint_mat, std::vector<HighsInt>& basis_idxs){
    this->B_.setup(constraint_mat, basis_idxs);
};

void ReducedHessian::HBuild(){
    this->B_.build();
}

void ReducedHessian::HBtran(std::vector<double>& vec){
    this->B_.btranCall(vec);
};

void ReducedHessian::HBtran(HVector& vec, const double expected_density){
    this->B_.btranCall(vec, expected_density);
};

void ReducedHessian::HFtran(std::vector<double>& vec){
    this->B_.ftranCall(vec);
};

void ReducedHessian::HFtran(HVector& vec, const double expected_density){
    this->B_.ftranCall(vec, expected_density);
};

void ReducedHessian::HUpdate(HighsInt idx_drop, HighsInt idx_new){ // TODO
    HighsInt hint { 99999 }; // same number as Micheal in Basis::updatebasis
    // build HVector to add to basis
    HVector hvec_aq;
    HFtran(hvec_aq, 1.0);
    // build HVector to point to constraint exiting basis
    HVector hvec_ep;
    hvec_ep.clear();
    hvec_ep.packFlag = true;
    hvec_ep.index[0] = idx_drop;
    hvec_ep.array[idx_drop] = 1.0;
    hvec_ep.count = 1;
    HBtran(hvec_ep, 1.0);
    // update basis matrix
    this->B_.update(&hvec_aq, &hvec_ep, &idx_drop, &hint);
};

HighsInt ReducedHessian::loc(const HighsInt& i, const HighsInt& j) {
    // returns the index for the chol_ vector given the indices for the triangular matrix it represents, stored row-wise as lower triangular
    return i*(i+1)/2 + j;
}

void ReducedHessian::recomputeExplicit(){
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
        z_col[this->Q_.dim_ - this->nullsp_dim_ + i] = 1.;
        HBtran(z_col); // solves returning a column of Z, which we store as a row of Z^T
        this->ZT_.push_back(z_col);
    }
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){// loop over the rows of Z^T
        std::vector<double> row(this->Q_.dim_); // row of Z^T Q
        this->Q_.product(this->ZT_[i], row); // compute it
        double sum {0.};
        for (HighsInt j {0}; j <= i; j++){ // loop through columns of Z, up to the current row of Z^T, to only compute lower triangle of red_hessian_
            for (HighsInt k {0}; k < this->Q_.dim_; k++){ // inner produce of row of Z^T Q with column of Z
                sum += row[k] * this->ZT_[j][k];
            }
            this->chol_[ loc(i,j)] = sum; // should be ordered such that chol_ is row-wise of M
        }
    }
}

void ReducedHessian::refactorize(){
    recomputeExplicit();
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

void ReducedHessian::fsolve(std::vector<double>& vec){
    if ( (HighsInt)vec.size() != this->nullsp_dim_) throw std::logic_error("FSolve requires a vector the size of the nullspace!");
    // solve Ly = b with forward substitution
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){
        for (HighsInt j {0}; j < i; j++){
            vec[i] -= this->chol_[ loc(i,j) ] * vec[j]; 
        }
        vec[i] /= this->chol_[ loc(i,i) ];
    }
}

void ReducedHessian::bsolve(std::vector<double>& vec){
    if ( (HighsInt)vec.size() != this->nullsp_dim_) throw std::logic_error("BSolve requires a vector the size of the nullspace!");
    // solve L^T z = y with backward substitution
    for (HighsInt i {this->nullsp_dim_ - 1}; i > -1; i--){
        for (HighsInt j {i}; j < this->nullsp_dim_; j++){// perform operation in place
            vec[i] -= this->chol_[ loc(i,j) ] * vec[j];
        }
        vec[i] /= this->chol_[ loc(i,i) ];
    }
}

void ReducedHessian::solve(std::vector<double>& vec){
    fsolve(vec);
    bsolve(vec);
}

void ReducedHessian::extend(const HighsInt& loc_deactivated){
    // TODO extend by paying attention to numerical instabilities
    // get new nullspace column
    std::vector<double> z_col(this->Q_.dim_);
    z_col[loc_deactivated] = 1.;
    HBtran(z_col);
    this->ZT_.push_back(z_col);
    // solve L l = Z^T ( Q z_col )
    std::vector<double> vec(this->Q_.dim_);
    this->Q_.product(z_col, vec);
    std::vector<double> sol(this->nullsp_dim_);
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){
        for (HighsInt j {0}; i < this->Q_.dim_; j++){
            sol[i] += this->ZT_[i][j] * vec[j];
        }
    }
    fsolve(sol);
    this->chol_.insert(this->chol_.end(),
                       sol.begin(),
                       sol.end());
    // compute new diagonal element for cholesky factor
    double lambda {0.};
    lambda += this->Q_.objectiveValue(sol);
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){
        lambda -= sol[i] * sol[i];
    }
    this->chol_.push_back( std::sqrt(lambda) );
}

void ReducedHessian::getFullStep(const std::vector<double>& delta, std::vector<double>& step){
    std::fill(step.begin(), step.end(), 0.);
    for (HighsInt i {0}; i < this->Q_.dim_; i++){
        for (HighsInt j {0}; j < this->nullsp_dim_; j++){
            step[i] += this->ZT_[j][i] * delta[j];
        }
    }
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
                     M_(Q),
                     qp_basis_(lp.num_col_, lp.num_row_),
                     loc_grad_(Q.dim_),
                     step_(Q.dim_) {
    if (this->Q_.dim_ != this->lp_.num_col_) throw std::logic_error("Dimension of hessian should match number of columns!");
};

void AsmSolver::addNullSpaceDim(){
    this->M_.nullsp_dim_++;
}

HighsInt AsmSolver::getNullSpaceSize(){
    return this->M_.nullsp_dim_;
}

void AsmSolver::setupBasisMat(){
    // TODO do not create constraint mat copy
    HighsSparseMatrix constraint_mat = this->lp_.a_matrix_; // create a copy of the constraint matrix
    constraint_mat.ensureRowwise(); // flip the way in which it is stored
    constraint_mat.format_ = MatrixFormat::kColwise; // but "trick it" into thinking it is still stored columnwise
    HighsInt temp_old_num_row = constraint_mat.num_row_; // flip the number of rows and columns
    constraint_mat.num_row_ = constraint_mat.num_col_; // so that when the matrix is used by HFactor
    constraint_mat.num_col_ = temp_old_num_row; // it received the constraint matrix "column wise"
    // merge active and free indices
    std::vector<HighsInt> basis_idxs = this->qp_basis_.active_idxs_;
    basis_idxs.insert(basis_idxs.end(), this->qp_basis_.free_idxs_.begin(), this->qp_basis_.free_idxs_.end());
    this->M_.HSetup(constraint_mat, basis_idxs); // where each column is a constraint. its inverse transpose will have as columns the nullspace basis
    this->M_.HBuild(); // factorize method
}

void AsmSolver::setupReducedHessian(){
    // change hessian to square for future
    if (this->Q_.format_ == HessianFormat::kTriangular) this->Q_ = this->Q_.toSquare();
    HighsInt chol_size = getNullSpaceSize() * (getNullSpaceSize() + 1) / 2; // number of elements in lower triangular matrix
    this->M_.chol_.assign(chol_size, 0.);
    this->M_.refactorize();
}

void AsmSolver::setupQpBasis(){
    HighsInt count_basis {0};
    HighsInt count_nonbasis {0};
    // loop through constraints
    for (HighsInt i {0}; i < this->lp_.num_row_; i++){
        this->qp_basis_.con_status_[i] = HighsStatusToAsm(this->lp_basis_.row_status[i], i, false);
        // ignore HighsBasisStatus::kNonbasic
        if ( isInBasis(this->qp_basis_.con_status_[i]) ){
            this->qp_basis_.active_idxs_.push_back(i);
            count_basis++;
            // constraints shouldn't be free in basis, ignore HighsBasisStatus::kZero
        } else {
            this->qp_basis_.inactive_idxs_.push_back(i);
            count_nonbasis++;
        }
    }
    // loop through variables
    for (HighsInt i {0}; i < this->lp_.num_col_; i++){
        HighsInt idx = i + this->lp_.num_row_;
        this->qp_basis_.var_status_[i] = HighsStatusToAsm(this->lp_basis_.col_status[i], i, true);
        // ignore HighsBasisStatus::kNonbasic
        if ( isInBasis(this->qp_basis_.var_status_[i]) ){
            this->qp_basis_.active_idxs_.push_back(idx);
            count_basis++;
            if ( isFreeInBasis(this->qp_basis_.var_status_[i]) ){
                addNullSpaceDim();
                this->qp_basis_.free_idxs_.push_back(idx);
            }
        } else {
            this->qp_basis_.inactive_idxs_.push_back(idx);
            count_nonbasis++;
        }
    }
    // setup HFactor
    setupBasisMat();
    // build Reduced Hessian
    setupReducedHessian();
}

void AsmSolver::feasibility(){
    // TODO hotstart if basis is provided
    // TODO minimize slacks in this first phase
    std::vector<double> col_cost_temp = this->lp_.col_cost_; // store linear costs
    this->lp_.col_cost_.assign(this->lp_.num_col_, 0.); // zero out objective
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
        this->objective_ = feasibility_lp.getObjectiveValue();
        setupQpBasis();
    }
    this->lp_.col_cost_ = col_cost_temp; // reset linear costs to original
};

HighsStatus AsmSolver::getHighsStatus(){
    return this->status_;
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
};

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
    for (HighsInt i {0}; i < this->lp_.num_col_; i++){ // add g to Q x_k
        this->loc_grad_[i] += this->lp_.col_cost_[i];
    }
}

double AsmSolver::norm(const std::vector<double>& vec){
    double sum {0.};
    for (size_t i {0}; i < vec.size(); i++){
        sum += vec[i] * vec[i];
    }
    return std::sqrt(sum);
}

double AsmSolver::computeReducedVecs(){
    // solve B x = (g + Q x_k) to compute (Dantzig?) prices and reduced gradient
    // returns the magnitude of the reduced gradient
    if (getNullSpaceSize() == 0) return 0.;
    else {
        computeLocGrad();
        std::vector<double> vec = this->loc_grad_; // TODO is loc_grad_ storing needed?
        this->M_.HFtran(vec); // compute B x = g_k
        this->pricing_.assign(vec.begin(), vec.end() - getNullSpaceSize()); // TODO, other types of pricing
        this->red_grad_.assign(vec.end() - getNullSpaceSize(), vec.end()); // extract last z elements of the result, i.e. Z^T (g + Q x_k)
        double loc_grad_norm = norm(this->red_grad_);
        return loc_grad_norm;
    }
}

void AsmSolver::deactivate(){ // TODO reduce loops to loop over active constraints only
    // loop through prices to find a constraint to deactivate
    double bestprice = this->pricing_[0];
    HighsInt bestidx = this->qp_basis_.active_idxs_[0];
    HighsInt bestloc = 0;
    // TODO only loop through elements active in basis
    for (size_t i {1}; i < this->qp_basis_.active_idxs_.size(); i++){
        HighsInt idx = this->qp_basis_.active_idxs_[i];
        if (idx < this->lp_.num_row_) { // it is a constraint
            if ( isActiveInequality( this->qp_basis_.con_status_[idx]) && this->pricing_[i] < 0){
                if (this->pricing_[i] < bestprice){
                    bestprice = this->pricing_[i];
                    bestidx = idx;
                    bestloc = i;
                }
            }
        } else {
            idx -= this->lp_.num_row_; // get variable index
            if ( isActiveInequality( this->qp_basis_.var_status_[idx]) && this->pricing_[i] < 0){
                if (this->pricing_[i] < bestprice){
                    bestprice = this->pricing_[i];
                    bestidx = idx + this->lp_.num_row_;
                    bestloc = i;
                }
            }
        }
    }
    // TODO what if the first active is an equality and then none is eligible for deactivation?
    // TODO better maxloop initialisation to avoid this check
    // check that bestprice is indeed negative, in case first price was best but positive
    if ( bestprice < 0. ){
        // deactivation requires updating the basis indices and extending the nullspace
        this->qp_basis_.active_idxs_.erase(this->qp_basis_.active_idxs_.begin() + bestloc);
        this->qp_basis_.free_idxs_.push_back(bestidx);
        addNullSpaceDim();
        // no need to change HFactor. Do you need to remember that the order of the basis has changed? TODO
        // perform corresponding update on HFactor and extend the reduced hessian
        this->M_.extend(bestloc);
    } else this->model_status_ = HighsModelStatus::kOptimal;
}

void AsmSolver::testidx(const HighsInt& idx){
    if (idx < this->lp_.num_row_){ // test constraints
        
    } else { // test bound

    }
}

void AsmSolver::activate(){
    // TODO
};

void AsmSolver::ratiotest(){
    // TODO is it more efficient to loop through all constraints and check the inactive ones
    // or is it more efficient to check index by index? likely the former, so to perform a matrix-vector product with A_ only once.
    for (size_t i {0}; i < this->qp_basis_.free_idxs_.size(); i ++){
        testidx(this->qp_basis_.free_idxs_[i]);
    }
    for (size_t i {0}; i < this->qp_basis_.inactive_idxs_.size(); i ++){
        testidx(this->qp_basis_.inactive_idxs_[i]);
    }
    activate();
}

void AsmSolver::solveREP(){
    // solve reduced equality problem
    // reduced gradient is assumed already up to date
    this->delta_ = this->red_grad_; // TODO is this efficient?
    for (size_t i {0}; i < this->delta_.size(); i++){
        this->delta_[i] *= -1.; // flip sign to solve reduced system
    }
    this->M_.solve(this->delta_);
    // then compute full space step
    this->M_.getFullStep(this->delta_, this->step_); // what if step is null? degeneracy TODO
}

void AsmSolver::run(){
    // TODO set iteration limit
    for (HighsInt i {0}; i < 1; i++){
        if (computeReducedVecs() < this->tol_){
            deactivate();
            if ( this->model_status_ == HighsModelStatus::kOptimal ) break;
        } else {
            solveREP();
            ratiotest();
        }
    }
}