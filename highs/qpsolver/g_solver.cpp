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
                   : basis_idxs_(num_var),
                   nonbasis_idxs_(num_con),
                   var_status_(num_var),
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

void ReducedHessian::HFtran(std::vector<double>& vec){
    this->B_.ftranCall(vec);
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
                     qp_basis_(lp.num_col_, lp.num_row_){};

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
    this->M_.HSetup(constraint_mat, this->qp_basis_.basis_idxs_); // where each column is a constraint. its inverse transpose will have as columns the nullspace basis
    this->M_.HBuild(); // factorize method
}

void AsmSolver::setupReducedHessian(){
    HighsInt chol_size = getNullSpaceSize() * (getNullSpaceSize() + 1) / 2;
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
            this->qp_basis_.basis_idxs_[count_basis] = i;
            count_basis++;
            // constraints shouldn't be free in basis, ignore HighsBasisStatus::kZero
        } else {
            this->qp_basis_.nonbasis_idxs_[count_nonbasis] = i;
            count_nonbasis++;
        }
    }
    // loop through variables
    for (HighsInt i {0}; i < this->lp_.num_col_; i++){
        HighsInt idx = i + this->lp_.num_row_;
        this->qp_basis_.var_status_[i] = HighsStatusToAsm(this->lp_basis_.col_status[i], i, true);
        // ignore HighsBasisStatus::kNonbasic
        if ( isInBasis(this->qp_basis_.var_status_[i]) ){
            this->qp_basis_.basis_idxs_[count_basis] = idx;
            count_basis++;
            if ( isFreeInBasis(this->qp_basis_.var_status_[i]) ) addNullSpaceDim();
        } else {
            this->qp_basis_.nonbasis_idxs_[count_nonbasis] = idx;
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
        this->model_status_ = feasibility_lp.getModelStatus(); // note Optimal in Phase1 is Feasible for ASM
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
    if (status == AsmBasisStatus::kInactive) return false;
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
    if (status == AsmBasisStatus::kFreeInBasis ||
        status == AsmBasisStatus::kInactive) return true;
    else return false;
}

void AsmSolver::run(){
    // run!
}