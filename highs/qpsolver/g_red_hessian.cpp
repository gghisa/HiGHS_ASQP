/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Highs.h"
#include "qpsolver/g_red_hessian.hpp"

ReducedHessian::ReducedHessian(HighsHessian& Q)
                                : Q_(Q){
};

const HighsInt ReducedHessian::getNullSpaceSize(){
    return this->nullsp_dim_;
};

void ReducedHessian::addOneNullSpaceDim(){
    this->nullsp_dim_++;
}

void ReducedHessian::Hsetup(HighsSparseMatrix& constraint_mat, std::vector<HighsInt>& basis_idxs){
    this->B_.setup(constraint_mat, basis_idxs);
};

void ReducedHessian::Hbtran(std::vector<double>& vec){
    this->B_.btranCall(vec);
};

void ReducedHessian::Hftran(std::vector<double>& vec){
    this->B_.ftranCall(vec);
};

void ReducedHessian::init(){
    this->perm_.assign(this->nullsp_dim_, 0);
    std::iota(this->perm_.begin(), this->perm_.end(), 0); // from claude.ai
    // dimensions are taken from the last index of the matrix, i.e. the bottom diagonal element
    this->chol_.assign(chol_idx(this->nullsp_dim_-1, this->nullsp_dim_-1) + 1, 0.);
};

void ReducedHessian::Hbuild(){
    this->B_.build();
};

void ReducedHessian::build(){
    // extracts explicit Z^T and then computes explicitly the reduced Hessian
    // then it factorises it
    // useful when starting point of ASM provides a non-empty null-space
    // computes the factorization row by row according to Cholesky—Banachiewicz
    // TODO what about pivoting
    // while Q is dense, we can't guarantee Z is too, so M is treated as dense, and likewise its factors
    // if we order M by the largest of its diagonal entries we need to first compute it all
    // BUILD RED HESSIAN FIRST
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){// get Z^T
        std::vector<double> z_col(this->Q_.dim_);
        z_col[this->Q_.dim_ - this->nullsp_dim_ + i] = 1.;
        Hbtran(z_col); // solves returning a column of Z, which we store as a row of Z^T
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
            chol(i,j) = sum; // should be ordered such that chol_ is row-wise of M
        }
    }
    // from claude.ai, get order of permutations based on the diagonal element's magnitude
    std::sort(this->perm_.begin(), this->perm_.end(), [&](int i, int j) {
       return this->chol_[chol_idx(i,i)] > this->chol_[chol_idx(j,j)];
    }); // from here on i can use chol(), which uses perm
    // now get permuted elements as this->chol_[ idx(this->perm[i], this->perm[j]) ]
    // FACTORIZE in place
    for (HighsInt i {0}; i<this->nullsp_dim_; i++){
        for (HighsInt j {0}; j <= i; j++){
            if (i == j){ // diagonal element
                for (HighsInt k {0}; k < j; k++){
                    double row_el = chol(i,k);
                    chol(i,i) -= row_el * row_el;
                }
                chol(i,i) = std::sqrt(chol(i,i));
            }
            else { // off diagonal element
                for (HighsInt k {0}; k < j; k++){
                    chol(i,j) -= chol(i,k) * chol(j,k);
                }
                chol(i,j) /= chol(j,j);
            }
        }
    }
}

void ReducedHessian::solve(std::vector<double>& vec){
    std::vector<double> sol(vec.size()); // could do the computations without, but need it anyways for final permutation
    // 1. solve Ly = P^T b with forward substitution
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){
        sol[i] += vec[ this->perm_[i] ];
        for (HighsInt j {0}; j < i; j++){
            sol[i] -= chol(i,j) * sol[j]; 
        }
        sol[i] /= chol(i,i);
    }
    // 2. solve L^T z = y with backward substitution
    for (HighsInt i {this->nullsp_dim_ - 1}; i > -1; i--){
        for (HighsInt j {i}; j < this->nullsp_dim_; j++){// perform operation in place
            sol[i] -= chol(i,j) * sol[j];
        }
        sol[i] /= chol(i,i);
    }
    // 3. compute P^T x = z (rearrange z into x)
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){
        vec[ this->perm_[i] ] = sol[i];
    }
};

void ReducedHessian::extend(){
    // extend(std::vector<double>& y){
};
