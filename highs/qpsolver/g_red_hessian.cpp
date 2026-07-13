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

void ReducedHessian::Hsetup(HighsSparseMatrix& constraint_mat, std::vector<HighsInt>& basis_idxs){
    this->B_.setup(constraint_mat, basis_idxs);
};

void ReducedHessian::Hbtran(std::vector<double>& vec){
    this->B_.btranCall(vec);
};

void ReducedHessian::Hftran(std::vector<double>& vec){
    this->B_.ftranCall(vec);
};

void ReducedHessian::init(HighsInt nullsp_dim){
    this->perm_.assign(this->nullsp_dim_, 0);
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
    for (size_t i {0}; i < this->nullsp_dim_; i++){// get Z^T
        std::vector<double> z_col(this->Q_.dim_);
        z_col[i] = 1.;
        Hbtran(z_col);
        this->ZT_.push_back(z_col);
    }
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){// loop over the rows of Z^T
        std::vector<double> row(this->Q_.dim_); // row of Z^T Q
        this->Q_.product(this->ZT_[i], row); // compute it
        double sum {0.};
        for (size_t j {0}; j <= i; j++){ // loop through columns of Z, up to the current row of Z^T, to only compute lower triangle of red_hessian_
            for (HighsInt k {0}; k < this->Q_.dim_; k++){ // inner produce of row of Z^T Q with column of Z
                sum += row[k] * this->ZT_[j][k];
            }
            this->chol_.push_back(sum); // should be ordered such that chol_ is row-wise of M
        }
    }
    // from claude.ai, get order of permutations based on the diagonal element's magnitude
    std::iota(this->perm_.begin(), this->perm_.end(), 0);
    std::sort(this->perm_.begin(), this->perm_.end(), [&](int i, int j) {
       return this->chol_[chol_idx(i,i)] > this->chol_[chol_idx(j,j)];
    }); // from here on i can use chol(), which uses perm
    // now get permuted elements as this->chol_[ idx(this->perm[i], this->perm[j]) ]
    // FACTORIZE in place
    for (HighsInt i {0}; i<this->nullsp_dim_; i++){
        for (HighsInt j {0}; j <= i; j++){
            if (i == j){
                // diagonal element (for brevity of code i create temp variable sum)
                for (HighsInt k {0}; k < i; k++){
                    double square = chol(i,k);
                    chol(i,i) -= square * square;
                }
                chol(i,i) = std::sqrt(chol(i,i));
            }
            else {
                // off diagonal element
                for (HighsInt k {0}; k < i; k++){
                    chol(i,j) -= chol(i,k) * chol(j,k);
                }
                chol(i,j) /= chol(j,j);
            }
        }
    }
}

void ReducedHessian::extend(){
    // extend(std::vector<double>& y){
};
