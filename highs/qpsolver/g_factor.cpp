/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "qpsolver/g_solver.hpp"

HighsInt AsmSolver::locL(const HighsInt& i, const HighsInt& j) {
    return i*(i+1)/2 + j; // assumes indices are given for lower triangular matrix
}

void AsmSolver::recomputeExplicit(){ // TODO pivoting?
    this->ZT_.assign(this->nullsp_dim_, HVector()); // TODO is it necessary to also recompute ZT_?
    HighsInt chol_size = this->nullsp_dim_ * (this->nullsp_dim_ + 1) / 2;
    this->chol_.assign(chol_size, 0.);
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){ // get Z^T
        // create unit HVector
        this->ZT_[i].setup(this->Q_.dim_);
        this->ZT_[i].index[0] = this->basis_perm_[this->rangsp_dim_ + i];
        this->ZT_[i].array[ this->ZT_[i].index[0] ] = 1.;
        this->ZT_[i].count = 1;
        this->ZT_[i].packFlag = true;
        // extract column of Z
        this->B_.btranCall(this->ZT_[i], 1.); // solves returning a column of Z, which we store as a row of Z^T
    }
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){// loop over the rows of Z^T
        this->Q_.product(this->ZT_[i].array, this->buffer_); // compute row of Z^T Q
        for (HighsInt j {0}; j <= i; j++){ // loop through columns of Z, up to the current row of Z^T, to only compute lower triangle of red_hessian_
            // TODO could this be done with HFtran to compute a full column of the reduced Hessian, without the need to store Z^T?
            // in which case, do not use this->buffer_ anymore
            double sum {0.};
            for (HighsInt k {0}; k < this->Q_.dim_; k++){ // inner produce of row of Z^T Q with column of Z
                sum += this->buffer_[k] * this->ZT_[j].array[k]; // factorization row by row according to Cholesky—Banachiewicz
            }
            this->chol_[ locL(i,j) ] = sum; // should be ordered such that chol_ is row-wise of M
        }
    }
    return;
}

void AsmSolver::refactorize(){
    // perform cholesky factorization in place
    for (HighsInt i {0}; i<this->nullsp_dim_; i++){
        for (HighsInt j {0}; j <= i; j++){
            if (i == j){ // diagonal element
                for (HighsInt k {0}; k < j; k++){
                    double row_el = this->chol_[ locL(i,k)];
                    this->chol_[ locL(i,i) ] -= row_el * row_el;
                }
                this->chol_[ locL(i,i) ] = std::sqrt(this->chol_[ locL(i,i) ]);
            }
            else { // off diagonal element
                for (HighsInt k {0}; k < j; k++){
                    this->chol_[ locL(i,j) ] -= this->chol_[ locL(i,k) ] * this->chol_[ locL(j,k) ];
                }
                this->chol_[ locL(i,j) ] /= this->chol_[ locL(j,j) ];
            }
        }
    }
    return;
}

void AsmSolver::Lsolve(std::vector<double>& vec){
    if ( (HighsInt)vec.size() != this->nullsp_dim_) throw std::logic_error("Fw solve requires a vector the size of the nullspace!");
    // solve Ly = b with forward substitution
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){
        for (HighsInt j {0}; j < i; j++){
            vec[i] -= this->chol_[ locL(i,j) ] * vec[j]; 
        }
        vec[i] /= this->chol_[ locL(i,i) ];
    }
    return;
}

void AsmSolver::LTsolve(std::vector<double>& vec){
    if ( (HighsInt)vec.size() != this->nullsp_dim_) throw std::logic_error("Bw solve requires a vector the size of the nullspace!");
    // solve L^T z = y with backward substitution
    HighsInt limit = this->nullsp_dim_ - 1;
    for (HighsInt i {limit}; i > -1; i--){
        for (HighsInt j {limit}; j > i; j--){// perform operation in place
            vec[i] -= this->chol_[ locL(j,i) ] * vec[j]; // note indices are swapped since we are accessing the upper triangular image of L
        }
        vec[i] /= this->chol_[ locL(i,i) ];
    }
    return;
}

void AsmSolver::LLTsolve(std::vector<double>& vec){
    Lsolve(vec);
    LTsolve(vec);
    return;
}

void AsmSolver::extend(const HighsInt& loc_deactivated){
    // TODO new column in V may not be the deactivated column, even though the deactivated index stays the same?
    // TODO if a vector is changed in basis matrix, weights may change too, update pricing and red grad accordingly
    // get new nullspace column
    // create unit HVector
    this->ZT_.emplace_back(); // create new z_col (first get unit HVector)
    this->ZT_.back().setup(this->Q_.dim_);
    this->ZT_.back().index[0] = loc_deactivated;
    this->ZT_.back().array[ this->ZT_.back().index[0] ] = 1.;
    this->ZT_.back().count = 1;
    this->ZT_.back().packFlag = true;
    this->B_.btranCall(this->ZT_.back(), 1.); // compute z_col inplace
    // TODO extend by paying attention to numerical instabilities
    double lambda {0.}; // new diagonal element for cholesky factor
    if (this->nullsp_dim_ > 0){ // nullspace dimension updated after calling extend()
        // solve L l = Z^T ( Q z_col ) = Z^T sol
        std::vector<double> sol(this->Q_.dim_);
        this->Q_.product(this->ZT_.back().array, sol);
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
    lambda += 2 * computeQuadObjective(this->ZT_.back().array);
    if (lambda <= this->options_.factor_pivot_tolerance) throw std::domain_error("Reduced matrix is either semi- or indefinite!");
    this->chol_.push_back( std::sqrt(lambda) );
    return;
}

void AsmSolver::reduce(const HighsInt& loc_activated){
    if ( loc_activated == this->nullsp_dim_){
        this->chol_.resize(this->chol_.size() - this->nullsp_dim_); // drop last row of L
        this->ZT_.pop_back(); // drop last row of Z^T (last column of Z)
        this->removeNullSpaceDim();
        return;
    }
    // if activated vector was free in basis, we must remove that row and column from Z^T, no arbitrary choice
    // TODO givens rotations are also needed when dealing with indefinite matrix
    // we remove row loc_activated, so we need to zero out the super-diagonal elements
    // from row loc_activated+1 till the end
    for (HighsInt i {loc_activated + 1}; i < this->nullsp_dim_; i++){
        double a = this->chol_[ locL(i, i - 1) ];
        double b = this->chol_[ locL(i, i) ]; // element to zero out
        double cos;
        double sin;
        if ( std::abs(a) < this->options_.factor_pivot_tolerance){
            cos = 0.;
            sin = 1.;
        } else {
            double hyp = std::sqrt( a*a + b*b ); // guaranteed to be > 0
            cos = a / hyp;
            sin = b / hyp;
        }
        // change elements in column i and then i+1
        for (HighsInt j {i}; j < this->nullsp_dim_; j++){ // loop through rows beneath diagonal
            double temp = cos * this->chol_[ locL(j,i-1) ] + sin * this->chol_[ locL(j, i) ];
            // TODO we are accessing a row-wise matrix by column, could be better
            //  when i == j the element (i, i+1) is zeroed out and then deleted so no need to change it
            if (j != i) this->chol_[ locL(j,i) ] = - sin * this->chol_[ locL(j,i-1) ] + cos * this->chol_[ locL(j, i) ];
            this->chol_[ locL(j,i-1) ] = temp; // in-place operation requires overwriting element when all computations are done
        }
    }
    // then we need to erase all of the zeroes, first erase all super-diagonal elements that were zeroed out
    for (HighsInt i {this->nullsp_dim_ - 1}; i > loc_activated; i--){
        this->chol_.erase( this->chol_.begin() + locL(i,i) );
    }
    // then delete exactly loc_activated elements which is the number of elements in row loc_activated
    HighsInt size_L1 = (loc_activated - 1) * loc_activated / 2; // size of lower diagonal matrix above removed row
    this->chol_.erase(this->chol_.begin() + size_L1, this->chol_.begin() + size_L1 + loc_activated + 1);
    //
    this->ZT_.erase(this->ZT_.begin() + loc_activated);
    removeNullSpaceDim();
    return;
}