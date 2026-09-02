/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "qpsolver/g_solver.hpp"

HighsInt AsmSolver::locL(const HighsInt& i, const HighsInt& j) {
    if ( j > i ) throw std::domain_error("Column index should not be larger than row index!");
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

void AsmSolver::extend(const HighsInt& loc_deactivated, const HighsInt& idx_deactivated){
    // get new nullspace column, creating unit HVector
    this->ZT_.emplace_back(); // create new z_col (first get unit HVector)
    this->ZT_.back().setup(this->Q_.dim_);
    this->ZT_.back().index[0] = loc_deactivated;
    this->ZT_.back().array[ this->ZT_.back().index[0] ] = 1.;
    this->ZT_.back().count = 1;
    this->ZT_.back().packFlag = true;
    this->B_.btranCall(this->ZT_.back(), 1.); // compute z_col inplace
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
    if ( idx_deactivated < this->lp_.num_row_ && false){
        // after adding a vector to Z, for numerical reasons we update the L and the factorisation of B
        // by changing the newly freed vector (that now pads A in B) with a unit vector
        HighsInt hint { 99999 }; // same number as Micheal in Basis::updatebasis
        HighsInt iRow = loc_deactivated; // because function argument loc_deactivated is constant
        HVector& yp = this->ZT_.back();
        HVector newcol;
        // find largest element modulus in yp
        double max_abs {0.};
        HighsInt max_idx {-1};
        for (HighsInt i {0}; i < yp.count; i++){ // yp is a sparse vector
            if ( std::abs( yp.array[yp.index[i]] ) > std::abs( max_abs ) ) {
                max_abs = yp.array[yp.index[i]];
                max_idx = yp.index[i];
            }
        }
        // build ep by pointing to the vector that is to be replaced with the unit column
        this->buffer_.assign(this->Q_.dim_,0.);
        this->buffer_[max_idx] = 1.;
        stdvec2hvec(this->buffer_, newcol);
        this->B_.ftranCall(newcol, 1.);
        // update basis matrix
        this->B_.update(&newcol, &yp, &iRow, &hint);
        // then update reduced hessian factor
        // first reorder elements of newcol (vector d in Fletcher) with the permutation in which vectors in Z sit
        for (HighsInt i { this->rangsp_dim_ - 1 }; i < this->Q_.dim_ - 1; i++){ // elements i < this->rangsp_dim_ - 1 in buffer_ are rubbish
            this->buffer_[ i ] = - newcol.array[ this->basis_perm_[i] ] / max_abs;
        }
        this->buffer_.back() = max_abs;
        // now elements d_[p+1 to n] in (24) of 10.1007/s101070050113 are the last n - (p+1) elements of buffer
        if ( this->nullsp_dim_ > 0 ){ // if nullspace wasn't empty before deactivation rotations have a reason to be used
            // apply givens rotation from the left to zero out all but the rightmost element in the last row of the enhanced L
            // use their memory space to store the spike column that appears in the rightmost column of L
            for (HighsInt j {this->nullsp_dim_}; j > -1; j--) leftGivens(j);
            // multiply spike column with eta colum
            HighsInt dim = this->nullsp_dim_ + 1;
            for (HighsInt i {0}; i < this->nullsp_dim_; i++){
                this->chol_[ locL(dim, i) ] += this->chol_[ locL(dim, dim) ] * this->buffer_[ this->rangsp_dim_ + i ];
            }
            this->chol_[ locL(dim, dim) ] *= this->buffer_.back(); // last element in the spike is only scaled
            // remove right spike
            for (HighsInt i {0}; i < this->nullsp_dim_; i++) rightGivensSpike(i);
        } else { // otherwise chol_ is a singleton that only needs scaling
            this->chol_.back() /= this->buffer_.back();
        }
    }
    return;
}

void AsmSolver::rightGivensSpike(const HighsInt& i){
    // Givens rotation on L from the right that affect columns
    // intended to remove the spike on the last, right-most, column
    // argument i referes to the row whose last-column element is to be zeroed out
    // the spike is stored in the last row of L, so change is made in place
    HighsInt j = this->nullsp_dim_ + 1; // at this point the size of the nullspace hasnt been updated yet, but L is enhanced already
    double cos {0.};
    double sin {1.};
    double a = this->chol_[ locL(i, i) ]; // element i in the row whose last element has to be zeroed out
    if ( std::abs(a) >= this->options_.factor_pivot_tolerance){
        double b = this->chol_[ locL(i, j) ]; // element to zero out
        double hyp = std::sqrt( a*a + b*b ); // guaranteed to be > 0
        cos = a / hyp;
        sin = b / hyp;
    }
    double temp_ki;
    double temp_kj;
    // change each row element in the two columns affected (i and last one) on and below row i up to second to last row
    for (HighsInt k {this->nullsp_dim_}; k > i; k--){
        temp_ki = this->chol_[ locL(k, i) ];
        temp_kj = this->chol_[ locL(k, j) ];
        // modify element in column i
        this->chol_[ locL(k, i) ] = cos * temp_ki + sin * temp_kj;
        // modify element in last column, while zeroing out spike element in the last column
        this->chol_[ locL(k, j) ] = - sin * temp_ki + cos * temp_kj;
    }
    // diagonal element in row whose last element is being zeroed out
    // note element (j,i) is in fact (i,j) of the spike column, but stored in the last row of L
    this->chol_[ locL(i, i) ] = cos * this->chol_[ locL(i, i) ] + sin * this->chol_[ locL(j, i) ];
    // element that was zero
    this->chol_[ locL(j, i) ] = sin * this->chol_[ locL(j, j) ];
    // bottom right element
    this->chol_[ locL(j, j) ] *= cos;
}

void AsmSolver::leftGivens(const HighsInt& j){
    // Givens rotations on L from the left that affect rows
    // intended for spike creation and reduction
    // argument j refers to column of L whose last-row element is to be zeroed out,
    // to create a non-zero entry in row j of the last column of L
    HighsInt i = this->nullsp_dim_ + 1; // at this point the size of the nullspace hasnt been updated yet, but L is enhanced already
    double cos {0.};
    double sin {1.};
    double a = this->chol_[ locL(j, j) ]; // diagonal element in the column whose bottom element has to be zeroed out
    if ( std::abs(a) >= this->options_.factor_pivot_tolerance){
        double b = this->chol_[ locL(i, j) ]; // element to zero out
        double hyp = std::sqrt( a*a + b*b ); // guaranteed to be > 0
        cos = a / hyp;
        sin = b / hyp;
    }
    double temp_jk;
    double temp_ik;
    // change elements in row j and then i (the last one), so loop through all columns k until diagonal (j,j)
    for (HighsInt k {0}; k < j; k++){
        temp_jk = this->chol_[ locL(j, k) ];
        temp_ik = this->chol_[ locL(i, k) ];
        // update elements in row j, (sub)diagonal
        this->chol_[ locL(j, k) ] = cos * temp_jk + sin * temp_ik;
        // update elements in row i (last row), it will be zeroed out later
        this->chol_[ locL(i, k) ] = - sin * temp_jk + cos * temp_ik;
    }
    // elements in column j: the element diagonal element is updated
    this->chol_[ locL(j, j) ] = cos * this->chol_[ locL(j, j) ] + sin * this->chol_[ locL(i, j) ];
    // the bottom element which is zeroed out is replaced with the new spike element (j,i)
    this->chol_[ locL(i, j) ] =  sin * this->chol_[ locL(i, i) ];
    // the bottom right element also needs updating
    this->chol_[ locL(i, i) ] *= cos;
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
    for (HighsInt i {loc_activated + 1}; i < this->nullsp_dim_; i++) rightGivensHess(i);
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

void AsmSolver::rightGivensHess(const HighsInt& i){
    // Givens rotations on L from the right that affect columns
    // intended for removing a row/column from L when an arbitrary vector is removed
    // argument i refers to element (i,i) in L that is to be zeroed out
    double cos {0.};
    double sin {1.};
    double a = this->chol_[ locL(i, i - 1) ];
    if ( std::abs(a) >= this->options_.factor_pivot_tolerance){
        double b = this->chol_[ locL(i, i) ]; // element to zero out
        double hyp = std::sqrt( a*a + b*b ); // guaranteed to be > 0
        cos = a / hyp;
        sin = b / hyp;
    }
    // change elements in column i and then i+1, so loop through rows beneath the diagonal
    for (HighsInt j {i}; j < this->nullsp_dim_; j++){
        double temp = cos * this->chol_[ locL(j,i-1) ] + sin * this->chol_[ locL(j, i) ];
        // TODO we are accessing a row-wise matrix by column, could be better
        //  when i == j the element (i, i+1) is zeroed out and then deleted so no need to change it
        if (j != i) this->chol_[ locL(j,i) ] = - sin * this->chol_[ locL(j,i-1) ] + cos * this->chol_[ locL(j, i) ];
        this->chol_[ locL(j,i-1) ] = temp; // in-place operation requires overwriting element when all computations are done
    }
}