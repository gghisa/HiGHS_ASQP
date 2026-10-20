/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "qpsolver/g_solver.hpp"

void AsmSolver::stdvec2hvec(const std::vector<double>& vec, HVector& hvec){
    hvec.setup(vec.size());
    for (size_t i {0}; i < vec.size(); i++){
        if (vec[i] != 0){
            hvec.index[hvec.count] = i;
            hvec.count += 1;
        }
    }
    hvec.array = vec;
    hvec.packFlag = true;
    return;
}

void AsmSolver::recomputeRedHessian(){
    highsLogDev(this->options_.log_options, HighsLogType::kInfo,
        "Iter %d: (Re)computing reduced Hessian" , this->info_.qp_iteration_count);
    //
    HVector ZTQi;
    HighsSparseMatrix ZTQ;
    ZTQ.format_ = MatrixFormat::kColwise;
    ZTQ.num_col_ = this->Q_.dim_;
    ZTQ.num_row_ = this->nullsp_dim_;
    HighsInt chol_size = this->nullsp_dim_ * (this->nullsp_dim_ + 1) / 2;
    this->chol_.assign(chol_size, 0.);
    for (HighsInt i {0}; i < this->Q_.dim_; i++){ // get Q Z column vectors
        ZTQi.setup(this->Q_.dim_);
        // create unit HVector with entries from relevant Hessian column
        ZTQi.count = this->Q_.start_[i + 1] - this->Q_.start_[i];
        for ( HighsInt j {0}; j < ZTQi.count; j++ ){
            ZTQi.index[j] = this->Q_.index_[ this->Q_.start_[i] + j ];
            ZTQi.array[ ZTQi.index[j] ] = this->Q_.value_[ this->Q_.start_[i] + j ];
        }
        ZTQi.packFlag = true;
        // extract column of Z
        this->B_.ftranCall(ZTQi, 1.); // on the last this->nullsp_dim_ elements here are meaningful for us Z^T Q e_i
        // append new column to sparse matrix
        double value;
        HighsInt count = 0;
        for ( HighsInt j {this->rangsp_dim_}; j < this->Q_.dim_; j++){
            value = ZTQi.array[ this->basis_perm_[j] ];
            if ( std::abs(value) > this->options_.factor_pivot_tolerance ){
                ZTQ.index_.push_back(j - this->rangsp_dim_);
                ZTQ.value_.push_back(value);
                count += 1;
            }
        }
        ZTQ.start_.push_back(ZTQ.start_.back() + count);
    }
    ZTQ.ensureRowwise(); // hold Z^T Q rowise, hence Q Z columnwise
    ZTQ.format_ = MatrixFormat::kColwise; // book keeping
    ZTQ.num_col_ = ZTQ.num_row_;
    ZTQ.num_row_ = this->Q_.dim_;
    double sum;
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){ // loop over the columns of Q Z
        ZTQi.setup(this->Q_.dim_);
        // create unit HVector with entries from relevant Q Z column
        ZTQi.count = ZTQ.start_[i + 1] - ZTQ.start_[i];
        for ( HighsInt j {0}; j < ZTQi.count; j++ ){
            ZTQi.index[j] = ZTQ.index_[ ZTQ.start_[i] + j ];
            ZTQi.array[ ZTQi.index[j] ] = ZTQ.value_[ ZTQ.start_[i] + j ];
        }
        ZTQi.packFlag = true;
        this->B_.ftranCall(ZTQi, 1.); // compute column i of reduced hessian (last nullsp elements in resulting vec)
        for (HighsInt j {i}; j < this->nullsp_dim_; j++){ // loop through rows of Z^T to only compute lower triangle of red_hessian_
            this->chol_[ locL(j,i) ] = ZTQi.array[ this->basis_perm_[this->rangsp_dim_ + j] ]; // should be ordered such that chol_ is row-wise of M
        }
    }
    // perform cholesky factorization in place, row by row according to Cholesky—Banachiewicz
    for (HighsInt i {0}; i<this->nullsp_dim_; i++){
        for (HighsInt j {0}; j <= i; j++){
            if (i == j){ // diagonal element
                sum = 0;
                for (HighsInt k {0}; k < j; k++){
                    double row_el = this->chol_[ locL(i,k)];
                    sum += row_el * row_el;
                }
                this->chol_[ locL(i,i) ] -= sum;
                this->chol_[ locL(i,i) ] = std::sqrt(this->chol_[ locL(i,i) ] + 1e-8); // TODO remove regularisation
            }
            else { // off diagonal element
                sum = 0;
                for (HighsInt k {0}; k < j; k++){
                    sum += this->chol_[ locL(i,k) ] * this->chol_[ locL(j,k) ];
                }
                this->chol_[ locL(i,j) ] -= sum;
                this->chol_[ locL(i,j) ] /= this->chol_[ locL(j,j) ];
            }
        }
    }
    return;
}

void AsmSolver::Lsolve(std::vector<double>& vec){
    // solve Ly = b with forward substitution
    double sum;
    for (HighsInt i {0}; i < this->nullsp_dim_; i++){
        sum = 0;
        for (HighsInt j {0}; j < i; j++){
            sum+= this->chol_[ locL(i,j) ] * vec[j + this->rangsp_dim_]; 
        }
        vec[i + this->rangsp_dim_] -= sum;
        vec[i + this->rangsp_dim_] /= this->chol_[ locL(i,i) ];
    }
    return;
}

void AsmSolver::LTsolve(std::vector<double>& vec){
    // solve L^T z = y with backward substitution
    HighsInt limit = this->nullsp_dim_ - 1;
    double sum;
    for (HighsInt i {limit}; i > -1; i--){
        sum = 0; // for numerical reasons, evaluate first looped sum, then add to potentially bigger number
        for (HighsInt j {limit}; j > i; j--) // note indices are swapped since we are accessing the upper triangular image of L
            sum += this->chol_[ locL(j,i) ] * vec[j + this->rangsp_dim_];
        vec[i + this->rangsp_dim_] -= sum;
        vec[i +  this->rangsp_dim_] /= this->chol_[ locL(i,i) ];
    }
    return;
}

HighsInt AsmSolver::replaceWithUnitVec(HighsInt iloc, HVector& newcol, double& max_abs){
    HighsInt max_idx {-1};
    for (HighsInt i {0}; i < this->Ztemp_.count; i++){ // Ztemp is a sparse vector
        if ( std::abs( this->Ztemp_.array[this->Ztemp_.index[i]] ) > std::abs( max_abs ) ) {
            max_abs = this->Ztemp_.array[this->Ztemp_.index[i]];
            max_idx = this->Ztemp_.index[i];
        }
    }
    // build ep as the unit column replacing the deactivated constraint
    this->buildConstraint(max_idx + this->lp_.num_row_, newcol); // extra FLOP necessary?
    // update basis matrix
    this->B_.update(&newcol, &this->Ztemp_, &iloc, &this->Bhint_);
    this->num_basis_updates_++;
    return max_idx + this->lp_.num_row_; // return which unit vector has been added to basis
}

void AsmSolver::buildZtemp(const HighsInt& iloc){ // TODO make Ztemp an object of the class
    this->buffer_.assign(this->Q_.dim_, 0.);
    this->buffer_[iloc] = 1.;
    this->stdvec2hvec(this->buffer_, this->Ztemp_);// create new z_col (first get unit HVector)
    this->B_.btranCall(this->Ztemp_, 1.); // compute z_col inplace
}

void AsmSolver::extend(){
    assert( this->relaxed_iloc_ > -1 );
    HighsInt idx_deactivated = this->basis_idxs_[ this->relaxed_iloc_ ];
    HighsInt loc_deactivated = this->basis_perm_[ this->relaxed_iloc_ ];
    // get new nullspace column, creating unit HVector
    if (this->nullsp_dim_ > 0){ // nullspace dimension updated after calling extend()
        // L l = Z^T ( Q z_col ) = Z^T sol from buffer
        this->chol_.insert(this->chol_.end(), this->Rtemp_.begin() + this->rangsp_dim_, this->Rtemp_.end());
        double sum {0.}; // for numerical reasons, evaluate first looped sum, then add to potentially bigger number
        for (HighsInt i {this->rangsp_dim_}; i < this->Q_.dim_; i++) sum += this->Rtemp_[i] * this->Rtemp_[i];
        this->yQytemp_ -= sum;
    }
    double tol = 10 * this->options_.factor_pivot_tolerance;
    if (this->yQytemp_ < - tol ){
        std::cout<<"Reduced matrix is either semi- or indefinite!\n"<<std::flush;
        throw std::domain_error("Reduced matrix is either semi- or indefinite!");
    } else if ( this->yQytemp_ < tol ){
        this->yQytemp_ += this->options_.primal_feasibility_tolerance; // TODO implement noncovex code
    }
    this->chol_.push_back( std::sqrt(this->yQytemp_) );
    // then change padding vector to unit vector if it wasn't a unit vector
    if ( idx_deactivated < this->lp_.num_row_ ){
        // after adding a vector to Z, for numerical reasons we update the L and the factorisation of B
        // by changing the newly freed vector (that now pads A in B) with a unit vector
        this->changeStatus(idx_deactivated, AsmBasisStatus::kInactive);
        HVector newcol;
        double max_abs {0.};
        HighsInt q = this->replaceWithUnitVec(loc_deactivated, newcol, max_abs);
        assert( this->getAsmBasisStatus(q) == AsmBasisStatus::kInactive );
        this->basis_idxs_[ this->relaxed_iloc_ ] = q;
        // then update reduced hessian factor
        if ( this->nullsp_dim_ > 0 ) // if nullspace wasn't empty before deactivation rotations have a reason to be used
            this->refresh(max_abs, newcol);
        else this->chol_.back() /= max_abs; // otherwise chol_ is a singleton that only needs scaling
    }
    // update status
    this->changeStatus(this->basis_idxs_[this->relaxed_iloc_], AsmBasisStatus::kFreeInBasis); // NB use location to account for both cases above
    this->fromActiveToPadding(this->relaxed_iloc_);
    this->addNullSpaceDim();
    return;
}

void AsmSolver::refresh(const double& max_abs, HVector& newcol){
    // update reduced hessian after a padding vector has been changed with newcol
    HighsInt dim = this->nullsp_dim_;
    // first reorder elements of newcol (vector d in Fletcher) with the permutation in which vectors in Z sit
    for (HighsInt i { this->rangsp_dim_ }; i < this->Q_.dim_; i++) // elements i < this->rangsp_dim_ - 1 in buffer_ are rubbish
        this->buffer_[ i ] = - newcol.array[ this->basis_perm_[i] ] / max_abs;
    // now elements d_[p+1 to n] in (24) of 10.1007/s101070050113 are the last elements of buffer
    // apply givens rotation from the left to zero out all but the rightmost element in the last row of the enhanced L
    // use their memory space to store the spike column that appears in the rightmost column of L
    this->addSpike(this->nullsp_dim_ - 1, this->nullsp_dim_); // add spike from the second to last row
    // multiply spike column with eta colum
    for (HighsInt i {0}; i < dim; i++)
        this->chol_[ locL(dim, i) ] += this->chol_[ locL(dim, dim) ] * this->buffer_[ this->rangsp_dim_ + i ];
    this->chol_.back() /= max_abs; // last element in the spike is only scaled
    // remove right spike
    this->removeSpike(this->nullsp_dim_);
}

void AsmSolver::addSpike(const HighsInt& start, const HighsInt& idx_last_col){
    // Givens rotation on L from the right
    // intended to add the spike on the last, right-most, column
    // subdiagonal sine is negative
    HighsInt j = idx_last_col; // at this point the size of the nullspace hasnt been updated yet, but L is enhanced already
    double sin, cos, hyp;
    double temp_ki, temp_kj;
    for (HighsInt i {start}; i > -1; i--){
        // argument i refers to the column whose last-row element is to be zeroed out
        // the spike is stored in the last row of L, so change is made in place
        cos = this->chol_[ locL(j, j) ]; // bottom right element
        sin = this->chol_[ locL(j, i) ]; // element of the last row to be zeroed out
        hyp = std::sqrt( cos*cos + sin*sin ); // guaranteed to be > 0
        cos /= hyp;
        sin /= hyp;
        // update bottom right element first ( locL(j,j) )
        this->chol_.back() = sin * this->chol_[ locL(j, i) ] + cos * this->chol_[ locL(j, j) ];
        // note element (j,i) is in fact (i,j) of the spike column, but stored in the last row of L
        this->chol_[ locL(j, i) ] = sin * this->chol_[ locL(i, i) ]; // create spike element where element is implicitly zeroed out
        // change each row element in the two columns affected (i and last one) on and below row i up to second to last row
        for (HighsInt k {j - 1}; k > i; k--){
            temp_ki = this->chol_[ locL(k, i) ];
            temp_kj = this->chol_[ locL(j, k) ]; // element (k,j) is stored in (j,k)
            // modify element in column i
            this->chol_[ locL(k, i) ] = cos * temp_ki - sin * temp_kj;
            // modify element in last column, while zeroing out spike element in the last column
            this->chol_[ locL(j, k) ] = sin * temp_ki + cos * temp_kj; // element (k, j) stored in (j, k)
        }
        // diagonal element in row where spike element appears
        this->chol_[ locL(i, i) ] *= cos;
    }
}

void AsmSolver::removeSpike(const HighsInt& idx_last_col){
    // Givens rotation on L from the right
    // intended to remove the spike on the last, right-most, column
    // subdiagonal sine is positive
    HighsInt j = idx_last_col; // at this point the size of the nullspace hasnt been updated yet, but L is enhanced already
    double cos, sin, hyp;
    double temp_ki, temp_kj;
    for (HighsInt i {0}; i < j; i++){
        // argument i referes to the row whose last-column element is to be zeroed out
        // the spike is stored in the last row of L, so change is made in place
        cos = this->chol_[ locL(i, i) ]; // element i in the row whose last element has to be zeroed out
        sin = this->chol_[ locL(j, i) ]; // element to zero out (in memory in the last row)
        hyp = std::sqrt( cos*cos + sin*sin ); // guaranteed to be > 0
        cos /= hyp;
        sin /= hyp;
        // diagonal element in row whose last element is being zeroed out
        // note element (j,i) is in fact (i,j) of the spike column, but stored in the last row of L
        this->chol_[ locL(i, i) ] = cos * this->chol_[ locL(i, i) ] + sin * this->chol_[ locL(j, i) ];
        // change each row element in the two columns affected (i and last one) on and below row i up to second to last row
        for (HighsInt k {i + 1}; k < j; k++){
            temp_ki = this->chol_[ locL(k, i) ];
            temp_kj = this->chol_[ locL(j, k) ]; // element (k,j) is stored in (j,k)
            // modify element in column i
            this->chol_[ locL(k, i) ] = cos * temp_ki + sin * temp_kj;
            // modify element in last column, while zeroing out spike element in the last column
            this->chol_[ locL(j, k) ] = - sin * temp_ki + cos * temp_kj; // element (k, j) stored in (j, k)
        }
        // element that was zero
        this->chol_[ locL(j, i) ] = sin * this->chol_.back();
        // bottom right element
        this->chol_.back() *= cos; // locL(j,j)
    }
    this->chol_.back() = std::abs( this->chol_.back() ); // TODO hack
}

void AsmSolver::reduceInBasis(const HighsInt& loc_activated){ // only called when element moved to A was an element of the padding V
    if ( loc_activated == this->nullsp_dim_ - 1 ){ // if index of activated corresponds to index of last row in L
        this->chol_.resize(this->chol_.size() - this->nullsp_dim_); // drop last row of L
        return;
    }
    // TODO givens rotations are also needed when dealing with indefinite matrix
    // we remove row loc_activated (a posteriori, below, for indexing reasons)
    // so we need to zero out the super-diagonal elements
    // from row loc_activated+1 till the end
    this->rightGivensHess(loc_activated + 1);
    // then we need to erase all of the zeroes, first erase all super-diagonal elements that were zeroed out
    for (HighsInt i {this->nullsp_dim_ - 1}; i > loc_activated; i--){
        this->chol_.erase( this->chol_.begin() + locL(i,i) );
    }
    // then delete exactly loc_activated elements which is the number of elements in row loc_activated
    HighsInt size_L1 = ((loc_activated + 1) * loc_activated) / 2; // size of lower diagonal matrix above removed row
    this->chol_.erase(this->chol_.begin() + size_L1, this->chol_.begin() + size_L1 + loc_activated + 1);
    return;
}

void AsmSolver::rightGivensHess(const HighsInt& start){
    // Givens rotations on L from the right that affect columns
    // intended for removing a row/column from L when an arbitrary vector is removed
    double cos, sin, hyp;
    double temp;
    for (HighsInt i {start}; i < this->nullsp_dim_; i++){
        // argument i refers to element (i,i) in L that is to be zeroed out
        cos = this->chol_[ locL(i, i - 1) ];
        sin = this->chol_[ locL(i, i) ]; // element to zero out
        hyp = std::sqrt( cos*cos + sin*sin ); // guaranteed to be > 0
        cos /= hyp;
        sin /= hyp;
        // change elements in column i and then i+1, so loop through rows beneath the diagonal
        for (HighsInt j {i}; j < this->nullsp_dim_; j++){
            temp = cos * this->chol_[ locL(j,i-1) ] + sin * this->chol_[ locL(j, i) ];
            // TODO we are accessing a row-wise matrix by column, could be better
            //  when i == j the element (i, i+1) is zeroed out and then deleted so no need to change it
            if (j != i) this->chol_[ locL(j,i) ] = - sin * this->chol_[ locL(j,i-1) ] + cos * this->chol_[ locL(j, i) ];
            this->chol_[ locL(j,i-1) ] = temp; // in-place operation requires overwriting element when all computations are done
        }
    }
}

void AsmSolver::reducePadding(const HighsInt& idx, HighsInt& loc_remove){
    // this function is run if there was no padding vector that matches the activated one
    // argument is index of new constraint to activate
    // then choose which constraint to drop from V if not already given
    // first extract constraint and build HVec
    HVector newcol;
    this->buildConstraint(idx, newcol);
    double max_abs {0.};
    // arbitrarily choose which padding element to remove
    loc_remove = this->Q_.dim_ - 1; // default remove last element in V
    // now select which index to drop by finding largest element modulus in newcol (Z^T a_q)
    // TODO exploit sparsity of newcol in this loop
    for (HighsInt i {this->rangsp_dim_}; i < this->Q_.dim_; i++){ // Ztemp is a sparse vector
        if ( std::abs( newcol.array[ this->basis_perm_[i] ] ) > std::abs( max_abs ) ) {
            // only look at the trailing elements of the vector and 
            max_abs = newcol.array[ this->basis_perm_[i] ];
            loc_remove = i;
        }
    }
    // build oldcol
    HighsInt iRow = this->basis_perm_[loc_remove];
    HVector oldcol;
    this->buffer_.assign(this->Q_.dim_, 0.);
    this->buffer_[ iRow ] = 1.;
    this->stdvec2hvec(this->buffer_, oldcol);
    this->B_.btranCall(oldcol, 1.);
    // update basis matrix
    this->B_.update(&newcol, &oldcol, &iRow, &this->Bhint_);
    this->num_basis_updates_++;
    if (this->nullsp_dim_ == 1){
        this->chol_.resize(0);
    } else {
        assert(std::abs(max_abs) > this->options_.factor_pivot_tolerance );
        // update L factorisation according to (28) in Fletcher
        // first store -d_k/d_p coefficients at the end of buffer_
        for (HighsInt i { this->rangsp_dim_ }; i < loc_remove; i++){ // elements i < this->rangsp_dim_ - 1 in buffer_ are rubbish
            this->buffer_[i + 1] = - newcol.array[ this->basis_perm_[i] ] / max_abs;
        }
        for (HighsInt i { loc_remove + 1 }; i < this->Q_.dim_; i++){ // skip pivot element
            this->buffer_[i] = - newcol.array[ this->basis_perm_[i] ] / max_abs;
        }
        loc_remove -= this->rangsp_dim_; // normalise location of removal to size of nullspace
        // then explicitly permute L according to P^T L P
        HighsInt dim = this->nullsp_dim_ - 1;
        // if we remove the last row no need to permute anything
        if ( loc_remove < dim ) this->permute(loc_remove, dim); // TODO remove, takes too long
        // then add spike elements until full
        if (loc_remove > 0) this->addSpike(loc_remove - 1, dim); // permuting first row already gives full spike
        // then multiply out with (nullsp_dim_ - 1, nullsp_dim_)-size eta matrix
        // upper (nullsp_dim_ - 1, nullsp_dim_ - 1)-size triangle is unchanged
        for(HighsInt j {0}; j < dim; j++){
            this->chol_[ locL(dim, j) ] += this->chol_.back() * this->buffer_[this->rangsp_dim_ + 1 + j];
        }
        this->removeSpike(dim); // finally remove spike
        this->chol_.resize(this->chol_.size() - this->nullsp_dim_); // drop last row of L
        loc_remove += this->rangsp_dim_; // restore for index update
    }
    return;
}
// from Claude.ai to avoid having to allocate most memory, this mostly happens in-place
//void AsmSolver::permute(const HighsInt& loc_remove, const HighsInt& dim){
//    std::vector<double>::iterator itc = this->chol_.begin();
//    std::vector<double> lastrow(dim + 1);
//    std::copy(itc + locL(loc_remove,0), itc + locL(loc_remove,loc_remove), lastrow.begin());
//    lastrow[dim] = this->chol_[locL(loc_remove, loc_remove)];
//    for (HighsInt i {loc_remove + 1}; i <= dim; i++){
//        lastrow[i - 1] = this->chol_[locL(i, loc_remove)];
//        std::copy(itc + locL(i,0),            itc + locL(i,loc_remove), itc + locL(i-1,0));
//        std::copy(itc + locL(i,loc_remove+1), itc + locL(i,i) + 1,      itc + locL(i-1,loc_remove));
//    }
//    std::copy(lastrow.begin(), lastrow.end(), itc + locL(dim,0));
//}

void AsmSolver::permute(const HighsInt& loc_remove, const HighsInt& dim){
    // TODO if profiling shows this is a bottleneck, consider merging with addSpike in a dedicated implementation
    std::vector<double> new_chol(this->chol_.size());
    // upper triangle above permuted row stays the same
    std::vector<double>::iterator itc = this->chol_.begin();
    std::vector<double>::iterator itn = new_chol.begin();
    std::copy(itc, itc + (loc_remove * (loc_remove+1))/2, itn);
    // copy lower left rectangle and lower right triangle, line by line
    HighsInt uppertr_size;
    for (HighsInt i {loc_remove + 1}; i < this->nullsp_dim_; i++){
        uppertr_size = (i * (i-1))/2;
        std::copy(itc + locL(i,0), itc + locL(i,loc_remove), itn + uppertr_size );
        std::copy(itc + locL(i, loc_remove+1), itc + locL(i, i)+1, itn + uppertr_size + loc_remove);
    }
    // elements in permuted row that effectively go in the last row
    std::copy(itc + locL(loc_remove, 0), itc + locL(loc_remove, loc_remove), itn + locL(dim, 0));
    // then elements from permuted column that go in the last column (but last row in memeory)
    for (HighsInt i {loc_remove}; i < dim; i++){
        new_chol[ locL(dim, i) ] = this->chol_[ locL(i + 1, loc_remove) ];
    }
    // then final element
    new_chol.back() = this->chol_[ locL(loc_remove, loc_remove) ];
    this->chol_ = std::move(new_chol);
}

void AsmSolver::replace(){
    assert(this->getAsmBasisStatus(this->newactive_idx_) != this->newactive_status_);
    // update statuses
    if ( this->basis_idxs_[ this->relaxed_iloc_] != this->newactive_idx_ ){ // if new index is different than old index
        HVector newcol;
        HighsInt iRow = this->basis_perm_[ this->relaxed_iloc_ ];
        this->buildConstraint(this->newactive_idx_, newcol);
        // update basis matrix
        this->B_.update(&newcol, &this->Ztemp_, &iRow, &this->Bhint_);
        this->num_basis_updates_++;
        this->changeStatus( this->basis_idxs_[ this->relaxed_iloc_ ], AsmBasisStatus::kInactive );
        this->basis_idxs_[ this->relaxed_iloc_ ] = this->newactive_idx_; // update basis indices
    }
    this->changeStatus( this->newactive_idx_, this->newactive_status_ ); // update status of new active constraint
}