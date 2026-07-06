/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Highs.h"

class ActiveSetData
{
    public:
        explicit ActiveSetData(const HighsBasis& basis, const HighsLp& lp, const HighsHessian& Q){
            this->n_ = lp.num_col_;
            std::vector<HighsInt> basis_indices = setActiveVarCon(basis.col_status, basis.row_status);
            setupBasisMat(lp, basis, basis_indices);
        }

        void print(){
            std::cout<<"\nActive constraints:\n";
            for (size_t i {0}; i<active_con_.size(); i++){
                std::cout<< active_con_[i] << " -- ";
            }
            std::cout<<"\nActive variables:\n";
            for (size_t i {0}; i<active_var_.size(); i++){
                std::cout<< active_var_[i] << " -- ";
            }
        }
    private:
        HighsInt n_; // size of problem
        HighsInt n_active_; // size of active constraints
        HighsInt n_inactive_; // size of null space
        // need to store active set
        std::vector<HighsInt> active_var_; // store indices of variables at bounds
        std::vector<HighsInt> active_con_; // store indices of constraints at bounds
        std::vector<HighsBasisStatus> status_var_; // store type of activity for each active variable bound
        std::vector<HighsBasisStatus> status_con_; // store type of activity for each active constraint
        // need to store matrix B (HFactor object)
        HFactor basis_mat_;
        // need to store reduced hessian (dense matrix)
        std::vector<std::vector<double>> basis_nullspaceT; // store dense Z^T to have column-wise access to Z
        std::vector<std::vector<double>> reduced_hessian; // do we store the reduced hessian or the representation of its inverse?
        // given a vector of statuses, extract whether active or not and the corresponding location index
        size_t setActive(const std::vector<HighsBasisStatus>& status,
                         std::vector<HighsInt>& index,
                         std::vector<HighsBasisStatus>& active_status,
                         const size_t offset){
            size_t count {0};
            for (size_t i {0}; i<status.size(); i++){
                if (status[i] == HighsBasisStatus::kLower ||
                    status[i] == HighsBasisStatus::kUpper){ // kNonbasic... does it ever happen after phase1?
                    // when the constraints and bounds in the problem are less than the number of variables, then there may be kNonbasic elements
                    // they are not in the simplex basis, yet we need them to construct the invertible matrix B = [A:V]
                    index.push_back(i + offset);
                    active_status.push_back(status[i]);
                    count++;
                } else if (status[i] == HighsBasisStatus::kNonbasic){ // dont count as an active constraint, though still in base
                    // what about nonbasic kZero variables?
                    index.push_back(i + offset);
                    active_status.push_back(status[i]);
                }
            }
            return count;
        }
        // define function to run after phase1 to extract varumns and cons that are active
        std::vector<HighsInt> setActiveVarCon(const std::vector<HighsBasisStatus>& var_status,
                             const std::vector<HighsBasisStatus>& con_status){
            size_t count_var {0}, count_con {0};
            // variables' indexes start counting from m, which is the number of constraints. Constraint count starts from 0, as required by HFactor
            count_var = setActive(var_status, this->active_var_, this->status_var_, var_status.size());
            count_con = setActive(con_status, this->active_con_, this->status_con_, 0);
            this-> n_active_= count_var + count_con; // count the numbers of active constraints (may be lower than size of basis also to start with)
            this-> n_inactive_ = n_ - n_active_;
            std::vector<HighsInt> basis_indices; // create vector for the basis required by HFactor
            basis_indices.insert(basis_indices.end(), this->active_con_.begin(), this->active_con_.end());
            basis_indices.insert(basis_indices.end(), this->active_var_.begin(), this->active_var_.end());
            return basis_indices;
        }
        void setupBasisNullspace(){
            for (HighsInt i {this->n_active_}; i < n_; i++){
                // create unit vector
                std::vector<double> z_col(n_);
                z_col.assign(n_,0.);
                z_col[i] = 1.; // set unit entry at the index for the desired column of B^{-T}
                this->basis_mat_.btranCall(z_col); // solve B^T\cdot e_i = z_col
                // then add newfound vector to dense matrix
                this->basis_nullspaceT.push_back( z_col ); // add column vector to transpose of Z, effectively adding a new row
                // do I have to check if the size of the matrix is correct? should I check if it is empty?
            }
        }
        // setup basis matrix
        void setupBasisMat(const HighsLp& lp, const HighsBasis& basis, std::vector<HighsInt>& basis_indices){
            HighsSparseMatrix constraint_mat = lp.a_matrix_; // create a copy of the constraint matrix
            constraint_mat.ensureRowwise(); // flip the way in which it is stored
            constraint_mat.format_ = MatrixFormat::kColwise; // but "trick it" into thinking it is still stored columnwise
            HighsInt temp_old_num_row = constraint_mat.num_row_; // flip the number of rows and columns
            constraint_mat.num_row_ = constraint_mat.num_col_; // so that when the matrix is used by HFactor
            constraint_mat.num_col_ = temp_old_num_row; // it received the constraint matrix "column wise"
            this->basis_mat_.setup(constraint_mat, basis_indices); // where each column is a constraint. its inverse transpose will have as columns the nullspace basis
            // once the basis matrix is set up, extract the null spaces basis
            setupBasisNullspace();
        }
};