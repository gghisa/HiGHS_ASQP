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
        explicit ActiveSetData(const HighsBasis& basis, const HighsLp& lp, HighsHessian& Q){
            this->n_ = lp.num_col_;
            this->m_ = lp.num_row_;
            this->n_inactive_ = n_;
            Q = Q.toSquare(); // make Hessian square to improve columns/rows accessing speed
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

        void printvector(const std::vector<double>& vec){
            // from Claude.ai
            for (double val : vec) {
                std::cout << val << " ";
            }
            std::cout.flush();
        }
        void printmatrix(const std::vector<std::vector<double>>& mat){
            // from Claude.ai
            for (const std::vector<double>& row : mat) {
                for (double val : row) {
                    std::cout << val << " ";
                }
                std::cout << "\n";
            }
            std::cout.flush();
        }
        void printsparse(const HighsSparseMatrix& mat){
            std::vector<std::vector<double>> dense;
            for (HighsInt i {0}; i < mat.num_row_; i++){
                std::vector<double> col;
                std::vector<double> x;
                x.assign(mat.num_row_, 0.);
                x[i] = 1.;
                mat.productTranspose(col, x);
                dense.push_back(col);
            }
            printmatrix(dense);
        }
    private:
        HighsInt n_; // number of variables
        HighsInt m_; // number of constraints
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
        void setActive(const std::vector<HighsBasisStatus>& status,
                         std::vector<HighsInt>& index,
                         std::vector<HighsBasisStatus>& active_status,
                         const size_t offset){
            size_t count {0};
            for (size_t i {0}; i<status.size(); i++){
                if (status[i] != HighsBasisStatus::kBasic){
                    // when the constraints and bounds in the problem are less than the number of variables, then there may be kNonbasic elements
                    // they are not in the simplex basis, yet we need them to construct the invertible matrix B = [A:V]
                    index.push_back(i + offset);
                    active_status.push_back(status[i]);
                    if (status[i] != HighsBasisStatus::kZero) this->n_inactive_--; // kZero means one available nullspace dimension
                    // kNonbasic is ignored
                }
            }
        }
        // define function to run after phase1 to extract varumns and cons that are active
        std::vector<HighsInt> setActiveVarCon(const std::vector<HighsBasisStatus>& var_status,
                             const std::vector<HighsBasisStatus>& con_status){
            // variables' indexes start counting from m, which is the number of constraints. Constraint count starts from 0, as required by HFactor
            setActive(var_status, this->active_var_, this->status_var_, this->m_);
            setActive(con_status, this->active_con_, this->status_con_, 0);
            std::vector<HighsInt> basis_indices; // create vector for the basis required by HFactor
            basis_indices.insert(basis_indices.end(), this->active_con_.begin(), this->active_con_.end());
            basis_indices.insert(basis_indices.end(), this->active_var_.begin(), this->active_var_.end());
            return basis_indices;
        }
        void setupBasisNullspace(){
            HighsInt n_active = this->n_ - this->n_inactive_; // wouldn't it be better to store this as a member of the class?
            for (HighsInt i {n_active}; i < this->n_; i++){
                std::vector<double> z_col(n_); // create unit vector
                z_col.assign(n_,0.);
                z_col[i] = 1.; // set unit entry at the index for the desired column of B^{-T}
                this->basis_mat_.btranCall(z_col); // solve B^T\cdot e_i = z_col
                this->basis_nullspaceT.push_back( z_col ); // add column vector to transpose of Z, effectively adding a new row
                // do I have to check if the size of the matrix is correct? should I check if it is empty?
            }
            printmatrix(this->basis_nullspaceT);
        }
        // setup basis matrix
        void setupBasisMat(const HighsLp& lp, const HighsBasis& basis, std::vector<HighsInt>& basis_indices){
            HighsSparseMatrix constraint_mat = lp.a_matrix_; // create a copy of the constraint matrix
            printsparse(constraint_mat);
            constraint_mat.ensureRowwise(); // flip the way in which it is stored
            constraint_mat.format_ = MatrixFormat::kColwise; // but "trick it" into thinking it is still stored columnwise
            HighsInt temp_old_num_row = constraint_mat.num_row_; // flip the number of rows and columns
            constraint_mat.num_row_ = constraint_mat.num_col_; // so that when the matrix is used by HFactor
            constraint_mat.num_col_ = temp_old_num_row; // it received the constraint matrix "column wise"
            this->basis_mat_.setup(constraint_mat, basis_indices); // where each column is a constraint. its inverse transpose will have as columns the nullspace basis
            // once the basis matrix is set up, extract the null spaces basis
            this->basis_mat_.build();
            setupBasisNullspace();
        }
        void init_reduced_hessian(const HighsHessian& Q){
            // compute Z^T Q Z = ( Q Z )^T Z
        }
};