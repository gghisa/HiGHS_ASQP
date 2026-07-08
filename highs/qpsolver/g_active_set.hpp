/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Highs.h"

class ActiveSetData {
    public:
        explicit ActiveSetData(const HighsBasis& basis,
                               const HighsLp& lp,
                               HighsSolution& solution,
                               HighsHessian& Q);

        size_t getSizeNullSpace();
        void printActive();
        void printvector(const std::vector<double>& vec);
        void printmatrix(const std::vector<std::vector<double>>& mat);
        void printsparse(const HighsSparseMatrix& mat);
    private:
            // members from initialisation arguments
        const HighsLp& lp_;
        HighsSolution& solution_;
        HighsHessian& Q_;
        // define own basis (better way?)
        std::vector<HighsInt> active_var_; // store indices of variables at bounds
        std::vector<HighsInt> active_con_; // store indices of constraints at bounds
        std::vector<HighsBasisStatus> status_var_; // store type of activity for each active variable bound
        std::vector<HighsBasisStatus> status_con_; // store type of activity for each active constraint
        // matrices
        HFactor B_; // basis matrix
        std::vector<std::vector<double>> ZT_; // nullspace basis, dense, gives column-wise access to Z
        std::vector<std::vector<double>> redhes_; // do we store the reduced hessian or the representation of its inverse?
        // vectors
        std::vector<double> loc_grad_; // current gradient g + Q x_k, where x_k = solution_.col_value
        std::vector<double> red_grad_; // current reduced gradient Z^T (g + Q x_k)
        std::vector<double> pricing_; // a value for each active constraint, based on which the choice of which to deactivate is made
        // setup functions
        void setupActive(const std::vector<HighsBasisStatus>& status,
                            std::vector<HighsInt>& index,
                            std::vector<HighsBasisStatus>& active_status,
                            const HighsInt offset);
        std::vector<HighsInt> setupActiveVarCon(const std::vector<HighsBasisStatus>& var_status,
                                                const std::vector<HighsBasisStatus>& con_status);
        void setupBasisMat(const HighsBasis& basis, std::vector<HighsInt>& basis_indices);
        void setupBasisNullSpace();
        void setupReducedHessian();
        void computeLocGrad();
        void computeRedGrad();
        void price();
};