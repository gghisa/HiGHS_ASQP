/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Highs.h"

class ReducedHessian {
    public:
        explicit ReducedHessian(HighsHessian& Q);
        const HighsInt getNullSpaceSize();
        void addOneNullSpaceDim();
        void Hsetup(HighsSparseMatrix& constraint_mat, std::vector<HighsInt>& basis_idxs);
        void Hbuild();
        void Hbtran(std::vector<double>& vec);
        void Hftran(std::vector<double>& vec);
        void init();
        void build();
        void solve(std::vector<double>& vec);
        void extend();
    private:
        HighsInt nullsp_dim_{0};
        HighsHessian& Q_;
        HFactor B_; // is this ok TODO even if the same element is member of another class?
        std::vector<double> chol_; // factorization vector storing lower triangular dense matrix row-wise
        std::vector<HighsInt> perm_; // column permutation indices of P, row permutations indices of P^T
        //TODO how does this^ change when matrix is extended?
        std::vector<std::vector<double>> ZT_;
        // from claude.ai
        inline HighsInt chol_idx(HighsInt i, HighsInt j) const {
            // returns the index for the chol_ vector given the indices for the triangular matrix it represents, stored row-wise as lower triangular
            return i*(i+1)/2 + j;
        }
        // from claude.ai
        inline double& chol(HighsInt i, HighsInt j) {
            return this->chol_[chol_idx(this->perm_[i], this->perm_[j])];
        }
        // from claude.ai
        inline double chol(HighsInt i, HighsInt j) const {
            return this->chol_[chol_idx(this->perm_[i], this->perm_[j])];
        }
};