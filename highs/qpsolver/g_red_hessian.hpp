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
        void Hsetup(HighsSparseMatrix& constraint_mat, std::vector<HighsInt>& basis_idxs);
        void Hbuild();
        void Hbtran(std::vector<double>& vec);
        void Hftran(std::vector<double>& vec);
        void build();
        void extend();
    private:
        HighsHessian& Q_;
        HFactor B_; // is this ok TODO even if the same element is member of another class?
        std::vector<double> chol; // factorization vector storing lower triangular dense matrix row-wise
};