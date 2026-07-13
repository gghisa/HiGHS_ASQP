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
        void extend(std::vector<double> y);
    private:
        HighsHessian& Q;
        std::vector<double> chol; // factorization vector storing lower triangular dense matrix row-wise
};