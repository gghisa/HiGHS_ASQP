/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Highs.h"
#include "qpsolver/g_red_hessian.hpp"

void gQP(HighsLp& lp,
         HighsBasis& basis,
         HighsSolution& solution,
         HighsModelStatus& model_status,
         HighsHessian& hessian,
         HighsTimer& timer);

enum class AsmBasisStatus : HighsInt {
    kUpper = -1, // active at upper bound, sign flip for pricing
    kEquality = 0, // always active
    kLower = 1, // active at lower bound
    kFreeInBasis = 2, // not active but in basis matrix, considered in ratio test
    kInactive = 3 // not active and not in basis, considered in ratio test
};

class AsmBasis {
    public:
    explicit AsmBasis(const HighsInt& num_var,
                      const HighsInt& num_con);
    private:
        std::vector<HighsInt> basis_idxs_; // ordered set
        std::vector<HighsInt> inactive_idxs_; // can be unordered TODO
        std::vector<AsmBasisStatus> var_status_;
        std::vector<AsmBasisStatus> con_status_;
};

class ReducedHessian {
    public:
        explicit ReducedHessian(HighsHessian& Q);
    private:
        HighsHessian& Q_;
        HighsInt nullsp_dim_ {-1}; // uninitialised value
};

class AsmSolver {
    public:
        explicit AsmSolver(HighsLp& lp,
                           HighsBasis& basis,
                           HighsSolution& solution,
                           HighsModelStatus& model_status,
                           HighsHessian& Q,
                           HighsTimer& timer);
        void feasibility();
    private:
        // problem data
        HighsLp& lp_;
        HighsBasis& lp_basis_; // basis with HighsBasisStatus vectors
        HighsSolution& solution_;
        HighsModelStatus& model_status_;
        HighsHessian& Q_;
        HighsTimer& timer_;
        double objective_; // objective function value
        HighsStatus status_;
        // ASM data
        ReducedHessian M_;
        AsmBasis qp_basis_;
        // Real numbers vectors
        std::vector<double> loc_grad_; // current gradient g + Q x_k, where x_k = solution_.col_value
        std::vector<double> red_grad_; // current reduced gradient Z^T (g + Q x_k)
        std::vector<double> pricing_; // a value for each active constraint, based on which the choice of which to deactivate is made
        std::vector<double> delta_; // reduced step, solution of M \delta = Z^T (g + Q x_k)
        std::vector<double> step_; // full step, result of Z \delta
        // Real numbers
        double alpha_; // step size for ratio test
        // Integers
        HighsInt iter_count_;        
};