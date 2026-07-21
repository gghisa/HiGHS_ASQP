/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Highs.h"

HighsStatus gQP(HighsLp& lp,
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
        // properties ? TODO
        std::vector<HighsInt> basis_idxs_; // ordered set of indices in basis, active and inactive
        std::vector<HighsInt> nonbasis_idxs_; // can be unordered TODO, of indices outside basis
        std::vector<AsmBasisStatus> var_status_;
        std::vector<AsmBasisStatus> con_status_;
};

class ReducedHessian {
    public:
        explicit ReducedHessian(HighsHessian& Q);
        //
        HighsInt nullsp_dim_ {0};
        std::vector<double> chol_; // explicit hessian or its cholesky factor
        // functions
        void HSetup(HighsSparseMatrix& constraint_mat, std::vector<HighsInt>& basis_idxs);
        void HBuild();
        void HBtran(std::vector<double>& vec);
        void HBtran(HVector& vec, const double expected_density);
        void HFtran(std::vector<double>& vec);
        void HFtran(HVector& vec, const double expected_density);
        void HUpdate(HighsInt idx_drop, HighsInt idx_new);
        void recomputeExplicit();
        void refactorize();
        inline HighsInt loc(const HighsInt& i, const HighsInt& j);
        void extend(const HighsInt& loc_deactivated);
        void fsolve(std::vector<double>& vec);
        void bsolve(std::vector<double>& vec);
        void solve(std::vector<double>& vec);
        void getFullStep(const std::vector<double>& delta, std::vector<double> step);
    private:
        HighsHessian& Q_;
        HFactor B_;
        std::vector<std::vector<double>> ZT_; // explicit null space span
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
        void setupQpBasis();
        HighsStatus getHighsStatus();
        inline bool isActive(const AsmBasisStatus& status);
        inline bool isActiveInequality(const AsmBasisStatus& status);
        inline bool isInactive(const AsmBasisStatus& status);
        inline bool isInBasis(const AsmBasisStatus& status); // for setup
        inline bool isFreeInBasis(const AsmBasisStatus& status); // for setup
        void addNullSpaceDim();
        HighsInt getNullSpaceSize();
        void setupBasisMat();
        void setupReducedHessian();
        void run();
        void computeLocGrad();
        double norm(const std::vector<double>& vec);
        double computeRedGrad();
        void price();
        void deactivate();
        void solveREP();
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
        ReducedHessian M_; // in reduced hessian is the HFactor matrix B_
        AsmBasis qp_basis_;
        // Real numbers vectors
        std::vector<double> loc_grad_; // current gradient g + Q x_k, where x_k = solution_.col_value
        std::vector<double> red_grad_; // current reduced gradient Z^T (g + Q x_k)
        std::vector<double> pricing_; // a value for each active constraint, based on which the choice of which to deactivate is made
        std::vector<double> delta_; // reduced step, solution of M \delta = Z^T (g + Q x_k)
        std::vector<double> step_; // full step, result of Z \delta
        // Real numbers
        double alpha_; // step size for ratio test
        double tol_ {1e-7}; // tolerance for zero checks
        // Integers
        HighsInt iter_count_;     
        // functions
        AsmBasisStatus HighsStatusToAsm(const HighsBasisStatus& status, const HighsInt i, const bool variable);
};