/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Highs.h"

enum class AsmBasisStatus : HighsInt {
    kUpper = -1, // active at upper bound, sign flip for pricing
    kEquality = 0, // always active
    kLower = 1, // active at lower bound
    kFreeInBasis = 2, // not active but in basis matrix, considered in ratio test
    kInactive = 3 // not active and not in basis, considered in ratio test
};

class AsmSolver {
    public:
        explicit AsmSolver(HighsLp& lp,
                           HighsBasis& basis,
                           HighsSolution& solution,
                           HighsModelStatus& model_status,
                           HighsHessian Q,
                           HighsTimer& timer,
                           HighsOptions& options);
        inline HighsStatus getHighsStatus();
        inline HighsModelStatus getHighsModelStatus();
        HighsStatus run();
    private:
        // problem data
        HighsLp& lp_;
        HighsLp feasibility_lp_;
        HighsBasis& lp_basis_; // basis with HighsBasisStatus vectors
        HighsSolution& solution_;
        HighsModelStatus& model_status_;
        HighsHessian Q_;
        HighsTimer& timer_;
        HighsOptions& options_;
        double objective_; // objective function value
        HighsStatus status_ {HighsStatus::kOk}; // TODO update as you go
        // ASM data
        HighsInt nullsp_dim_ {0};
        HighsInt rangsp_dim_ {this->Q_.dim_};
        std::vector<double> chol_; // explicit hessian or its cholesky factor
        HFactor B_;
        std::vector<std::vector<double>> ZT_; // explicit null space span
        std::vector<double> buffer_; // for operations with permutations
        // Real numbers vectors
        std::vector<double> loc_grad_; // current gradient g + Q x_k, where x_k = solution_.col_value
        std::vector<double> red_grad_; // current reduced gradient Z^T (g + Q x_k)
        std::vector<double> pricing_; // a value for each active constraint, based on which the choice of which to deactivate is made
        std::vector<double> delta_; // reduced step, solution of M \delta = Z^T (g + Q x_k)
        std::vector<double> step_; // full step, result of Z \delta
        // Real numbers
        double alpha_ {1.}; // step size for ratio test (TODO unused for now)
        double tol_ {1e-7}; // tolerance for zero checks
        // permutation has to be used when FTRAN and BTRAN are called
        std::vector<HighsInt> basis_idxs_; // ordered active and free indices in basiss
        std::vector<HighsInt> basis_perm_; // ordered active and free indices permutation in basis
        std::vector<AsmBasisStatus> var_status_;
        std::vector<AsmBasisStatus> con_status_;
        // HFactor functions
        void HSetup(const HighsSparseMatrix& constraint_mat);
        void HBuild();
        void HBtran(std::vector<double>& vec);
        void HFtran(std::vector<double>& vec);
        void HBtran(HVector& vec, const double expected_density);
        void HFtran(HVector& vec, const double expected_density);
        void HUpdate(HighsInt loc_idxdrop, HighsInt idx_new);
        HVector stdvec2hvec(const std::vector<double>& vec);
        HVector unit_hvec(const HighsInt& p);
        // Reduced Hessian operations
        inline HighsInt locL(const HighsInt& i, const HighsInt& j);
        void recomputeExplicit();
        void refactorize();
        void Lsolve(std::vector<double>& vec);
        void LTsolve(std::vector<double>& vec);
        void LLTsolve(std::vector<double>& vec);
        void extend(const HighsInt& loc_deactivated);
        void reduce();
        // Feasibility phase functions
        bool feasibility();
        void setupFeasibilityProblem();
        void setupQpBasis();
        void setupBasisMat();
        void setupReducedHessian();
        // Main loop functions
        bool deactivate();
        void solveREP();
        void ratiotest();
        void activate(const HighsInt& idx, const AsmBasisStatus& status);
        // Object computations
        void computeLocGrad();
        double computeReducedVecs();
        void compute_newloc(const double& alpha, std::vector<double>& loc);
        void computeFullStep(const std::vector<double>& delta, std::vector<double>& step);
        double computeQuadObjective(const std::vector<double>& vec);
        void updateObjective();
        void signPrices();
        // Helper functions
        inline void addNullSpaceDim();
        inline void removeNullSpaceDim();
        AsmBasisStatus HighsStatusToAsm(const HighsBasisStatus& status, const HighsInt i, const bool variable);
        inline bool isInBasis(const AsmBasisStatus& status);
        inline bool isFreeInBasis(const AsmBasisStatus& status);
        inline bool isActive(const AsmBasisStatus& status);
        inline bool isActiveInequality(const AsmBasisStatus& status);
        inline bool isInactive(const AsmBasisStatus& status);
        double norm(const std::vector<double>& vec);
};