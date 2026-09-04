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
        explicit AsmSolver(const HighsOptions& options,
                           HighsTimer& timer,
                           HighsLp lp,
                           HighsHessian hessian,
                           HighsBasis& basis,
                           HighsSolution& solution,
                           HighsModelStatus& model_status,
                           HighsInfo& info,
                           HighsCallback& callback);
        HighsStatus getHighsStatus();
        HighsModelStatus getHighsModelStatus();
        HighsStatus run();
    private:
        // problem data
        const HighsOptions& options_;
        HighsTimer& timer_;
        HighsLp lp_;
        HighsLp lp_relaxed_; // for double pass ratiotest
        HighsHessian Q_;
        HighsBasis& lp_basis_; // basis with HighsBasisStatus vectors
        HighsSolution& solution_;
        HighsModelStatus& model_status_;
        HighsInfo& info_;
        HighsCallback& callback_;
        //
        HighsLp feasibility_lp_;
        double objective_; // objective function value
        HighsStatus status_ {HighsStatus::kOk}; // TODO update as you go
        // ASM data
        HighsInt nullsp_dim_ {0};
        HighsInt rangsp_dim_ {this->Q_.dim_};
        std::vector<double> chol_; // explicit hessian or its cholesky factor
        HFactor B_;
        std::vector<double> buffer_; // for operations with permutations
        std::vector<HighsInt> Vi_; // vector of indices of the unit vectors padding A in B
        // Real numbers vectors
        std::vector<double> loc_grad_; // current gradient g + Q x_k, where x_k = solution_.col_value
        std::vector<double> red_grad_; // current reduced gradient Z^T (g + Q x_k)
        std::vector<double> pricing_; // a value for each active constraint, based on which the choice of which to deactivate is made
        std::vector<double> delta_; // reduced step, solution of M \delta = Z^T (g + Q x_k)
        std::vector<double> step_; // full step, result of Z \delta
        // vectors for forward stepping
        std::vector<double> newvarvals_;
        std::vector<double> newconvals_;
        std::vector<double> newconpivots_;
        // Numbers
        double alpha_relaxed_ {1.}; // step size for ratio test
        HighsInt n_iter_ {0};
        // Truth values
        bool step_taken_ {false};
        // permutation has to be used when FTRAN and BTRAN are called
        std::vector<HighsInt> basis_idxs_; // ordered active and free indices in basiss
        std::vector<HighsInt> basis_perm_; // ordered active and free indices permutation in basis
        std::vector<AsmBasisStatus> var_status_;
        std::vector<AsmBasisStatus> con_status_;
        std::vector<HighsInt> degenerate_idxs_;
        std::vector<AsmBasisStatus> degenerate_status_;
        // HFactor functions
        void HBtran(std::vector<double>& vec);
        void HFtran(std::vector<double>& vec);
        HVector stdvec2hvec(const std::vector<double>& vec, HVector& hvec);
        // Reduced Hessian operations
        HighsInt locL(const HighsInt& i, const HighsInt& j);
        void recomputeExplicit();
        void refactorize();
        void Lsolve(std::vector<double>& vec);
        void LTsolve(std::vector<double>& vec);
        void LLTsolve(std::vector<double>& vec);
        void extend(const HighsInt& loc_deactivated, const HighsInt& idx_deactivated);
        void reduce(const HighsInt& loc_activated);
        void rightGivensHess(const HighsInt& i);
        void addSpikeElement(const HighsInt& i);
        void removeSpikeElement(const HighsInt& i);
        // Feasibility phase functions
        void feasibility();
        void setupFeasibilityLp();
        void setupQpBasis();
        void setupBasisMat(std::vector<HighsInt>& basis_idxs);
        void buildRelaxedLp();
        // Main loop functions
        void deactivate();
        void ratiotest_pass1();
        void ratiotest_pass2(HighsInt& newactive_idx, AsmBasisStatus& newactive_status);
        void takeStep();
        void activate(const HighsInt& idx, const AsmBasisStatus& status);
        // Object computations
        void computeLocGrad();
        void computeReducedVecs();
        void compute_varvals(const double& alpha, std::vector<double>& loc);
        void computeFullStep(const std::vector<double>& delta, std::vector<double>& step);
        double computeQuadObjective(const std::vector<double>& vec);
        void updateObjective();
        void signPrices();
        // Main loop breaks
        bool iterlimit();
        bool timelimit();
        bool maximalsteptaken();
        bool nullsizelimit();
        bool isoptimal();
        // Helper functions
        static void ratio1(const double tol, const double denom, const double lower, const double upper,
                           const double oldval, const double newval, double& alpha); // static from Claude.ai
        static void ratio2(double& max_pivot, const double denom, const double lower, const double upper,
                           const double oldval, const double newval, const double alpha,
                           const HighsInt idx, HighsInt& newactive_idx, AsmBasisStatus& newactive_status);
        void stepSanity();
        void addNullSpaceDim();
        void removeNullSpaceDim();
        AsmBasisStatus HighsStatusToAsm(const HighsBasisStatus& status, const HighsInt i, const bool variable);
        double norm(const std::vector<double>& vec);
};