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
        Highs feasibility_;
        HighsLp feasibility_lp_;
        double objective_; // objective function value
        HighsStatus status_ {HighsStatus::kOk}; // TODO update as you go
        HighsSparseMatrix HFactor_amatrix_;
        // ASM data
        HighsInt nullsp_dim_ {0};
        HighsInt rangsp_dim_ {this->Q_.dim_};
        std::vector<double> chol_; // explicit hessian or its cholesky factor
        HFactor B_;
        std::vector<double> buffer_; // for operations with permutations
        std::vector<double> Rtemp_; // buffer for the solution of L x = Z^T Q y_p
        std::vector<double> Qtemp_; // buffer for Q_.product(solution_.col_value) after updates
        HVector Ztemp_;
        double yQytemp_; 
        // Real numbers vectors
        std::vector<double> red_grad_; // current reduced gradient Z^T (g + Q x_k)
        std::vector<double> pricing_; // a value for each active constraint, based on which the choice of which to deactivate is made
        std::vector<double> step_; // full step, result of Z \delta
        // vectors for forward stepping
        std::vector<double> newvarvals_;
        std::vector<double> newconvals_;
        std::vector<double> newconpivots_;
        // Numbers
        HighsInt Bhint_ { 99999 }; // same number as Micheal in Basis::updatebasis
        double alpha_relaxed_ {1.}; // step size for relaxed (first pass) ratio test
        double alpha_ {1.}; // step size for second pass ratio test
        HighsInt num_basis_updates_ {0};
        HighsInt reinversion_freq_ { 10000 };
        bool diagonalQ_ {false};
        bool stepAlreadyTaken_ {false};
        bool atFSEP_ {true};
        HighsInt relaxed_idx_ {-1};
        HighsInt relaxed_iloc_ {-1};
        HighsInt newactive_idx_ {-1};
        AsmBasisStatus newactive_status_ { AsmBasisStatus::kInactive };
        // permutation has to be used when FTRAN and BTRAN are called
        std::vector<HighsInt> basis_idxs_; // ordered active and free indices in basiss
        std::vector<HighsInt> basis_perm_; // ordered active and free indices permutation in basis
        std::vector<HighsInt> HFactor_basis_;
        std::vector<AsmBasisStatus> var_status_;
        std::vector<AsmBasisStatus> con_status_;
        std::vector<HighsInt> degenerate_idxs_;
        std::vector<AsmBasisStatus> degenerate_status_;
        // HFactor functions
        void HBtran(std::vector<double>& vec);
        void HFtran(std::vector<double>& vec);
        void stdvec2hvec(const std::vector<double>& vec, HVector& hvec);
        // Reduced Hessian operations
        inline HighsInt locL(const HighsInt& i, const HighsInt& j){
            assert( j <= i );
            return i*(i+1)/2 + j; // assumes indices are given for lower triangular matrix
        }
        void recomputeRedHessian();
        void Lsolve(std::vector<double>& vec);
        void LTsolve(std::vector<double>& vec);
        void extend();
        void reduceInBasis(const HighsInt& loc_activated);
        void reducePadding(const HighsInt& idx, HighsInt& loc_remove);
        void rightGivensHess(const HighsInt& start);
        void addSpike(const HighsInt& start, const HighsInt& idx_last_col);
        void removeSpike(const HighsInt& idx_last_col);
        void permute(const HighsInt& loc_remove, const HighsInt& dim);
        void refresh(const double& max_abs, HVector& newcol);
        // Feasibility phase functions
        HighsModelStatus feasibility();
        void setupFeasibility();
        void setupQpBasis();
        void setupBasisMat();
        void buildRelaxedLp();
        // Main loop functions
        void computeReducedDirection();
        bool ratiotest_pass1();
        void ratiotest_pass2();
        void ratiotest();
        void activate();
        void replace();
        void doUpdates();
        HighsInt replaceWithUnitVec(HighsInt iloc, HVector& newcol, double& max_abs);
        // Object computations
        void computeRelaxedDirection();
        void updateRedGrad();
        void computeReducedVecs();
        void compute_varvals(const double& alpha, std::vector<double>& loc);
        void updateObjective();
        // Main loop breaks
        bool keepLooping();
        bool isOptimal();
        bool isUnbounded();
        // loggin
        HighsInt logged_atiter {0};
        double logged_attime {0.};
        void logging();
        std::string AsmStatusToString(const HighsInt idx);
        std::string AsmStatusToString(const AsmBasisStatus status);
        // Helper functions
        void extractPermutation();
        void buildZtemp(const HighsInt& iloc);
        bool isQdiagonal();
        AsmBasisStatus getAsmBasisStatus(const HighsInt& idx);
        void fromActiveToPadding(const HighsInt& loc);
        void fromPaddingToActive(const HighsInt& loc);
        void buildConstraint(const HighsInt& idx, HVector& hvec);
        void changeStatus(const HighsInt& idx, const AsmBasisStatus& newstatus);
        void findBestPrice(double& bestmultiplier, double& bestpricesign);
        static void ratio1(const double tol, const double denom, const double lower, const double upper,
                           const double oldval, const double newval, double& alpha); // static from Claude.ai
        static void ratio2(double& max_pivot, const double denom, const double lower, const double upper,
                           const double oldval, const double alphamax, double& alpha,
                           const HighsInt idx, HighsInt& newactive_idx, AsmBasisStatus& newactive_status);
        void cleanOutActive();
        void reinvertBasis();
        void addNullSpaceDim();
        void removeNullSpaceDim();
        AsmBasisStatus HighsStatusToAsm(const HighsBasisStatus& status, const HighsInt i, const bool variable);
        double norm(const std::vector<double>& vec);
        void AsmToHighs();
        HighsBasisStatus convertAsmBasisToHighs(const AsmBasisStatus& status);
};