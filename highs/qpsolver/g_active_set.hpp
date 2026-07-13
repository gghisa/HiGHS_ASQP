/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Highs.h"
#include "qpsolver/g_red_hessian.hpp"

enum class AsmBasisStatus : HighsInt {
    kUpper = -1, // active at upper bound, sign flip for pricing
    kEquality = 0, // always active
    kLower = 1, // active at lower bound
    kFreeInBasis = 2, // not active but in basis matrix, considered in ratio test
    kInactive = 3 // not active and not in basis, considered in ratio test
};

class ActiveSetData {
    public:
        explicit ActiveSetData(const HighsLp& lp,
                               const HighsBasis& basis,
                               HighsSolution& solution,
                               HighsHessian& Q);

        HighsInt getSizeNullSpace();
        HighsInt getSizeRangeSpace();
        // void printActive(); TODO if needed
        void printvector(const std::vector<double>& vec);
        void printmatrix(const std::vector<std::vector<double>>& mat);
        void printsparse(const HighsSparseMatrix& mat);
        HighsModelStatus deactivate();
    private:
        // members from initialisation arguments
        const HighsLp& lp_;
        // kBasic    : inactive constraint/variable       -> Inactive, not in ASM basis
        // kNonbasic : never returned by simplex phase 1  -> ?
        // kZero     : free var/con                       -> Inactive, completes ASM basis (makes sense?)
        // kLower    : var/con at lower bound or equality -> Active, necessarily in ASM basis
        // kUpper    : var/con at upper bound             -> Active, necessarily in ASM basis
        HighsSolution& solution_;
        HighsHessian& Q_;
        // matrices
        HFactor B_; // basis matrix
        // would it be better to only have one inverse matrix?
        // also, since I am extracting it, I am getting an explicit representation of the inverse
        // is that normal?
        HighsInt nullsp_dim_; // Null Space dimension
        std::vector<std::vector<double>> ZT_; // nullspace basis, dense, gives column-wise access to Z
        std::vector<std::vector<double>> redhes_; // do we store the reduced hessian or the representation of its inverse?
        // basis information
        std::vector<HighsInt> basis_idxs_; // for HFactor and to keep up to date
        std::vector<AsmBasisStatus> basis_status_; // to keep up to date
        // Real numbers vectors
        std::vector<double> loc_grad_; // current gradient g + Q x_k, where x_k = solution_.col_value
        std::vector<double> red_grad_; // current reduced gradient Z^T (g + Q x_k)
        std::vector<double> pricing_; // a value for each active constraint, based on which the choice of which to deactivate is made
        // setup functions
        void initAsmBasis(const HighsBasis& basis);
        void initAsmBasisLoop(const std::vector<HighsBasisStatus>& status, const bool isconstr);
        void setupBasisMat();
        void setupInvBasisSpace();
        void setupReducedHessian();
        HighsBasisStatus AsmStatusToHighs(const AsmBasisStatus& status);
        AsmBasisStatus HighsStatusToAsm(const HighsBasisStatus& status, HighsInt index);
        // computations
        void computeLocGrad();
        void computeRedGrad();
        void price();
        void extendReducedHessian();
        //
        bool isActiveInequality(const AsmBasisStatus& status);
};