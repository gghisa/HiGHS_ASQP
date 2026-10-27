/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file qpsolver/QpAsmWrapper.cpp
 * @brief
 */

#include "qpsolver/QpAsmWrapper.h"

#include "model/HighsHessianUtils.h"
#include "qpsolver/g_solver.hpp"

HighsStatus solveQpAsm(HighsQpSolverObject& solver_object) {
  return solveQpAsm(solver_object.options_, solver_object.timer_,
                    solver_object.model_.lp_, solver_object.model_.hessian_,
                    solver_object.basis_, solver_object.solution_,
                    solver_object.model_status_, solver_object.highs_info_,
                    solver_object.callback_);
}

HighsHessianFunctionType testOracleCallSquareHessian =
    [](const HighsInt call_type, const HighsInt* x_num_entries,
       const HighsInt* x_index, const double* x_value,
       HighsInt* q_x_num_entries, HighsInt* q_x_index, double* q_x_value,
       void* hessian_p) {
      assert(kHessianOracleCallTypeMin <= call_type &&
             call_type <= kHessianOracleCallTypeMax);

      HighsHessian hessian = *(static_cast<HighsHessian*>(hessian_p));
      assert(hessian.format_ == HessianFormat::kSquare);

      // Lambda for adding multiple of Hessian column into q_x_value
      auto addScaledQcol = [&](const HighsInt iCol, const double x_value) {
        for (HighsInt iEl = hessian.start_[iCol];
             iEl < hessian.start_[iCol + 1]; iEl++) {
          HighsInt iRow = hessian.index_[iEl];
          q_x_value[iRow] += hessian.value_[iEl] * x_value;
        }
      };

      if (call_type == kHessianOracleCallTypeEntry) {
        assert(x_num_entries == nullptr);
        assert(x_value == nullptr);
        assert(x_index != nullptr);
        assert(q_x_num_entries == nullptr);
        assert(q_x_index != nullptr);
        assert(q_x_value != nullptr);
        HighsInt iCol = x_index[0];
        HighsInt iRow = q_x_index[0];
        // Zero Qx value in case the Hessian entry requested is zero
        q_x_value[0] = 0;
        for (HighsInt iEl = hessian.start_[iCol];
             iEl < hessian.start_[iCol + 1]; iEl++) {
          if (hessian.index_[iEl] == iRow) {
            q_x_value[0] = hessian.value_[iEl];
            return 0;
          }
        }
      } else if (call_type == kHessianOracleCallTypeColumn) {
        // Get the entries in column iCol
        assert(x_num_entries == nullptr);
        assert(x_value == nullptr);
        assert(x_index != nullptr);
        assert(q_x_num_entries != nullptr);
        assert(q_x_index != nullptr);
        assert(q_x_value != nullptr);
        (*q_x_num_entries) = 0;
        HighsInt iCol = x_index[0];
        for (HighsInt iEl = hessian.start_[iCol];
             iEl < hessian.start_[iCol + 1]; iEl++) {
          q_x_index[*q_x_num_entries] = hessian.index_[iEl];
          q_x_value[*q_x_num_entries] = hessian.value_[iEl];
          (*q_x_num_entries)++;
        }
      } else {
        assert(x_index == nullptr || *x_num_entries >= 0);
        assert(q_x_num_entries == nullptr);
        assert(q_x_index == nullptr);
        assert(q_x_value != nullptr);
        if (x_index == nullptr) {
          // Simple product with full vector x, full vector q_x
          for (HighsInt iCol = 0; iCol < hessian.dim_; iCol++)
            addScaledQcol(iCol, x_value[iCol]);
        } else {
          // x is scattered with x_num_entries entries in rows x_index
          for (HighsInt iX = 0; iX < *x_num_entries; iX++) {
            HighsInt iCol = x_index[iX];
            addScaledQcol(iCol, x_value[iCol]);
          }
        }
      }
      return 0;
    };

HighsStatus solveQpAsm(const HighsOptions& options, HighsTimer& timer,
                       const HighsLp& lp, const HighsHessian& hessian,
                       HighsBasis& basis, HighsSolution& solution,
                       HighsModelStatus& model_status, HighsInfo& info,
                       HighsCallback& callback) {
  AsmSolver asm_solver(options,
                       timer,
                       lp,
                       hessian,
                       basis,
                       solution,
                       model_status,
                       info,
                       callback);
  return asm_solver.run();
}
