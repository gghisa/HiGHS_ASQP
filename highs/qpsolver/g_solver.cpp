/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Highs.h"
#include "qpsolver/g_solver.hpp"

void gQP(HighsLp& lp,
         HighsBasis& basis,
         HighsSolution& solution,
         HighsModelStatus& model_status,
         HighsHessian& hessian,
         HighsTimer& timer){
    // initialiser solver object
    AsmSolver solver(lp, basis, solution, model_status, hessian, timer);
    solver.feasibility();
};

AsmBasis::AsmBasis(const HighsInt& num_var,
                   const HighsInt& num_con)
                   // initialise basis by allocating memory
                   : basis_idxs_(num_var),
                   inactive_idxs_(num_con),
                   var_status_(num_var),
                   con_status_(num_con){};

AsmSolver::AsmSolver(HighsLp& lp,
                     HighsBasis& basis,
                     HighsSolution& solution,
                     HighsModelStatus& model_status,
                     HighsHessian& Q,
                     HighsTimer& timer)
                     : lp_(lp),
                     lp_basis_(basis),
                     solution_(solution),
                     model_status_(model_status),
                     Q_(Q),
                     timer_(timer),
                     M_(Q),
                     qp_basis_(lp.num_col_, lp.num_row_){};

void AsmSolver::feasibility(){
    // TODO hotstart if basis is provided
    // TODO minimize slacks in this first phase
    std::vector<double> col_cost_temp = this->lp_.col_cost_; // store linear costs
    this->lp_.col_cost_.assign(this->lp_.num_col_, 0.); // zero out objective
    Highs qp_feasibility;
    qp_feasibility.passModel(this->lp_);
    qp_feasibility.setOptionValue("presolve", kHighsOnString); // presolving phase1 makes it faster, im guessing the postsolve is included
    qp_feasibility.setOptionValue("output_flag", false); // don't print anything
    qp_feasibility.setOptionValue("simplex_strategy", kSimplexStrategyDual); // specifying what solver to use in case a basis is set that is known to be either primal or dual feasible
    // use dual simplex if the objective value is all zeros, beacuse that means dual feasibility is guaranteed
    this->status_ = qp_feasibility.run();
    if (this->status_ != HighsStatus::kError){ // why not returning after extracting the model status too?
        this->model_status_ = qp_feasibility.getModelStatus(); // note Optimal in Phase1 is Feasible for ASM
        this->lp_basis_ = qp_feasibility.getBasis();
        this->solution_ = qp_feasibility.getSolution();
        this->objective_ = qp_feasibility.getObjectiveValue();
    }
};