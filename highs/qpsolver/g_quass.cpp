/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "qpsolver/g_quass.hpp"
#include "qpsolver/g_active_set.hpp"

HighsStatus QpPhase1(const HighsLp& lp, HighsModelStatus& model_status_,
              HighsBasis& basis_, HighsSolution& solution_,
              HighsTimer& timer_){
    //create a copy of the problem
    HighsLp ph1_lp {lp}; // local copy
    ph1_lp.col_cost_.assign(lp.num_col_, 0.0); // zero out all costs
    // create Highs instance for phase 1
    Highs qp_ph1; //rename this instance to qp-phase1-related name
    qp_ph1.passModel(ph1_lp);
    qp_ph1.setOptionValue("presolve", kHighsOnString); // presolving phase1 makes it faster, im guessing the postsolve is included
    qp_ph1.setOptionValue("output_flag", false); // don't print anything
    qp_ph1.setOptionValue("simplex_strategy", kSimplexStrategyPrimal); // specifying what solver to use in case a basis is set that is known to be either primal or dual feasible
    // use dual simplex if the objective value is all zeros, beacuse that means dual feasibility is guaranteed
    // any time limit? highs.setOptionValue("time_limit", ???);
    HighsStatus status_ph1 = qp_ph1.run();
    if (status_ph1 == HighsStatus::kError) return status_ph1; // why not returning after extracting the model status too?
    model_status_ = qp_ph1.getModelStatus(); // note Optimal in Phase1 is Feasible for ASM
    basis_ = qp_ph1.getBasis();
    solution_ = qp_ph1.getSolution();
    return status_ph1;
}

HighsModelStatus gQP(const HighsLp& lp, HighsHessian hessian, // make hessian a copy so that rest of QP solver from Micheal still works.
                    HighsModelStatus& model_status,
                    HighsBasis& basis, HighsSolution& solution,
                    HighsTimer& timer){

    // first we need a presolve. How many rules are needed? Is presolve used to fix issues that would otherwise
    // lead to the solver not solving?
    if (!basis.valid) { // if the basis is not valid run phase 1
        HighsStatus status_ph1 = QpPhase1(lp, model_status, basis, solution, timer); // simplex
        if (status_ph1 == HighsStatus::kError) return HighsModelStatus::kModelError; // is this returned object correct?
    }
    ActiveSetData asm_data(lp, basis, solution, hessian);
    for (HighsInt i {0}; i < 1; i++){ // set iteration limit
        if (asm_data.getSizeNullSpace() == 0){
            asm_data.price();
            // we now have access to a list of prices, but what price corresponds to what constraint?
            // check if it is possible to deactivate a constraint
            // else break and return optimal
        } else {
            // solve current equality problem to find descent direction
            // perform ratio test then possibly activate constraint and continue
        }
    }
    // at the end gather the solution and postsolve and stuff.

    // what do we need to keep track of while we solve? timer/logging....?

    std::cout<< "\nfrom inside my QP solver!!!!!"<<"\n";
    return HighsModelStatus::kNotset;
}