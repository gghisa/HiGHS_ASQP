/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "qpsolver/g_quass.hpp"
#include "qpsolver/g_active_set.hpp"

void QpPhase1(HighsLp& lp, HighsModelStatus& model_status_,
              HighsBasis& basis_, HighsSolution& solution_,
              HighsTimer& timer_){
    // set up new linear problem
    std::vector<double> col_cost {lp.col_cost_}; // store objective cost
    lp.col_cost_.assign(lp.num_col_, 0.0); // zero out all costs
    // create Highs lp
    Highs highs;
    highs.passModel(lp);
    highs.setOptionValue("presolve", kHighsOnString); // presolving phase1 makes it faster, im guessing the postsolve is included
    highs.setOptionValue("output_flag", false); // don't print anything
    highs.setOptionValue("simplex_strategy", kSimplexStrategyPrimal); // do we need to specify this? is it always the faster option?
    // any time limit? highs.setOptionValue("time_limit", ???);
    HighsStatus status_phase1 = highs.run();
    if (status_phase1 == HighsStatus::kError) return; // why not returning after extracting the model status too?
    model_status_ = highs.getModelStatus(); // note Optimal in Phase1 is Feasible for ASM
    basis_ = highs.getBasis();
    solution_ = highs.getSolution();
    lp.col_cost_ = col_cost; // restore objective cost
}

HighsModelStatus gQP(HighsLp& lp, HighsHessian& hessian,
                    HighsModelStatus& model_status_,
                    HighsBasis& basis_, HighsSolution& solution_,
                    HighsTimer& timer_){

    // first we need a presolve. How many rules are needed? Is presolve used to fix issues that would otherwise
    // lead to the solver not solving?

    // then we need to find an initial feasible point. We can run Highs simplex with an empty objective, but ideally
    // we change it so that it finds a starting point with the least amount of free variables, so that the null space
    // is as small as possible to start with, so that factorization of the reduced hessian does not start with a large
    // matrix factorization. This is Micheal's suggestion, is it indeed good to implement? How?

    QpPhase1(lp, model_status_, basis_, solution_, timer_); // simplex
    ActiveSet active_set(basis_.col_status, basis_.row_status); // extract active set
    active_set.print(); // debug TOREMOVE
    // Once an initial basis is found, we can set up the loop to check whether the current point solves the current FSEP
    // by checking that a trivial step solves the EP

    // if we do: deactive a constraint and recompute nullspace and reduced hessian
    // otherwise: solve for a direction step and perfom ratio tests

    // at the end gather the solution and postsolve and stuff.

    // what do we need to keep track of while we solve? timer/logging....?

    std::cout<< "\nfrom inside my QP solver!!!!!"<<"\n";
    return HighsModelStatus::kNotset;
}