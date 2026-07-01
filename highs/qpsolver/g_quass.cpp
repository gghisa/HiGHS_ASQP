/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "qpsolver/g_quass.hpp"

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
    HighsStatus status_phase1 = highs.run();
    if (status_phase1 == HighsStatus::kError) return; // why not returning after extracting the model status too?
    model_status_ = highs.getModelStatus(); // note Optimal in Phase1 is Feasible for ASM
    basis_ = highs.getBasis();
    solution_ = highs.getSolution();
    lp.col_cost_ = col_cost; // restore objective cost
};

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
    // now we need to create a list of QP basis which we will use to track active and inactive constraints
    std::vector<HighsInt> active_set(lp.num_col_); // the active set should be as large as there are degrees of freedom
    // then we need to keep track of where these active constraints/bounds are active or if they are free variables
    std::vector<HighsBasisStatus> active_status(lp.num_col_); // for each element in active_set, there is a correponding entry here
    // these two are going to be modified together, so it would make sense to merge them into a class and operate on them at the same time

    // extract basis indexes and assign negative indexes to rows (from -1) and nonnegative indexes to columns
    for (int i{0}; i<lp.num_col_; i++){

    };
    for (int i{0}; i<lp.num_row_; i++){

    };

    // Once an initial basis is found, we can set up the loop to check whether the current point solves the current FSEP
    // by checking that a trivial step solves the EP

    // if we do: deactive a constraint and recompute nullspace and reduced hessian
    // otherwise: solve for a direction step and perfom ratio tests

    // at the end gather the solution and postsolve and stuff.

    // what do we need to keep track of while we solve? timer/logging....?

    printf("from inside my QP solver!!!!!\n");
    return HighsModelStatus::kNotset;
};