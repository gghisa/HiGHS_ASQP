/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "qpsolver/g_quass.hpp"
#include "qpsolver/g_active_set.hpp"

void QpPhase1(const HighsLp& lp, HighsModelStatus& model_status_,
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
    if (status_ph1 == HighsStatus::kError) return; // why not returning after extracting the model status too?
    model_status_ = qp_ph1.getModelStatus(); // note Optimal in Phase1 is Feasible for ASM
    basis_ = qp_ph1.getBasis();
    solution_ = qp_ph1.getSolution();
}

HighsModelStatus gQP(const HighsLp& lp, const HighsHessian& hessian, // can remove const and modify
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
    ActiveSet active_set; // do we need an active set object? how would you modify vectors in here?
    std::vector<HighsInt> basis = active_set.setup(basis_.col_status, basis_.row_status); // extract active set
    active_set.print(); // debug TOREMOVE
    HFactor basis_mat; // basis matrix, where columns are the basis vectors
    HighsSparseMatrix constraint_mat = lp.a_matrix_; // create a copy of the constraint matrix
    constraint_mat.ensureRowwise(); // flip the way in which it is stored
    constraint_mat.format_ = MatrixFormat::kColwise; // but "trick it" into thinking it is still stored columnwise
    HighsInt temp_old_num_row = constraint_mat.num_row_; // flip the number of rows and columns
    constraint_mat.num_row_ = constraint_mat.num_col_; // so that when the matrix is used by HFactor
    constraint_mat.num_col_ = temp_old_num_row; // it received the constraint matrix "column wise"
    basis_mat.setup(constraint_mat, basis); // where each column is a constraint. its inverse transpose will have as columns the nullspace basis
    // Once an initial basis is found, we can set up the loop to check whether the current point solves the current FSEP
    // first find the reduced hessian
    // so first compute Z by btran calls of basis_mat

    // by checking that a trivial step solves the EP

    // if we do: deactive a constraint and recompute nullspace and reduced hessian
    // otherwise: solve for a direction step and perfom ratio tests

    // at the end gather the solution and postsolve and stuff.

    // what do we need to keep track of while we solve? timer/logging....?

    std::cout<< "\nfrom inside my QP solver!!!!!"<<"\n";
    return HighsModelStatus::kNotset;
}