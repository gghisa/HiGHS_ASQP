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
}

class ActiveSet
{
    public:
        explicit ActiveSet(std::vector<HighsBasisStatus>& col_status,
                           std::vector<HighsBasisStatus>& row_status){
            setActiveColRow(col_status, row_status); // construct basis by default. with hot starting may this be a problem?
        };

        void clear(){
            this->active_col_.clear();
            this->active_row_.clear();
            this->status_col_.clear();
            this->status_row_.clear();
        }
        // given a vector of statuses, extract whether active or not and the corresponding location index
        HighsInt setActive(const std::vector<HighsBasisStatus>& status,
                       std::vector<HighsInt>& index,
                       std::vector<HighsBasisStatus>& active_status){
            HighsInt count {0};
            for (size_t i{0}; i<status.size(); i++){
                if (status[i] == HighsBasisStatus::kLower ||
                    status[i] == HighsBasisStatus::kUpper ||
                    status[i] == HighsBasisStatus::kNonbasic){ // kNonbasic... does it ever happen after phase1?
                    // when the constraints and bounds in the problem are less than the number of variables, then there may be kNonbasic elements
                    // they are not in the simplex basis, yet we need them to construct the invertible matrix B = [A:V]
                    index.push_back(i);
                    active_status.push_back(status[i]);
                    count++;
                }
            }
            return count;
        }
        // define function to run after phase1 to extract columns and rows that are active
        void setActiveColRow(const std::vector<HighsBasisStatus>& col_status,
                          const std::vector<HighsBasisStatus>& row_status){
            clear();
            HighsInt count_col {0}, count_row{0};
            count_col = setActive(col_status, this->active_col_, this->status_col_);
            count_row = setActive(row_status, this->active_row_, this->status_row_);
            assert(count_col + count_row == col_status.size()); // check  that the number of active contraints equals the number of variables
        }

        void print(){
            std::cout<<"Active columns:\n";
            for (size_t i {0}; i<active_col_.size(); i++){
                std::cout<< active_col_[i] << " -- ";
            }
            std::cout<<"\nActive rows:\n";
            for (size_t i {0}; i<active_row_.size(); i++){
                std::cout<< active_row_[i] << " -- ";
            }
        }
    private:
        // define vectors for storing the indexes of the active columns and rows
        std::vector<HighsInt> active_col_;
        std::vector<HighsInt> active_row_;
        // define vectors for storing the status of the active columns and rows
        std::vector<HighsBasisStatus> status_col_;
        std::vector<HighsBasisStatus> status_row_;
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