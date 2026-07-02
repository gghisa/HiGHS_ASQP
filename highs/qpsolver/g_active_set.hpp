/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Highs.h"

class ActiveSet
{
    public:
        std::vector<HighsInt> setup(std::vector<HighsBasisStatus>& var_status,
                                    std::vector<HighsBasisStatus>& con_status){
            std::vector<HighsInt> basis;
            basis = setActiveVarCon(var_status, con_status); // construct basis by default. with hot starting may this be a problem?
            return basis;
        };

        void clear(){
            this->active_var_.clear();
            this->active_con_.clear();
            this->status_var_.clear();
            this->status_con_.clear();
        }
        // given a vector of statuses, extract whether active or not and the corresponding location index
        size_t setActive(const std::vector<HighsBasisStatus>& status,
                         std::vector<HighsInt>& index,
                         std::vector<HighsBasisStatus>& active_status,
                         const size_t offset){
            size_t count {0};
            for (size_t i {0}; i<status.size(); i++){
                if (status[i] == HighsBasisStatus::kLower ||
                    status[i] == HighsBasisStatus::kUpper ||
                    status[i] == HighsBasisStatus::kNonbasic){ // kNonbasic... does it ever happen after phase1?
                    // when the constraints and bounds in the problem are less than the number of variables, then there may be kNonbasic elements
                    // they are not in the simplex basis, yet we need them to construct the invertible matrix B = [A:V]
                    index.push_back(i + offset);
                    active_status.push_back(status[i]);
                    count++;
                }
            }
            return count;
        }
        // define function to run after phase1 to extract varumns and cons that are active
        std::vector<HighsInt> setActiveVarCon(const std::vector<HighsBasisStatus>& var_status,
                                              const std::vector<HighsBasisStatus>& con_status){
            clear();
            size_t count_var {0}, count_con {0};
            // variables' indexes start counting from m, which is the number of constraints. Constraint count starts from 0, as required by HFactor
            count_var = setActive(var_status, this->active_var_, this->status_var_, var_status.size());
            count_con = setActive(con_status, this->active_con_, this->status_con_, 0);
            assert(count_var + count_con == var_status.size()); // check  that the number of active contraints equals the number of variables
            std::vector<HighsInt> basis; // create vector for the basis required by HFactor
            basis.insert(basis.end(), this->active_con_.begin(), this->active_con_.end());
            basis.insert(basis.end(), this->active_var_.begin(), this->active_var_.end());
            return basis;
        }

        void print(){
            std::cout<<"\nActive constraints:\n";
            for (size_t i {0}; i<active_con_.size(); i++){
                std::cout<< active_con_[i] << " -- ";
            }
            std::cout<<"Active variables:\n";
            for (size_t i {0}; i<active_var_.size(); i++){
                std::cout<< active_var_[i] << " -- ";
            }
        }

    // need to add methods to (de)active a constraint and reshuffle them (or is permutation taken care of by the factorization routine?)
    private:
        // define vectors for storing the indexes of the active varumns and cons
        std::vector<HighsInt> active_var_;
        std::vector<HighsInt> active_con_;
        // define vectors for storing the status of the active varumns and cons
        std::vector<HighsBasisStatus> status_var_;
        std::vector<HighsBasisStatus> status_con_;
};