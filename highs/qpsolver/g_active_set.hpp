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

    // need to add methods to (de)active a constraint and reshuffle them (or is permutation taken care of by the factorization routine?)
    private:
        // define vectors for storing the indexes of the active columns and rows
        std::vector<HighsInt> active_col_;
        std::vector<HighsInt> active_row_;
        // define vectors for storing the status of the active columns and rows
        std::vector<HighsBasisStatus> status_col_;
        std::vector<HighsBasisStatus> status_row_;
};