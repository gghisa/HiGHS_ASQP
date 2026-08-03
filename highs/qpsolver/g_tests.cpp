/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Highs.h"
#include "qpsolver/g_tests.hpp"

double innerprod(const std::vector<double>& vec1, const std::vector<double> vec2){
    double sum {0};
    if (vec1.size() != vec2.size()) throw std::logic_error("Vectors must have the same size!");
    for (size_t i {0}; i < vec1.size(); i++){
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

bool AsmSolver::testOrtho(){
    for (HighsInt i {0}; i < this->rangsp_dim_; i++){ // for all active constraints
        HighsInt idx {this->basis_idxs_[i]};
        if (idx <this->lp_.num_row_){ // if we are looking at a constraint
            std::vector<double> ep(this->lp_.num_row_);
            ep[idx] = 1.;
            std::vector<double> constraint(this->lp_.num_col_);
            this->lp_.a_matrix_.productTranspose(constraint, ep); // extract constraint
            for (HighsInt j {0}; j < this->nullsp_dim_; j++){
                if (innerprod(this->ZT_[j], constraint) != 0){ // check that it is (not) orthogonal to free dimension
                    return false;
                }
            }
        } else { // if we are looking at a variable bound
            idx -= this->lp_.num_row_; // get index of variable
            for (HighsInt j {0}; j < this->nullsp_dim_; j++){
                if (this->ZT_[j][idx] != 0){ // check that the free dimension doesnt change the active variable
                    return false;
                }
            }
        }
    }
    return true;
}

bool AsmSolver::testYTAid(){
    for (HighsInt i {0}; i < this->rangsp_dim_; i++){ // for all active constraints
        HighsInt idx {this->basis_idxs_[i]};
        if (idx <this->lp_.num_row_){ // if we are looking at a constraint
            std::vector<double> ep(this->lp_.num_row_);
            ep[idx] = 1.;
            std::vector<double> constraint(this->lp_.num_col_);
            this->lp_.a_matrix_.productTranspose(constraint, ep); // extract constraint
            HFtran(constraint); // compute B^{-1} \vec
            for (HighsInt j {0}; j <this->Q_.dim_; j++){
                if (j == i && constraint[j] != 1.){
                    return false;
                } else if (j != i && constraint[j] != 0.){
                    return false;
                }
            }
        } else { // if we are looking at a variable bound
            idx -= this->lp_.num_row_; // get index of variable
            std::vector<double> ep(this->lp_.num_row_);
            ep[idx] = 1.; // build "constraint"
            HFtran(ep); // compute B^{-1} \vec 
            for (HighsInt j {0}; j <this->Q_.dim_; j++){
                if (j == i && ep[j] != 1.){
                    return false;
                } else if (j != i && ep[j] != 0.){
                    return false;
                }
            }
        }
    }
    return true;
}