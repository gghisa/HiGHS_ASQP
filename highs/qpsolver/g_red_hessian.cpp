/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "Highs.h"
#include "qpsolver/g_red_hessian.hpp"

ReducedHessian::ReducedHessian(HighsHessian& Q)
                                : Q_(Q){
};

void ReducedHessian::Hsetup(HighsSparseMatrix& constraint_mat, std::vector<HighsInt>& basis_idxs){
    this->B_.setup(constraint_mat, basis_idxs);
};

void ReducedHessian::Hbtran(std::vector<double>& vec){
    this->B_.btranCall(vec);
};

void ReducedHessian::Hftran(std::vector<double>& vec){
    this->B_.ftranCall(vec);
};

void ReducedHessian::Hbuild(){
    this->B_.build();
};

void ReducedHessian::extend(){
    // extend(std::vector<double>& y){
};
