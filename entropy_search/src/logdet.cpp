// Copyright 2020 Max Planck Society. All rights reserved.
// 
// Author: Alonso Marco Valle (amarcovalle/alonrot) amarco(at)tuebingen.mpg.de
// Affiliation: Max Planck Institute for Intelligent Systems, Autonomous Motion
// Department / Intelligent Control Systems
// 
// This file is part of EntropySearchCpp.
// 
// EntropySearchCpp is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version.
// 
// EntropySearchCpp is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
// details.
// 
// You should have received a copy of the GNU General Public License along with
// EntropySearchCpp.  If not, see <http://www.gnu.org/licenses/>.
//
//
#include <Eigen/Dense>

/**
 * Returns the logarithm of determinant of a positive definite matrix
 * 
 * @param M positive definite square matrix
 * @return logarithm of determinant
 */
double logdet(Eigen::MatrixXd M) {

  Eigen::LLT<Eigen::MatrixXd> llt(M);

  // D = 2 * sum(log(diag(chol(M))));

  Eigen::VectorXd res = ((Eigen::MatrixXd) llt.matrixU()) // Calc chol())
          .diagonal().array().log() // Log of diagonal
          .colwise().sum(); // Calc sum of logs

  return 2 * res[0];
}
