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
#include <cmath>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include "DummyFunction.hpp"

// Slice sampler (not worth to make a class just for this):
extern Eigen::VectorXd SliceShrinkRank_nolog(Eigen::VectorXd V, bfgs_optimizer::ObjectiveFunction * fun, double s0);

int main (int argc, char const *argv[])
{

  // Initialization:
  int D = 3;
  double xmin_p = 0;
  double xmax_p = 2;

  Eigen::VectorXd param(D);
  Eigen::VectorXd xx_in  = Eigen::VectorXd::Zero(D);
  Eigen::VectorXd xx_out = Eigen::VectorXd::Zero(D);
  
  // Input dimension interval:
  double xmin_s =  0;
  double xmax_s =  10;

  // Domain boundaries:
  Eigen::VectorXd xmin = Eigen::VectorXd::Ones(D) * xmin_s;
  Eigen::VectorXd xmax = Eigen::VectorXd::Ones(D) * xmax_s;

  param = param.setRandom().array()*(xmax_p - xmin_p)/2;
  param = param.array() + (xmax_p + xmin_p)/2;
  std::cout << "param = " << std::endl << param << std::endl;

  // Input function:
  bfgs_optimizer::ObjectiveFunction * dummy_fun = new DummyFunctionChildBFGS(param,xmin_s,xmax_s);

  // Initial x:
  xx_in << 1,4,2.5;

  // Call the slicer:
  double S0 = 0.5 * (xmax - xmin).norm();
  xx_out = SliceShrinkRank_nolog(xx_in,dummy_fun,S0);

  std::cout << std::endl;
  std::cout << "Results" << std::endl;
  std::cout << "=======" << std::endl;
  std::cout << "xx_out = " << std::endl << xx_out << std::endl;

}
