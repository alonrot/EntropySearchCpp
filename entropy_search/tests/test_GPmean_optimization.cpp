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
#include "DummyFunction.hpp"

int main (int argc, char const *argv[])
{

	// Initialization:
	int D = 3; int N = 10;
  Eigen::VectorXd x_init_gp = Eigen::VectorXd(D);
  Eigen::VectorXd x_final_gp = Eigen::VectorXd(D);
  Eigen::VectorXd x_min = Eigen::VectorXd(D);
  Eigen::VectorXd x_max = Eigen::VectorXd(D);
  x_min << 0,0,0;
  x_max << 10,10,10;

  // Gaussian process:
  // libgp::GaussianProcess * gp = new libgp::GaussianProcess(D, "CovSEard");
  libgp::GaussianProcess * gp = new libgp::GaussianProcess(D, "CovSum ( CovSEard, CovNoise)");
  // Computing the kernel derivative is only possible with CovSEard, CovNoise, and the case CovSum with these two kernels.
  
  // Initialize the hyperparameters:
  Eigen::VectorXd params(D+2);
  params << log(1.5), log(1.5), log(1.5), log(50), log(0.5); // lengthscale, signal std, noise std
  gp->covf().set_loghyper(params);

  // Data sampled uniformly within [0, 10]
  double X[10][3] = { {7.0605, 4.3874, 2.7603},
                      {0.3183, 3.8156, 6.7970},
                      {2.7692, 7.6552, 6.5510},
                      {0.4617, 7.9520, 1.6261},
                      {0.9713, 1.8687, 1.1900},
                      {8.2346, 4.8976, 4.9836},
                      {6.9483, 4.4559, 9.5974},
                      {3.1710, 6.4631, 3.4039},
                      {9.5022, 7.0936, 5.8527},
                      {0.3445, 7.5469, 2.2381}};

  // Function used: Y(i) = sum(X(i,:).^2)
  double Y[] = { 76.7188, 60.8596, 109.1856, 66.0917, 5.8516, 116.6319, 160.2442, 63.4135, 174.8659, 62.0830};

  // Add data to the GP:
  double x_new[D];
  for(size_t i = 0; i < N; ++i) 
  {
    for(int j = 0; j < D; ++j) 
      x_new[j] = X[i][j];
    
    gp->add_pattern(x_new, Y[i]);
  }

  // GP function wrapper:
  bfgs_optimizer::ObjectiveFunction * GPmean = new WrapperGP(gp,x_min,x_max);
  std::cout << "[DBG]: Initialization sucessful!" << std::endl;

  // Construct local optimizer:
  bfgs_optimizer::BFGS bfgs_GP(GPmean,100); // We need bfgs_optimizer::ObjectiveFunction

  // Initial point:
  x_init_gp << 1.5,2.5,3.5;
  // x_init_gp << 9,9,1;

  // Call optimizer:
  x_final_gp = bfgs_GP.minimize(x_init_gp);

  // Verbosity:
  std::cout << std::endl << std::endl;
  std::cout << "Optimization of the posterior GP mean" << std::endl;
  std::cout << "=====================================" << std::endl << std::endl;
	std::cout << "x_init_gp = " << std::endl << x_init_gp << std::endl;
  std::cout << "x_final_gp = " << std::endl << x_final_gp << std::endl;

}