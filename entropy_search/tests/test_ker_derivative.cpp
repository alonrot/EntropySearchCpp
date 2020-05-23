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

  // Gaussian process:
  // libgp::GaussianProcess * gp = new libgp::GaussianProcess(D, "CovSEard");
  // libgp::GaussianProcess * gp = new libgp::GaussianProcess(D, "CovSum ( CovSEard, CovNoise)");
  libgp::GaussianProcess * gp = new libgp::GaussianProcess(D, "CovSum ( CovMatern3iso, CovNoise)");
  // libgp::GaussianProcess * gp = new libgp::GaussianProcess(D, "CovSum ( CovMatern5iso, CovNoise)");
  // Computing the kernel derivative is only possible with CovSEard, CovNoise, and the case CovSum with these two kernels.
  
  // Initialize the hyperparameters:
  // Eigen::VectorXd params(D+2);
  // params << log(1.5), log(1.5), log(1.5), log(50), log(0.5); // lengthscale, signal std, noise std

  Eigen::VectorXd params(3);
  params << log(1.5), log(50), log(0.5); // lengthscale, signal std, noise std
  gp->covf().set_loghyper(params);

  // Domain boundaries:
  Eigen::VectorXd x_min = Eigen::VectorXd(D);
  Eigen::VectorXd x_max = Eigen::VectorXd(D);
  x_min << 0,0,0;
  x_max << 10,10,10;

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
	bfgs_optimizer::ObjectiveFunction * mu_and_dmu = new WrapperGP(gp,x_min,x_max);
	std::cout << "[DBG]: Initialization sucessful!" << std::endl;

  // Query point:
  Eigen::VectorXd x(D);
  x << 1.5,2.5,3.5;

 //  // Call the function wrapper:
 //  double f = 0.0; Eigen::VectorXd df(D); df.setZero();
	// mu_and_dmu->evaluate(x,&f,&df);

 //  // Verbosity:
 //  std::cout << std::endl << std::endl;
 //  std::cout << "Results for the GP mean derivative" << std::endl;
 //  std::cout << "==================================" << std::endl << std::endl;
	// std::cout << "df = " << std::endl << df << std::endl;
 //  std::cout << "This is the same result as in test_mean_derivative.m" << std::endl;
 //  std::cout << "Both, for k = CovSum ( CovSEard, CovNoise), and " << std::endl;
 //  std::cout << "          k = CovSEard" << std::endl;

  // Numerical solution:
  double step = 1e-6;
  Eigen::MatrixXd dkXxdx_num(D,N);
  Eigen::VectorXd x_p(D);
  Eigen::VectorXd x_n(D);
  Eigen::VectorXd kXx_p(N);
  Eigen::VectorXd kXx_n(N);
  for(size_t i=0;i<D;++i){
    x_p = x;
    x_n = x;
    x_p(i) = x(i) + step;
    x_n(i) = x(i) - step;
    gp->covf().getX(gp->get_sampleset(),x_p.transpose(),kXx_p); // kXx is here a column vector [N 1], because x is a single point
    gp->covf().getX(gp->get_sampleset(),x_n.transpose(),kXx_n); // kXx is here a column vector [N 1], because x is a single point
    dkXxdx_num.row(i).array() = (kXx_p - kXx_n).array() / (2*step);
  }

  std::cout << "dkXxdx_num = \n" << dkXxdx_num << std::endl;

  // Analytical solution:
  Eigen::VectorXd kXx_foo(N);
  Eigen::MatrixXd dkXxdx_ana(D,N);
  gp->covf().compute_dkdx(x,kXx_foo,gp->get_sampleset(),dkXxdx_ana);
  std::cout << "dkXxdx_ana = \n" << dkXxdx_ana << std::endl;

}