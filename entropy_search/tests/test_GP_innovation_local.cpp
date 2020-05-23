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
#include "GP_innovation_local.hpp"

int main (int argc, char const *argv[])
{
  // Initialization:
  int D = 3; int N = 2;
  int Nb = 50;
  double xmin_s = 0;
  double xmax_s = 10;
  Eigen::VectorXd x_init_gp = Eigen::VectorXd(D);
  Eigen::VectorXd x_final_gp = Eigen::VectorXd(D);
  Eigen::VectorXd x_min = Eigen::VectorXd::Ones(D);
  Eigen::VectorXd x_max = Eigen::VectorXd::Ones(D);
  x_min = x_min * xmin_s;
  x_max = x_max * xmax_s;

  // Gaussian process:
  // libgp::GaussianProcess * gp = new libgp::GaussianProcess(D, "CovSEard");
  libgp::GaussianProcess * gp = new libgp::GaussianProcess(D, "CovSum ( CovSEard, CovNoise)");
  libgp::GaussianProcess * gp_noise = new libgp::GaussianProcess(D, "CovNoise");
  Eigen::VectorXd params_noise(1);
  params_noise << log(0.5);
  gp_noise->covf().set_loghyper(params_noise);
  // Computing the kernel derivative is only possible with CovSEard, CovNoise, and the case CovSum with these two kernels.
  
  // Initialize the hyperparameters:
  Eigen::VectorXd params(D+2);
  params << log(1.5), log(1.5), log(1.5), log(50), log(0.5); // lengthscale, signal std, noise std
  gp->covf().set_loghyper(params);

  // Data sampled uniformly within [0, 10]
  double X[10][3] = { {7.0605, 4.3874, 2.7603},
                      {0.3183, 3.8156, 6.7970} };
                      // {2.7692, 7.6552, 6.5510},
                      // {0.4617, 7.9520, 1.6261},
                      // {0.9713, 1.8687, 1.1900},
                      // {8.2346, 4.8976, 4.9836},
                      // {6.9483, 4.4559, 9.5974},
                      // {3.1710, 6.4631, 3.4039},
                      // {9.5022, 7.0936, 5.8527},
                      // {0.3445, 7.5469, 2.2381}};

  // Function used: Y(i) = sum(X(i,:).^2)
  double Y[] = { 76.7188, 60.8596 }; //, 109.1856, 66.0917, 5.8516, 116.6319, 160.2442, 63.4135, 174.8659, 62.0830};

  // Add data to the GP:
  double x_new[D];
  for(size_t i = 0; i < N; ++i) 
  {
    for(int j = 0; j < D; ++j) 
      x_new[j] = X[i][j];
    
    gp->add_pattern(x_new, Y[i]);
    gp_noise->add_pattern(x_new, Y[i]);
  }


  // Define zbel:
  double zbel_vec[] = {5.5273,2.7481,2.4150,2.4315,1.5416,9.5642,9.3566,8.1871,7.2826,1.7581,3.6037,1.8879,0.0120,3.1642,6.9962,6.2526,5.4306,4.3904,2.8743,5.0166,7.6155,7.6241,5.7606,7.4766,6.4553,1.2322,5.0440,3.4726,0.9215,1.4785,1.9817,6.7227,4.3151,6.9440,2.5678,0.0976,5.3228,2.7939,9.4623,9.0644,3.9268,0.2486,6.7144,8.3717,9.7150,0.5693,4.5032,5.8247,6.8664,7.1943,6.5004,7.2691,3.7385,5.8158,1.1612,0.5765,9.7977,2.8482,5.9497,9.6216,1.8578,1.9304,3.4164,9.3290,3.9067,2.7322,1.5195,3.9711,3.7472,1.3111,4.3504,0.9151,6.1463,0.1098,5.7326,7.8973,2.3537,4.4802,5.6936,0.6140,4.9629,6.4232,2.2127,8.3706,9.7108,8.4637,5.0600,2.7888,7.4662,2.3693,9.5735,6.2026,6.0026,1.7260,0.9035,2.5526,8.5857,9.1107,6.9963,7.2518,2.2989,5.7605,8.1063,4.0384,9.8844,0.9000,3.2094,5.1141,0.6061,7.2569,5.5656,5.2936,8.2998,8.5876,7.8903,3.1783,4.5221,7.5223,1.0986,1.0974,2.6988,5.2464,9.7265,7.1041,3.1186,2.9146,8.5036,9.1165,6.3928,2.5537,0.8867,8.3826,5.8472,9.4811,0.6103,5.8464,2.8511,8.2773,1.9099,4.4253,3.9341,8.2657,6.7687,2.0760,3.1810,1.3381,6.7146,5.7099,1.6977,1.4766};
  Eigen::Map<Eigen::MatrixXd> zbel(zbel_vec,Nb,D);

  // Construct the GP innovation class:
  std::cout << "[DBG]: Construction of GPInnovationLocal" << std::endl;
  GPInnovationLocal dGP(gp,zbel);
  std::cout << "[DBG]: After construction of GPInnovationLocal" << std::endl;

  Eigen::VectorXd Lx(Nb);
  Eigen::MatrixXd dLxdx(Nb,D);

  Lx.setZero();
  dLxdx.setZero();

  // Reboot seed:
  // srand((unsigned int) time(0));
  sranddev();

  // Where to evaluate:
  Eigen::VectorXd x_eval(D);
  x_eval.setRandom().array() * (xmax_s - xmin_s) / 2;
  x_eval = x_eval.array() + (xmax_s + xmin_s) / 2;
  std::cout << "x_eval = " << std::endl << x_eval.transpose() << std::endl;

  // Little analysis about kbx_noise:
  // Eigen::MatrixXd kbx_noise(Nb,Nb);
  // (gp_noise->covf()).get(zbel, zbel, kbx_noise);
  // std::cout << "kbx_noise = " << std::endl << kbx_noise << std::endl;

  dGP.efficient_innovation(x_eval,Lx,dLxdx);
  std::cout << "Lx = " << std::endl << Lx.transpose() << std::endl;

  // Update variables:
  MathTools::sample_unif_vec(xmin_s,xmax_s,zbel.col(1)); // Replace second column by random numbers
  Eigen::VectorXd x_more(D);
  x_more << 4,7,9.5;
  double Y_more = 60;
  gp->add_pattern(x_more.data(), Y_more);

  dGP.update_variables(gp,zbel);

  dGP.efficient_innovation(x_eval,Lx,dLxdx);
  std::cout << "Lx = " << std::endl << Lx.transpose() << std::endl;


}
