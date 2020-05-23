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
#include "EntropySearch.hpp"

int main (int argc, char const *argv[])
{
  // Declare the GP:
  libgp::GaussianProcess * gp;

  int D, param_dim, n_ll;
  int T;

  size_t Nb;
  size_t Nsub;
  size_t N;
  size_t N_div;
  size_t Nwarm_starts;

  Nb        = 50; 
  Nsub      = 5;
  D         = 3; 
  param_dim = D+2; // lengthscales + signal variance + measurement noise
  n_ll      = D;
  T         = 4;
  N_div     = 100;
  N         = 2;
  Nwarm_starts = 5;

  // Input dimension interval:
  double xmin_s =  0;
  double xmax_s =  10;

  // Initialize the GP:
  gp = new libgp::GaussianProcess(D, "CovSum ( CovSEard, CovNoise)");
  // Don't use "CovSum ( CovNoise, CovSEard )", as this will mess up with the order of the hyperparameters in the code.
  // TODO: automate this.

  // Initialize the hyperparameters:
  Eigen::VectorXd params(param_dim);
  params << log(1.5), log(1.5), log(1.5), log(50), log(0.5); // lengthscale, signal std, noise std
  // gp->covf().init(D); // internally done, not needed
  gp->covf().set_loghyper(params);

  Eigen::VectorXd pars_function(D);
  MathTools::sample_unif_vec(1,6,pars_function);
  bfgs_optimizer::ObjectiveFunction * function = new DummyFunctionChildBFGS(pars_function,xmin_s,xmax_s);

  Eigen::MatrixXd X(N,D);
  Eigen::VectorXd foo(D);
  Eigen::VectorXd Y(N);
  X << 7.0605, 4.3874, 2.7603,
       0.3183, 3.8156, 6.7970;

   for(size_t i=0;i<N;++i)
    function->evaluate(X.row(i),&Y(i),&foo);

  // // Data sampled uniformly within [0, 10]
  // double X[2][3] = { {7.0605, 4.3874, 2.7603},
  //                     {0.3183, 3.8156, 6.7970}};

  // // Function used: Y(i) = sum(X(i,:).^2)
  // double Y[] = { 76.7188, 60.8596 };

  // Add data to the GP:
  // double x_new[D];
  for(size_t i = 0; i < N; ++i) 
  {
    // for(int j = 0; j < D; ++j) 
    //   x_new[j] = X[i][j];
    
    // gp->add_pattern(x_new, Y[i]);
    gp->add_pattern(X.row(i).data(), Y(i));
  }

  // // zbel:
  // Eigen::MatrixXd zbel      = Eigen::MatrixXd(Nb,D);
  // zbel.setZero();

  // lower boundary:
  // Eigen::VectorXd xmin = Eigen::VectorXd::Ones(D);
  // xmin = xmin.array() * xmin_s;

  // upper boundary:
  // Eigen::VectorXd xmax = Eigen::VectorXd::Ones(D);
  // xmax = xmax.array() * xmax_s;

  //   // zbel as linspace (although in ES these are the representer points, randomly sampled);
  //   double dx = (x_max - x_min) / Nb;
  //   for(size_t i = 0; i < Nb; ++i) 
  //     zbel.row(i) = xmin.array() + i*dx;

  // // std::cout << "[DBG]: zbel.row(0) = " << zbel.row(0) << std::endl;
  // // std::cout << "[DBG]: zbel.row(Nb-1) = " << zbel.row(Nb-1) << std::endl;

  // // Obtain evaluations from the GP itself:
  // N = 10;
  // Eigen::MatrixXd X = Eigen::MatrixXd(N,D);
  // X.setRandom();
  // X = X.array() * (x_max - x_min)  + 2*x_min;
  // Eigen::VectorXd y = gp->covf().draw_random_sample(X);
  // double x[D];
  // for(size_t i = 0; i < N; ++i) 
  // {
  //   for(int j = 0; j < D; ++j) 
  //     x[j] = X(i,j);
    
  //   gp->add_pattern(x, y(i));
  // }

  // std::cout << "[DBG]: y.row(0) = " << y.row(0) << std::endl;
  // std::cout << "[DBG]: y.row(N-1) = " << y.row(N-1) << std::endl;

  // gp->add_pattern(X.row(i).data(), y(i));


  // Intialize input structure:
  INSetup * in = new INSetup;
  in->real_system = function;
  in->D  = D;
  in->T  = T;
  in->param_dim = param_dim;
  in->N  = N;
  in->Nb = Nb;
  in->Nsub = Nsub;
  in->Nwarm_starts = Nwarm_starts;
  in->gp = gp;
  in->LearnHypers = false;
  in->n_ll = n_ll;
  in->xmin_s = xmin_s;
  in->xmax_s = xmax_s;
  in->MaxEval = 5;
  in->N_div   = N_div;

  EntropySearch es(in);
  es.run();


}
