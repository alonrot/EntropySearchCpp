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
#include "TestPolymorphismm.hpp"

int main (int argc, char const *argv[])
{

	int D = 1;
	Eigen::VectorXd param = Eigen::VectorXd(D);
	param(0) = 1.6;

  // Input dimension interval:
  double xmin_s =  0;
  double xmax_s =  10;

  // Domain boundaries:
  Eigen::VectorXd xmin = Eigen::VectorXd::Ones(D) * xmin_s;
  Eigen::VectorXd xmax = Eigen::VectorXd::Ones(D) * xmax_s;

	DummyFunction * my_fun = new DummyFunctionChild(param);
	std::cout << "Function name: " << my_fun->function_name() << std::endl;
	param(0) = 3;

	my_fun->update_something(param);

	TestPolymorphismm test_polymorphism;

	bfgs_optimizer::ObjectiveFunction * gp_sample = new GPsample(0,1,D,200);


	test_polymorphism.enter_function(my_fun);
	test_polymorphism.enter_function(gp_sample);

	// some_test.acquire_function_dummyfun(my_fun);
	// some_test.acquire_function_dummyfun(gp_sample);


	// bfgs_optimizer::ObjectiveFunction * my_fun_bfgs = new DummyFunctionChildBFGS(param,xmin_s,xmax_s);
	// // DummyFunctionChildBFGS * my_fun_bfgs = new DummyFunctionChildBFGS(param);

	// Eigen::VectorXd x = Eigen::VectorXd(3);
	// x(0) = 0.0;
	// x(1) = 3.0;
	// x(2) = 0.0;

	// double f = 0.0;
	// Eigen::VectorXd df = Eigen::VectorXd(3);

	// my_fun->evaluate_function(x,f,df);
	// std::cout << "f = " << f << std::endl;
	// std::cout << "df = " << df << std::endl;

	// my_fun_bfgs->evaluate(x,&f,&df);
	// std::cout << "f = " << f << std::endl;
	// std::cout << "df = " << df << std::endl;

	// // Call to GP Wrapper:
 //  // Input dimension interval:
 //  double x_min =  0;
 //  double x_max =  10;
 //  Eigen::VectorXd x_min_vec(D);
 //  Eigen::VectorXd x_max_vec(D);
 //  x_min_vec << x_min,x_min,x_min;
 //  x_max_vec << x_max,x_max,x_max;
 //  int N = 10;

	// libgp::GaussianProcess * gp = new libgp::GaussianProcess(D, "CovSEard");
 //  // Initialize the hyperparameters:
 //  gp->covf().init(D);
 //  Eigen::VectorXd params(D+1);
 //  params << log(1.5), log(1.5), log(1.5), log(0.01); // lengthscale, signal std, noise std
 //  gp->covf().set_loghyper(params);
 //  // Obtain evaluations from the GP itself:
 //  Eigen::MatrixXd X = Eigen::MatrixXd(N,D);
 //  MathTools::sample_unif_mat(xmin_s,xmax_s,X);
 //  Eigen::VectorXd y = gp->covf().draw_random_sample(X);
 //  for(size_t i = 0; i < N; ++i) 
 //    gp->add_pattern(X.row(i).data(), y(i));

	// bfgs_optimizer::ObjectiveFunction * mu_and_dmu = new WrapperGP(gp,xmin,xmax);
	// std::cout << "[DBG]: Initialization sucessful!" << std::endl;


	// mu_and_dmu->evaluate(x,&f,&df);
	// std::cout << "f = " << f << std::endl;
	// std::cout << "df = " << df << std::endl;

}