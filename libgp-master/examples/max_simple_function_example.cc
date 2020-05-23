// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "simple_fun_wrapper.h"

int main (int argc, char const *argv[])
{

  // Polinomial function definition:
  Eigen::VectorXd param = Eigen::VectorXd(3);
  param(0) = -5.0;
  param(1) = -2.0;
  param(2) =  1.0;

  // x^2*param(2) + x^1*param(1) + param(0)

  // Define an object from the class SimpleFunction
  SimpleFunction * f;
  f = new SimpleFunction(param);

  // Wrapper to a generic class that inherits from
  // ObjectiveFunction class, that contains some virtual members,
  // and is the one that the bfgs_optimizer::BFGS class understands.
  SimpleFunWrapper fw(f);

  // Call to the optimizer class. It is constructed with an object
  // from ObjectiveFunction.
  bfgs_optimizer::BFGS bfgs(&fw,1);

  // Initial guess:
  Eigen::VectorXd x_init  = Eigen::VectorXd(1);
  Eigen::VectorXd x_final = Eigen::VectorXd(1);

  x_init  << 2;
  x_final << 0;

  x_final = bfgs.minimize(x_init);

  std::cout << "x_init = " << x_init << std::endl;
  std::cout << "x_final = " << x_final << std::endl;

}
