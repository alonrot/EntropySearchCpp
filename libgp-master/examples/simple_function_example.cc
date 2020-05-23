// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "simple_function.h"

int main (int argc, char const *argv[])
{

  // Polinomial function definition:
  Eigen::VectorXd param = Eigen::VectorXd(3);
  param(0) = -1.0;
  param(1) = 0.0;
  param(2) = -1.0;

  double y, yd, ydd, x;

  // Define an object from the class SimpleFunction
  SimpleFunction f(param);

  // Define location x:
  x = 3;

  // Evaluate at location x:
  y   = f.get_value(x);
  yd  = f.get_gradient(x);
  ydd = f.get_hessian(x);

  // Print on screen:
  std::cout << "f(" << x << ")   = " << y << std::endl;

  std::cout << "f'(" << x << ")  = " << yd << std::endl;

  std::cout << "f''(" << x << ") = " << ydd << std::endl;

}
