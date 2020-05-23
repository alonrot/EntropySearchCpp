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
#include "TestPolymorphism.hpp"

void
TestPolymorphism::enter_function(bfgs_optimizer::ObjectiveFunction * fun){
	std::cout << "@TestPolymorphism::enter_function" << std::endl;

	// Call some of its members:
	Eigen::VectorXd x(1);
	x(0) = 0.5;
	double fx = 0.0;
	Eigen::VectorXd dfx(1);
	dfx(0) = 0.0;

	// Call a method implemented by the base class bfgs_optimizer::ObjectiveFunction:
	fun->evaluate(x,&fx,&dfx);
  std::cout << "    x = [ " << x << " ]" << std::endl;
  std::cout << "    y = f(x) = " << fx << std::endl;
  std::cout << "    d/dx{f}(x) = " << dfx << std::endl;

	// Call a method implemented by the base class DummyFunction:
	// std::cout << "Function name:" << fun->function_name() << std::endl; // This doesn't compile, as expected

}