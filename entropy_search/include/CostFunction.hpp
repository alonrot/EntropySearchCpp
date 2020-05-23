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
#ifndef __COST_FUNCTION_H__
#define __COST_FUNCTION_H__

#include <Eigen/Dense>
#include <iostream>
#include "ReadYamlParameters.hpp"
#include "MathTools.hpp"

class CostFunction{

public:
	CostFunction();
	virtual ~CostFunction(){}
	virtual double get_value(const Eigen::VectorXd x) = 0;

};

class ExampleCostFunction : public CostFunction {

public:
	ExampleCostFunction(){};
	virtual ~ExampleCostFunction(){}
	double get_value(const Eigen::VectorXd x);
};

class CostFunctionFromPrior : public CostFunction {

  public:
    CostFunctionFromPrior(size_t Dim, size_t Ndiv_plot, std::string path2evalfun);
    ~CostFunctionFromPrior() {}
    double get_value(const Eigen::VectorXd x);

  private:
    Eigen::MatrixXd X;
    Eigen::VectorXd Y;
    size_t Ndiv_plot;
    size_t Dim;
};

#endif /* __COST_FUNCTION_H__ */