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
#include "CostFunction.hpp"

CostFunction::CostFunction(){};

double
ExampleCostFunction::get_value(Eigen::VectorXd x){

	Eigen::VectorXd x_rand(1);

	x_rand.setRandom();

	return x_rand(0);

}

CostFunctionFromPrior::CostFunctionFromPrior(size_t Dim, size_t Ndiv_plot, std::string path2evalfun){

	// Read file:
	YAML::Node node_to_read = YAML::LoadFile(path2evalfun);

	// Error checking:
	if( Dim != (size_t)node_to_read["Dim"].as<double>() )
		throw std::runtime_error("Evaluation function has a different dimension than the one passed");

	if( Ndiv_plot != (size_t)node_to_read["Ndiv_plot"].as<double>() )
		throw std::runtime_error("Evaluation function grid division is different than the one passed");

	// Parse data:
	Eigen::MatrixXd Y_aux = node_to_read["Y"].as<Eigen::MatrixXd>();
	this->Y = Y_aux.col(0);
	this->X = node_to_read["X"].as<Eigen::MatrixXd>();
	this->Ndiv_plot = Ndiv_plot;
	this->Dim = Dim;

}

double
CostFunctionFromPrior::get_value(const Eigen::VectorXd x) {

	size_t ind_up = 0;
	size_t ind_down = 0;
	double Yval = 0.0;

	// Special cases (TODO: do all the corners!)
	if(x.transpose().isApprox(this->X.row(0)))
		return this->Y(0);

	if(x.transpose().isApprox(this->X.row(this->Ndiv_plot-1)))
		return this->Y(this->Ndiv_plot-1);

	// Find the closest vector x within the domain this->X
	ind_up = MathTools::find_closest_in_grid(this->X, x);
	
	// Indices:
	ind_down = ind_up - 1;

	// Interpolate in 1D:
	if(this->Dim == 1)
		Yval = MathTools::interpolate1D(this->X(ind_down,0), this->X(ind_up,0), this->Y(ind_down), this->Y(ind_up), x(0));
	else
		Yval = this->Y(ind_down);
	
	return Yval;
}