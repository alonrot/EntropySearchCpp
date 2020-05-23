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
#ifndef INCLUDE_MATH_TOOLS_H
#define INCLUDE_MATH_TOOLS_H

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>

namespace MathTools {

	bool isNaN_vec(Eigen::VectorXd v);
	bool isNaN_mat(Eigen::MatrixXd M);
	bool isNaN_3Darray(std::vector<Eigen::MatrixXd> M, size_t D);
	bool isInf_vec(Eigen::VectorXd v);
	bool any_negative_vec(Eigen::VectorXd v);
	void sample_unif_vec(double xmin_s, double xmax_s, Eigen::Ref<Eigen::VectorXd> v);
	void sample_unif_mat(double xmin_s, double xmax_s, Eigen::Ref<Eigen::MatrixXd> M);
	bool outside_dom_vec(double xmin_s, double xmax_s, Eigen::VectorXd v);
	void project_to_boundaries(double xmin_s, double xmax_s, Eigen::Ref<Eigen::VectorXd> v);
	size_t find_closest_in_grid(const Eigen::MatrixXd X, const Eigen::VectorXd x);
	double interpolate1D(double x_low, double x_high, double y_low, double y_high, double x);


}


#endif // INCLUDE_MATH_TOOLS_H