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
#include "MathTools.hpp"

bool
MathTools::isNaN_vec(Eigen::VectorXd v){

	size_t N = v.size();
	size_t i = 0;
	bool nan_found = false;

	while(i<N && !nan_found){

		if( std::isnan(v(i)) )
			nan_found = true;

		++i;

	}

	return nan_found;

}

bool
MathTools::isNaN_mat(Eigen::MatrixXd M){

	size_t N = M.cols();
	size_t i = 0;
	bool nan_found = false;

	while(i<N && !nan_found){

		if( MathTools::isNaN_vec(M.col(i)) )
			nan_found = true;

		++i;

	}

	return nan_found;

}

bool
MathTools::isNaN_3Darray(std::vector<Eigen::MatrixXd> M, size_t D){

	size_t i = 0;
	bool nan_found = false;

	while(i<D && !nan_found){

		if( MathTools::isNaN_mat(M[i]) )
			nan_found = true;

		++i;

	}

	return nan_found;

}

bool
MathTools::isInf_vec(Eigen::VectorXd v){

	size_t N = v.size();
	size_t i = 0;
	bool inf_found = false;

	while(i<N && !inf_found){

		if( v(i) == -INFINITY || v(i) == INFINITY )
			inf_found = true;

		++i;

	}

	return inf_found;

}

bool
MathTools::any_negative_vec(Eigen::VectorXd v){

	size_t N = v.size();
	size_t i = 0;
	bool nan_found = false;

	while(i<N && !nan_found){

		if( v(i) < 0 )
			nan_found = true;

		++i;

	}

	return nan_found;

}

void
MathTools::sample_unif_vec(double xmin_s, double xmax_s, Eigen::Ref<Eigen::VectorXd> v){
	
	v.setRandom();

	v = v.array() * (xmax_s - xmin_s) / 2;
	v = v.array() + (xmax_s + xmin_s) / 2;

	return;

}

void
MathTools::sample_unif_mat(double xmin_s, double xmax_s, Eigen::Ref<Eigen::MatrixXd> M){
	
	M.setRandom();

	M = M.array() * (xmax_s - xmin_s) / 2;
	M = M.array() + (xmax_s + xmin_s) / 2;

	return;

}

bool
MathTools::outside_dom_vec(double xmin_s, double xmax_s, Eigen::VectorXd v){

	if(v.minCoeff() < xmin_s || v.maxCoeff() > xmax_s)
		return true;

	return false;

}

void
MathTools::project_to_boundaries(double xmin_s, double xmax_s, Eigen::Ref<Eigen::VectorXd> v){

	size_t D = v.size();

	for(size_t i=0;i<D;++i){

		if(v(i)<xmin_s)
			v(i) = xmin_s;

		if(v(i)>xmax_s)
			v(i) = xmax_s;
	}

	return;

}

size_t 
MathTools::find_closest_in_grid(const Eigen::MatrixXd X, const Eigen::VectorXd x){

	// Get size:
	size_t Nrows = X.rows();
	size_t i = 0;
	size_t ind_up = 0;
	bool higher_bound_found = false;

	// Rest of the cases:
	while(i < Nrows && !higher_bound_found){
		
		// Extend this for 2D: use (a-b).isMuchSmallerThan(tol)
		if( (X.row(i).array()>=x.transpose().array()).all() ){
			
			higher_bound_found = true;
			ind_up = i;
		}
		
		++i;
	}

	return ind_up;

}

double
MathTools::interpolate1D(double x_low, double x_high, double y_low, double y_high, double x){

		// Interpolate:
		double alpha = (x - x_low) / (x_high - x_low);
		return y_low + ( y_high - y_low ) * alpha;

}
