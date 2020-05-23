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
#include "gp.h"
#include <Eigen/Dense>
#include <memory>
#include "CostFunction.hpp"

typedef struct{

	// Number of input dimensions:
	int 		Dim;

	int T;
	// Number of representer points:
	size_t 	Nrepresenters;
	// Number of subsamples:
	size_t 	Nsubsamples;
	// Number of warm starts:
	size_t Nwarm_starts;
	// Number of divisions for each dimension, for getting the grid of test points (plotting):
	size_t  Ndiv_plot;
	// Number of line searches to the local optimizer:
	size_t Nline_searches;

	// Gaussian Process:
	libgp::GaussianProcess * gp;

	// Learn the hyperparameters:
	bool LearnHypers;

	// Number of lengthscales:
	size_t Nll;

	// // Box boundaries:
	double xmin_s;
	double xmax_s;

	// Maximum number of evaluations:
	int MaxEval;

	// Debugging variables:
	bool write_matrices_to_file;
	bool read_for_test_dH_MC_local_flag;
	bool read_for_test_SampleBeliefLocations_flag;
	bool write2file;

	// Plot with python:
	bool write2pyplot;
	bool plot_true_function;
	std::string path2data_logging_absolute;

	// Vector of hyperparameters:
	std::vector<Eigen::VectorXd> hyperparam_per_ker;

	// Evaluation function:
	std::shared_ptr<CostFunction> cost_function;

}INSetup;

typedef struct{

	// Input structure:
	std::shared_ptr<INSetup> in;

	// Results:
	Eigen::MatrixXd global_min_esti_x;
	Eigen::VectorXd global_min_esti_mux;
	Eigen::VectorXd global_min_esti_varx;

}OutResults;

