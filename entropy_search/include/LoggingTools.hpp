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
#ifndef INCLUDE_LOGGING_TOOLS_H
#define INCLUDE_LOGGING_TOOLS_H

#include <Eigen/Dense>
#include <fstream>
#include "DummyFunction.hpp"
#include "gp.h"
#include "ReadYamlParameters.hpp"
#include "CostFunction.hpp"

namespace LoggingTools {

	void write2file_mat(std::ofstream * file, Eigen::MatrixXd M, std::string var_name);
	void write2file_vec(std::ofstream * file, Eigen::VectorXd M, std::string var_name);
	void write2file_sca(std::ofstream * file, double s, std::string var_name);
	void log_variables(size_t D, double xmin_s, double xmax_s, size_t numiter, size_t ind_min, size_t Ndiv,
										libgp::GaussianProcess * gp, Eigen::VectorXd dH_plot, double EdH_max, 
										std::shared_ptr<CostFunction> cost_function,
										bool plot_true_function,
										Eigen::VectorXd Mb, Eigen::MatrixXd Vb, Eigen::MatrixXd zb,
										Eigen::VectorXd x_most_informative, Eigen::MatrixXd global_min_esti_x,
										Eigen::VectorXd global_min_esti_mux, Eigen::VectorXd global_min_esti_varx);
	void read_for_test_dH_MC_local(	const std::string path2read,
																	const size_t Dim,
																	Eigen::MatrixXd & zb_mat,
	                                Eigen::VectorXd & lmb_mat,
	                                Eigen::VectorXd & logP_mat,
	                                Eigen::MatrixXd & dlogPdM_mat,
	                                Eigen::MatrixXd & dlogPdV_mat,
	                                std::vector<Eigen::MatrixXd> & ddlogPdMdM_mat,
	                                Eigen::VectorXd & Mb_mat,
	                                Eigen::MatrixXd & Vb_mat,
	                                Eigen::VectorXd & x_most_informative_mat,
	                                double & y_new_mat,
	                                double & EdH_max_mat);
	void read_for_test_SampleBeliefLocations(	const std::string path2read,
																						const size_t Dim,
																						Eigen::MatrixXd & zb_mat,
						                                Eigen::VectorXd & lmb_mat,
						                                Eigen::VectorXd & x_most_informative_mat,
						                                double & EdH_max_mat);

	void read_for_test_SampleBeliefLocations(	const std::string path2read,
																						const size_t Dim,
																						Eigen::MatrixXd & zb_mat,
						                                Eigen::VectorXd & lmb_mat,
						                                Eigen::VectorXd & Mb,
																						Eigen::MatrixXd & Vb,
						                                Eigen::VectorXd & x_most_informative_mat,
						                                double & y_new_mat,
						                                double & EdH_max_mat);

	void write_for_pyplot(	std::string path2store,
														size_t Dim,
														size_t numiter,
														bool plot_true_function,
														size_t Ndiv_plot,
														libgp::GaussianProcess * gp,
														std::shared_ptr<CostFunction> cost_function,
														Eigen::MatrixXd z_plot,
														Eigen::VectorXd dH_plot,
														double EdH_max,
														Eigen::VectorXd x_next);

void write_progress(std::string path2store,
										size_t Dim,
										libgp::GaussianProcess * gp,
										Eigen::VectorXd global_min_esti_x,	
										Eigen::VectorXd global_min_esti_mux,
										Eigen::VectorXd global_min_esti_varx);


}

#endif // INCLUDE_LOGGING_TOOLS_H