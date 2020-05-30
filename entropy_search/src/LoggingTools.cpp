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
#include "LoggingTools.hpp"

void
LoggingTools::write2file_mat(std::ofstream * file, Eigen::MatrixXd M, std::string var_name) {

  // std::ofstream file(file_name);
  if (file->is_open()) {
  	(*file) << var_name << " = [" << "\n";
    (*file) << M << '\n';
    (*file) << "];" << "\n";
  }

  return;

}

void
LoggingTools::write2file_vec(std::ofstream * file, Eigen::VectorXd v, std::string var_name) {

  // std::ofstream file(file_name);
  if (file->is_open()) {
  	(*file) << var_name << " = [ ";
    (*file) << v.transpose();
    (*file) << " ]';" << "\n";
  }

  return;

}

void
LoggingTools::write2file_sca(std::ofstream * file, double s, std::string var_name) {

  // std::ofstream file(file_name);
  if (file->is_open()) {
  	(*file) << var_name << " = " << s << ";\n";
  }

  return;

}

void
LoggingTools::log_variables(size_t D, double xmin_s, double xmax_s, size_t numiter, size_t ind_min, size_t Ndiv,
														libgp::GaussianProcess * gp, Eigen::VectorXd dH_plot, double EdH_max, 
														std::shared_ptr<CostFunction> cost_function,
														bool plot_true_function,
														Eigen::VectorXd Mb, Eigen::MatrixXd Vb, Eigen::MatrixXd zb,
														Eigen::VectorXd x_most_informative, Eigen::MatrixXd global_min_esti_x,
														Eigen::VectorXd global_min_esti_mux, Eigen::VectorXd global_min_esti_varx){

	if(D > 2)
		throw std::runtime_error("@LoggingTools::log_variable not prepared for D > 2");

	std::cout << "    Logging out variables..." << std::endl;
	std::string file_name;
	std::string path2store = "/Users/alonrot/MPI/projects_WIP/userES/data_from_cpp";
	file_name = path2store + "/ES_iter_" + std::to_string(numiter+1) + ".m";
	std::ofstream * file = new std::ofstream(file_name);
	size_t Nel = std::pow(Ndiv,D);
	Eigen::VectorXd mpost = Eigen::VectorXd::Zero(Nel);
	// Eigen::VectorXd dH_plot = Eigen::VectorXd::Zero(Nel);
	Eigen::VectorXd stdpost = Eigen::VectorXd::Zero(Nel);
	Eigen::VectorXd z_plot_single = Eigen::VectorXd::Zero(Ndiv);
	Eigen::MatrixXd z_plot = Eigen::MatrixXd::Zero(Nel,D);
	Eigen::VectorXd f_true = Eigen::VectorXd::Zero(Nel);
	Eigen::VectorXd z_plot_i = Eigen::VectorXd::Zero(D);
	Eigen::VectorXd foo = Eigen::VectorXd::Zero(D);

	// Scalars:
	// double x_next = x_most_informative(0);
	// std::cout << "@log_variables() [DBG1]: x_most_informative = " << x_most_informative << std::endl;
	// std::cout << "@log_variables() [DBG1]: x_next = " << x_next << std::endl;
	// double EdH_max;
	// std::cout << "@log_variables() [DBG1]: EdH_max = " << EdH_max << std::endl;
	// EdH_max = -EdH_max;
	// std::cout << "@log_variables() [DBG2]: EdH_max = " << EdH_max << std::endl;
	Eigen::VectorXd x_bg = global_min_esti_x.row(numiter);
	double mu_bg = global_min_esti_mux(numiter);
	double var_bg = global_min_esti_varx(numiter);
	double mu_x_next = gp->f(x_most_informative.data());

	// Grid the space:
	double dz = (xmax_s - xmin_s)/(Ndiv-1);
	double var;
	for(size_t i=0;i<Ndiv;++i){
		z_plot_single(i) = xmin_s + dz * i;
	}

	if(D == 1){
		z_plot = z_plot_single;
	}
	else if(D == 2){
		for(size_t i=0;i<Ndiv;++i){
			z_plot.block(i*Ndiv,0,Ndiv,1) = Eigen::VectorXd::Ones(Ndiv)*z_plot_single(i);
			z_plot.block(i*Ndiv,1,Ndiv,1) = z_plot_single;
		}
	}

	// Export true function:
	for(size_t i=0;i<Nel;++i){

		z_plot_i = z_plot.row(i);

		// GP posterior mean and std:
		mpost(i) = gp->f(z_plot_i.data());
		var = gp->var(z_plot_i.data());
		if(var < 0.0)
			var = 0.0;
		stdpost(i) = std::sqrt(var);

		if(plot_true_function)
			f_true(i) = cost_function->get_value(z_plot_i);

		// EdH_objective->evaluate(z_plot_i,&dH_plot(i),&foo);
	}

	// Flip sign:
	// dH_plot.array() = -dH_plot.array();

	// std::cout << "z_plot.block(40,0,20,2).transpose() = " << std::endl << z_plot.block(40,0,20,2).transpose() << std::endl; 
	// std::cout << "mpost.segment(40,20) = " << mpost.segment(40,20).transpose() << std::endl;



			// std::cout << "@log_variables()1 z_plot = " << z_plot.transpose() << std::endl;
			// std::cout << "@log_variables()1 dH_plot = " << dH_plot.transpose() << std::endl;
	// for(size_t i=0;i<Ndiv;++i){
	// 	z_plot(i) = xmin_s + dz * i;
	// 	z_plot_i(0) = z_plot(i);
	// 	EdH_objective->evaluate(z_plot_i,&dH_plot(i),&foo);
	// 	dH_plot(i) = -dH_plot(i);
	// }
	// 		std::cout << "@log_variables()2 z_plot = " << z_plot.head(8).transpose() << std::endl;
	// 		std::cout << "@log_variables()2 dH_plot = " << dH_plot.head(8).transpose() << std::endl;


	// Sample set:
	size_t Nset = gp->get_sampleset_size();
	Eigen::MatrixXd X_set(Nset,D);
	Eigen::VectorXd Y_set(Nset);
	for(size_t i=0;i<Nset;++i){
		X_set.row(i) = gp->get_sampleset()->x(i).transpose();
		Y_set(i) = gp->get_sampleset()->y(i);
	}

	// Write to file:
	LoggingTools::write2file_mat(file,z_plot,"z_plot");
	LoggingTools::write2file_vec(file,f_true,"f_true");
	LoggingTools::write2file_vec(file,mpost,"mpost");
	LoggingTools::write2file_vec(file,stdpost,"stdpost");
	LoggingTools::write2file_vec(file,x_most_informative,"x_next");
	LoggingTools::write2file_sca(file,mu_x_next,"mu_x_next");
	LoggingTools::write2file_sca(file,EdH_max,"EdH_max");
	LoggingTools::write2file_vec(file,dH_plot,"dH_plot");
	LoggingTools::write2file_mat(file,X_set,"x");
	LoggingTools::write2file_vec(file,Y_set,"y");
	LoggingTools::write2file_vec(file,Mb,"Mb");
	LoggingTools::write2file_mat(file,Vb,"Vb");
	LoggingTools::write2file_mat(file,zb,"zb");
	LoggingTools::write2file_vec(file,x_bg,"x_bg");
	LoggingTools::write2file_sca(file,mu_bg,"mu_bg");
	LoggingTools::write2file_sca(file,var_bg,"var_bg");
	file->close();

}

void
LoggingTools::read_for_test_dH_MC_local(const std::string path2read,
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
				                                double & EdH_max_mat){

	if(Dim > 2)
		throw std::runtime_error("@LoggingTools::read_for_test_dH_MC_local() should not be called");

	std::cout << "[WARNING]: Replacing variables by some others read from a .yaml file!!!!" << std::endl;

	// Load yaml files to mimic matlab's ES variables generated in the loop:
	std::cout << "Loading files..." << std::endl;
	// std::string path2file("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/tests/matlab/test_dH_new_loop/test_ESloop_" + std::to_string(numiter+1) + ".yaml");
	YAML::Node node = YAML::LoadFile(path2read);
	zb_mat = node["zb"].as<Eigen::MatrixXd>();
	// std::cout << "[DBG1]" << std::endl;
	Eigen::MatrixXd Mb_mat_aux = node["Mb"].as<Eigen::MatrixXd>();
	// std::cout << "[DBG2]" << std::endl;
	// std::cout << "Mb = " << Mb.head(8).transpose() << std::endl;
	Vb_mat = node["Vb"].as<Eigen::MatrixXd>();
	// std::cout << "[DBG3]" << std::endl;
	// std::cout << "Vb = " << Vb.block(0,0,8,8) << std::endl;
	Eigen::MatrixXd logP_mat_aux = node["logP"].as<Eigen::MatrixXd>();
	// std::cout << "[DBG2]" << std::endl;
	dlogPdM_mat = node["dlogPdM"].as<Eigen::MatrixXd>();
	// std::cout << "[DBG3]" << std::endl;
	dlogPdV_mat = node["dlogPdV"].as<Eigen::MatrixXd>();
	// std::cout << "[DBG4]" << std::endl;
	ddlogPdMdM_mat = node["ddlogPdMdM"].as<std::vector<Eigen::MatrixXd>>();
	// std::cout << "[DBG5]" << std::endl;
	Eigen::MatrixXd lmb_mat_aux = node["lmb"].as<Eigen::MatrixXd>();
	// std::cout << "[DBG6]" << std::endl;
	// double S0_mat = node["S0"].as<double>();
	// std::cout << "[DBG7]" << std::endl;
	x_most_informative_mat = Eigen::VectorXd(Dim);
	if(Dim == 1)
		x_most_informative_mat(0) = node["x_next"].as<double>();
	else
		x_most_informative_mat = node["x_next"].as<Eigen::VectorXd>();
	// std::cout << "[DBG8]" << std::endl;
	EdH_max_mat = node["EdH_next"].as<double>();
	y_new_mat = node["y_next"].as<double>();
	// std::cout << "[DBG9]" << std::endl;
	std::cout << "Succesfully loaded!" << std::endl;

	// std::cout << "zb_mat.cols() = " << std::endl << zb_mat.cols() << std::endl;
	// std::cout << "zb_mat.rows() = " << std::endl << zb_mat.rows() << std::endl;
	// std::cout << "logP_mat.cols() = " << std::endl << logP_mat.cols() << std::endl;
	// std::cout << "logP_mat.rows() = " << std::endl << logP_mat.rows() << std::endl;
	// std::cout << "lmb_mat.cols() = " << std::endl << lmb_mat.cols() << std::endl;
	// std::cout << "lmb_mat.rows() = " << std::endl << lmb_mat.rows() << std::endl;
	// std::cout << "dlogPdM_mat.cols() = " << std::endl << dlogPdM_mat.cols() << std::endl;
	// std::cout << "dlogPdM_mat.rows() = " << std::endl << dlogPdM_mat.rows() << std::endl;
	// std::cout << "dlogPdV_mat.cols() = " << std::endl << dlogPdV_mat.cols() << std::endl;
	// std::cout << "dlogPdV_mat.rows() = " << std::endl << dlogPdV_mat.rows() << std::endl;
	std::cout << "Parsing appropiately..." << std::endl;
	logP_mat = logP_mat_aux.col(0);
	lmb_mat = lmb_mat_aux.col(0);
	std::cout << "Succesfully parsed!" << std::endl;
	Mb_mat 					= Mb_mat_aux.col(0);
	// std::cout << "this->Mb = " << this->Mb.head(8).transpose() << std::endl;

	return;

}

void
LoggingTools::read_for_test_SampleBeliefLocations(const std::string path2read,
																									const size_t Dim,
																									Eigen::MatrixXd & zb_mat,
									                                Eigen::VectorXd & lmb_mat,
									                                Eigen::VectorXd & x_most_informative_mat,
									                                double & EdH_max_mat){

	if(Dim > 2)
		throw std::runtime_error("@LoggingTools::read_for_test_SampleBeliefLocations() should not be called");

	std::cout << "[WARNING]: Replacing variables by some others read from a .yaml file!!!!" << std::endl;

	// Load yaml files to mimic matlab's ES variables generated in the loop:
	std::cout << "Loading files..." << std::endl;
	// std::string path2file("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/tests/matlab/test_dH_new_loop/test_ESloop_" + std::to_string(numiter+1) + ".yaml");
	YAML::Node node = YAML::LoadFile(path2read);
	zb_mat = node["zb"].as<Eigen::MatrixXd>();
	// std::cout << "[DBG1]" << std::endl;
	Eigen::MatrixXd lmb_mat_aux = node["lmb"].as<Eigen::MatrixXd>();
	// std::cout << "[DBG6]" << std::endl;
	x_most_informative_mat = Eigen::VectorXd(1);
	x_most_informative_mat(0) = node["x_next"].as<double>();
	// std::cout << "[DBG8]" << std::endl;
	EdH_max_mat = node["EdH_next"].as<double>();
	// std::cout << "[DBG9]" << std::endl;
	std::cout << "Succesfully loaded!" << std::endl;

	// std::cout << "zb_mat.cols() = " << std::endl << zb_mat.cols() << std::endl;
	// std::cout << "zb_mat.rows() = " << std::endl << zb_mat.rows() << std::endl;
	// std::cout << "logP_mat.cols() = " << std::endl << logP_mat.cols() << std::endl;
	// std::cout << "logP_mat.rows() = " << std::endl << logP_mat.rows() << std::endl;
	// std::cout << "lmb_mat.cols() = " << std::endl << lmb_mat.cols() << std::endl;
	// std::cout << "lmb_mat.rows() = " << std::endl << lmb_mat.rows() << std::endl;
	// std::cout << "dlogPdM_mat.cols() = " << std::endl << dlogPdM_mat.cols() << std::endl;
	// std::cout << "dlogPdM_mat.rows() = " << std::endl << dlogPdM_mat.rows() << std::endl;
	// std::cout << "dlogPdV_mat.cols() = " << std::endl << dlogPdV_mat.cols() << std::endl;
	// std::cout << "dlogPdV_mat.rows() = " << std::endl << dlogPdV_mat.rows() << std::endl;
	std::cout << "Parsing appropiately..." << std::endl;
	lmb_mat = lmb_mat_aux.col(0);
	std::cout << "Succesfully parsed!" << std::endl;

	return;

}


void
LoggingTools::read_for_test_SampleBeliefLocations(const std::string path2read,
																									const size_t Dim,
																									Eigen::MatrixXd & zb_mat,
									                                Eigen::VectorXd & lmb_mat,
									                                Eigen::VectorXd & Mb_mat,
																									Eigen::MatrixXd & Vb_mat,
									                                Eigen::VectorXd & x_most_informative_mat,
									                                double & y_new_mat,
									                                double & EdH_max_mat){

	if(Dim > 2)
		throw std::runtime_error("@LoggingTools::read_for_test_SampleBeliefLocations() should not be called");

	std::cout << "[WARNING]: Replacing variables by some others read from a .yaml file!!!!" << std::endl;

	// Load yaml files to mimic matlab's ES variables generated in the loop:
	std::cout << "Loading files..." << std::endl;
	// std::string path2file("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/tests/matlab/test_dH_new_loop/test_ESloop_" + std::to_string(numiter+1) + ".yaml");
	YAML::Node node = YAML::LoadFile(path2read);
	zb_mat = node["zb"].as<Eigen::MatrixXd>();
	// std::cout << "[DBG1]" << std::endl;
	Eigen::MatrixXd lmb_mat_aux = node["lmb"].as<Eigen::MatrixXd>();
	Eigen::MatrixXd Mb_mat_aux = node["Mb"].as<Eigen::MatrixXd>();
	Vb_mat = node["Vb"].as<Eigen::MatrixXd>();
	// std::cout << "[DBG2]" << std::endl;
	// std::cout << "[DBG3]" << std::endl;

	if(Dim == 1){
		x_most_informative_mat(0) = node["x_next"].as<double>();
	}
	else{
		x_most_informative_mat = node["x_next"].as<Eigen::VectorXd>();
	}

	// std::cout << "[DBG6]" << std::endl;
	EdH_max_mat = node["EdH_next"].as<double>();
	// std::cout << "[DBG7]" << std::endl;
	std::cout << "Succesfully loaded!" << std::endl;

	y_new_mat = node["y_next"].as<double>();

	// std::cout << "zb_mat.cols() = " << std::endl << zb_mat.cols() << std::endl;
	// std::cout << "zb_mat.rows() = " << std::endl << zb_mat.rows() << std::endl;
	// std::cout << "logP_mat.cols() = " << std::endl << logP_mat.cols() << std::endl;
	// std::cout << "logP_mat.rows() = " << std::endl << logP_mat.rows() << std::endl;
	// std::cout << "lmb_mat.cols() = " << std::endl << lmb_mat.cols() << std::endl;
	// std::cout << "lmb_mat.rows() = " << std::endl << lmb_mat.rows() << std::endl;
	// std::cout << "dlogPdM_mat.cols() = " << std::endl << dlogPdM_mat.cols() << std::endl;
	// std::cout << "dlogPdM_mat.rows() = " << std::endl << dlogPdM_mat.rows() << std::endl;
	// std::cout << "dlogPdV_mat.cols() = " << std::endl << dlogPdV_mat.cols() << std::endl;
	// std::cout << "dlogPdV_mat.rows() = " << std::endl << dlogPdV_mat.rows() << std::endl;
	std::cout << "Parsing appropiately..." << std::endl;
	lmb_mat = lmb_mat_aux.col(0);
	Mb_mat = Mb_mat_aux.col(0);
	std::cout << "Succesfully parsed!" << std::endl;

	return;

}

void LoggingTools::write_for_pyplot(	std::string path2store,
																				size_t Dim,
																				size_t numiter,
																				bool plot_true_function,
																				size_t Ndiv_plot,
																				libgp::GaussianProcess * gp,
																				std::shared_ptr<CostFunction> cost_function,
																				Eigen::MatrixXd z_plot,
																				Eigen::VectorXd dH_plot,
																				double EdH_max,
																				Eigen::VectorXd x_next){

	// Error checking:
	if(Dim > 2)
		throw std::runtime_error("@LoggingTools::write_for_pyplot() should not be called for Dim > 2");

	size_t Nss = gp->get_sampleset_size();
	size_t Nel = std::pow(Ndiv_plot,Dim);
	Eigen::VectorXd mpost 	= Eigen::VectorXd::Zero(Nel);
	Eigen::VectorXd stdpost = Eigen::VectorXd::Zero(Nel);
	// Eigen::VectorXd dH_plot = Eigen::VectorXd::Zero(Nel);
	Eigen::MatrixXd Xdata = Eigen::MatrixXd::Zero(Nss,Dim);
	Eigen::VectorXd Ydata = Eigen::VectorXd::Zero(Nss);
	Eigen::VectorXd f_true = Eigen::VectorXd::Zero(Nel);
	Eigen::VectorXd z_plot_i = Eigen::VectorXd::Zero(Dim);
	Eigen::VectorXd foo = Eigen::VectorXd::Zero(Dim);
	// Eigen::VectorXd x_bg = global_min_esti_x.row(numiter);
	// double mu_bg 		= global_min_esti_mux(numiter);
	// double var_bg 	= global_min_esti_varx(numiter);
	double mu_next 	= gp->f(x_next.data());
	double var;

	// Export true function:
	for(size_t i=0;i<Nel;++i){

		z_plot_i = z_plot.row(i);

		// GP posterior mean and std:
		mpost(i) = gp->f(z_plot_i.data());
		var = gp->var(z_plot_i.data());
		if(var < 0.0)
			var = 0.0;
		stdpost(i) = std::sqrt(var);

		if(plot_true_function)
			f_true(i) = cost_function->get_value(z_plot_i);

		// EdH_objective->evaluate(z_plot_i,&dH_plot(i),&foo);
	}

	// dH_plot.array() = -dH_plot.array();

	// Sample set:
	size_t Nset = gp->get_sampleset_size();
	for(size_t i=0;i<Nset;++i){
		Xdata.row(i) = gp->get_sampleset()->x(i).transpose();
		Ydata(i) = gp->get_sampleset()->y(i);
	}

  // YAML::Emitter node_to_emit; // This corresponds to the lod API. Better not to use it
  YAML::Node node_to_write;
  node_to_write.SetStyle(YAML::EmitterStyle::Block);

  // Write into the node:
  node_to_write["z_plot"] = z_plot;
  node_to_write["dH_plot"] = dH_plot;
  node_to_write["EdH_max"] = EdH_max;
  node_to_write["mpost"] = mpost;
  node_to_write["stdpost"] = stdpost;
  node_to_write["Xdata"] = Xdata;
  node_to_write["Ydata"] = Ydata;
  node_to_write["f_true"] = f_true;
  node_to_write["x_next"] = x_next;
  node_to_write["mu_next"] = mu_next;
  node_to_write["numiter"] = numiter+1;

  // Define path to file:
	std::string file_name = path2store + "/tmp.yaml";

  // Write to file:
	std::cout << "    @write_for_pyplot: Logging out variables..." << std::endl;
	std::cout << "    file_name = " << file_name << std::endl;
  std::ofstream fout(file_name);
  fout << node_to_write;
  fout.close();

  return;
}

void LoggingTools::write_progress(std::string path2store,
																		size_t Dim,
																		libgp::GaussianProcess * gp,
																		Eigen::VectorXd global_min_esti_x,	
																		Eigen::VectorXd global_min_esti_mux,
																		Eigen::VectorXd global_min_esti_varx){

  // Plotting stuff:
  size_t Nevals = gp->get_sampleset_size();
  Eigen::MatrixXd Xdata(Nevals,Dim);
  Eigen::VectorXd Ydata(Nevals);
  for(size_t i=0;i<Nevals;++i){
    Xdata.row(i) = gp->get_sampleset()->x(i);
    Ydata(i)     = gp->get_sampleset()->y(i);
  }

  // YAML::Emitter node_to_emit; // This corresponds to the lod API. Better not to use it
  YAML::Node node_to_write;
  node_to_write.SetStyle(YAML::EmitterStyle::Block);

  // Write into the node:
  node_to_write["Xdata"] = Xdata;
  node_to_write["Ydata"] = Ydata;
  node_to_write["global_min_esti_mux"] = global_min_esti_mux;
  node_to_write["global_min_esti_varx"] = global_min_esti_varx;
  node_to_write["global_min_esti_x"] = global_min_esti_x;

  // Define path to file:
	// std::string file_name = path2store + "/tmp_iter_" + std::to_string(numiter+1) + ".yaml";
	std::string file_name = path2store + "/progress_log.yaml";

  // Write to file:
	std::cout << "    @write_progress: Logging out variables..." << std::endl;
	std::cout << "    Logging into " << path2store << std::endl;
  std::ofstream fout(file_name);
  fout << node_to_write;
  fout.close();


	return;
}