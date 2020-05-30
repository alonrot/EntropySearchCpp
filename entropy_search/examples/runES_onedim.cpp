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
#include "EntropySearch.hpp"

// For Debugging:
    #include <chrono>
    #include <thread>

int main (int argc, char const *argv[])
{

  std::cout << "Loading input parameters" << std::endl;
  std::cout << "========================" << std::endl;
  std::cout << "Path: " << YAML_CONFIG_PATH << std::endl;
  std::cout << "Note: This is defined in the CMakeLists.txt" << std::endl;

  // Load parameters:
  ReadYamlParameters params = ReadYamlParameters(std::string(YAML_CONFIG_PATH) + "input_parameters_tmpl.yaml"); // this is defined in the CMakeLists.txt
  
  // Standard options:
  size_t  Dim             = params.get<size_t>("Dim");
  size_t  Nrepresenters   = params.get<size_t>("Nrepresenters");
  size_t  Nsubsamples     = params.get<size_t>("Nsubsamples");
  size_t  T               = params.get<size_t>("T");
  size_t  Ndiv_plot       = params.get<size_t>("Ndiv_plot");
  size_t  Ndata_init      = params.get<size_t>("Ndata_init");
  size_t  Nwarm_starts    = params.get<size_t>("Nwarm_starts");
  size_t  MaxEvals        = params.get<size_t>("MaxEvals");
  size_t  Nline_searches  = params.get<size_t>("Nline_searches");
  double  xmin_s          = params.get<double>("xmin_s");
  double  xmax_s          = params.get<double>("xmax_s");
  bool    learn_hypers    = params.get<bool>("learn_hypers");
  double  sleep_time      = params.get<double>("sleep_time");
  std::string which_kernel    = params.get<std::string>("which_kernel");
  std::string name_evalfun = params.get<std::string>("name_evalfun");
  std::string path2data_logging_relative = params.get<std::string>("path2data_logging_relative");

  // DBG flags:
  bool    write2file                                = params.get<bool>("write2file");
  bool    write2pyplot                              = params.get<bool>("write2pyplot");
  bool    write_matrices_to_file                    = params.get<bool>("write_matrices_to_file");
  bool    read_for_test_dH_MC_local_flag            = params.get<bool>("read_for_test_dH_MC_local_flag");
  bool    read_for_test_SampleBeliefLocations_flag  = params.get<bool>("read_for_test_SampleBeliefLocations_flag");
  bool    plot_true_function                        = params.get<bool>("plot_true_function");

  // Initialize the hyperparameters:
  size_t Nll = Dim;
  size_t hyperparam_dim = Nll + 1 + 1; // lengthscales + signal variance + measurement noise
  std::vector<double> lengthscale_s_aux = params.get<std::vector<double>>("lengthscale_s");
  Eigen::VectorXd lengthscale_s = Eigen::Map<Eigen::VectorXd>(lengthscale_s_aux.data(), lengthscale_s_aux.size());
  double  prior_std_s     = params.get<double>("prior_std_s");
  double  prior_std_n     = params.get<double>("prior_std_n");
  Eigen::VectorXd hyperparams(hyperparam_dim);
  hyperparams.head(Nll) = lengthscale_s.array().log();
  hyperparams(Nll)    = std::log(prior_std_s);
  hyperparams(Nll+1)  = std::log(prior_std_n);

  // Initialize the GP:
  libgp::GaussianProcess * gp = new libgp::GaussianProcess(Dim,which_kernel);
  gp->covf().set_loghyper(hyperparams);

  // Evaluation function:
  std::string path2evalfun(std::string(YAML_CONFIG_PATH) + "../examples/" + name_evalfun);
  std::shared_ptr<CostFunction> cost_function = std::make_shared<CostFunctionFromPrior>(Dim,Ndiv_plot,path2evalfun);

  // Add data:
  if(Ndata_init > 0){
    Eigen::MatrixXd X(Ndata_init,Dim);
    Eigen::VectorXd foo(Dim);
    Eigen::VectorXd Y(Ndata_init);
    MathTools::sample_unif_mat(xmin_s,xmax_s,X);

    for(size_t i = 0; i < Ndata_init; ++i)
      Y(i) = cost_function->get_value(X.row(i));

    // Add to GP:
    for(size_t i = 0; i < Ndata_init; ++i) 
      gp->add_pattern(X.row(i).data(), Y(i));
  }

  // Path to plot:
  std::string path2data_logging_absolute(std::string(YAML_CONFIG_PATH) + "../" + path2data_logging_relative);
  std::cout << "path2data_logging_absolute = " << path2data_logging_absolute << std::endl;

  // Intialize input structure:
  std::shared_ptr<INSetup> in = std::make_shared<INSetup>();

  // Standard options:
  in->Dim  = Dim;
  in->Nrepresenters = Nrepresenters;
  in->Nsubsamples = Nsubsamples;
  in->T  = T;
  in->Ndiv_plot   = Ndiv_plot;
  in->Nwarm_starts = Nwarm_starts;
  in->MaxEval = MaxEvals;
  in->Nline_searches = Nline_searches;
  in->xmin_s = xmin_s;
  in->xmax_s = xmax_s;
  in->LearnHypers = learn_hypers;

  // DBG flags:
  in->write2file = write2file;
  in->write_matrices_to_file = write_matrices_to_file;
  in->read_for_test_dH_MC_local_flag = read_for_test_dH_MC_local_flag;
  in->read_for_test_SampleBeliefLocations_flag = read_for_test_SampleBeliefLocations_flag;

  // Plot with python:
  in->write2pyplot = write2pyplot;
  in->path2data_logging_absolute = path2data_logging_absolute;
  in->plot_true_function = plot_true_function;
  in->sleep_time = sleep_time;

  // Gaussian process:
  in->gp = gp;

  // Evaluation function:
  in->cost_function = cost_function;
  in->Nll = Nll;

  // Construct ES
  EntropySearch es(in);

  // Run ES:
  OutResults out_results;
  out_results = es.run();

}
