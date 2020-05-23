// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

// #include "entropy_change.h"
#include "dH_MC_local.hpp"
#include "DummyFunction.hpp"
#include "ReadYamlParameters.hpp"

// For Debugging:
    #include <chrono>
    #include <thread>

int main (int argc, char const *argv[])
{
  // Declare the GP:
  libgp::GaussianProcess * gp;

  size_t Dim;
  size_t Nrepresenters; 
  size_t hyperparam_dim; // lengthscales + signal variance + measurement noise
  size_t Nll;
  size_t T;
  size_t Ndiv_plot; // for plotting
  std::string which_kernel;
  double xmin_s;
  double xmax_s;
  double lengthscale_s;
  double prior_std_s;
  double prior_std_n;

  std::cout << "Loading input parameters" << std::endl;
  std::cout << "========================" << std::endl;
  std::cout << "Path: " << YAML_CONFIG_FILE << std::endl;
  std::cout << "Note: This is defined in the CMakeLists.txt" << std::endl;

  // Load parameters:
  ReadYamlParameters params = ReadYamlParameters(YAML_CONFIG_FILE); // this is defined in the CMakeLists.txt
  Dim             = params.get<size_t>("Dim");
  Nrepresenters   = params.get<size_t>("Nrepresenters");
  T               = params.get<size_t>("T");
  Ndiv_plot       = params.get<size_t>("Ndiv_plot");
  which_kernel    = params.get<std::string>("which_kernel");
  xmin_s          = params.get<double>("xmin_s");
  xmax_s          = params.get<double>("xmax_s");
  lengthscale_s   = params.get<double>("lengthscale_s");
  prior_std_s     = params.get<double>("prior_std_s");
  prior_std_n     = params.get<double>("prior_std_n");

  // Compute some other parameters:
  Nll = Dim;
  hyperparam_dim = Nll + 1 + 1;

  // Initialize the GP:
  gp = new libgp::GaussianProcess(Dim,which_kernel);
  // Initialize the hyperparameters:
  Eigen::VectorXd hyperparams(hyperparam_dim);
  hyperparams << log(lengthscale_s), log(prior_std_s), log(prior_std_n); // lengthscale, signal std, noise std
  gp->covf().set_loghyper(hyperparams);

  YAML::Node node = YAML::LoadFile("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/test_dH_input.yaml");
  Eigen::MatrixXd zbel      = node["zb"].as<Eigen::MatrixXd>();
  Eigen::VectorXd logP      = node["logP"].as<Eigen::VectorXd>();
  Eigen::MatrixXd dlogPdMu  = node["dlogPdMu"].as<Eigen::MatrixXd>();
  Eigen::MatrixXd dlogPdSigma = node["dlogPdSigma"].as<Eigen::MatrixXd>();
  std::vector<Eigen::MatrixXd> dlogPdMudMu_std = node["dlogPdMudMu"].as<std::vector<Eigen::MatrixXd>>();
  Eigen::VectorXd lmb = node["lmb"].as<Eigen::VectorXd>();
  Eigen::VectorXd x_data_set = node["x_data_set"].as<Eigen::VectorXd>();
  Eigen::VectorXd y_data_set = node["y_data_set"].as<Eigen::VectorXd>();
  bool invertsign = false;
  Eigen::VectorXd xmin = Eigen::VectorXd::Ones(Dim) * xmin_s;
  Eigen::VectorXd xmax = Eigen::VectorXd::Ones(Dim) * xmax_s;

  size_t Ndata_set = y_data_set.size();
  for(size_t i=0;i<Ndata_set;++i)
    gp->add_pattern(x_data_set.row(i).data(),y_data_set(i));


  std::cout << "x_data_set = \n" << x_data_set << std::endl;
  std::cout << "y_data_set = " << y_data_set << std::endl;

  std::cout << "zbel.rows() = " << zbel.rows() << std::endl;
  std::cout << "zbel.cols() = " << zbel.cols() << std::endl;
  std::cout << "logP.size() = " << logP.size() << std::endl;
  std::cout << "dlogPdMu.rows() = " << dlogPdMu.rows() << std::endl;
  std::cout << "dlogPdMu.cols() = " << dlogPdMu.cols() << std::endl;
  std::cout << "dlogPdSigma.rows() = " << dlogPdSigma.rows() << std::endl;
  std::cout << "dlogPdSigma.cols() = " << dlogPdSigma.cols() << std::endl;
  std::cout << "dlogPdMudMu_std.size() = " << dlogPdMudMu_std.size() << std::endl;
  std::cout << "dlogPdMudMu_std[0].rows() = " << dlogPdMudMu_std[0].rows() << std::endl;
  std::cout << "dlogPdMudMu_std[0].cols() = " << dlogPdMudMu_std[0].cols() << std::endl;
  std::cout << "lmb.size() = " << lmb.size() << std::endl;
  std::cout << "[DBG]: @EntropySearch::run - Paused for debugging" <<std::endl;

  // std::chrono::seconds dura(2);
  // std::this_thread::sleep_for(dura);

  // TODO: We construct the class. In the final code, this class must be constructed
  // in the main file. In mex files, the function mexFunction() is like the main() function.
  std::vector<Eigen::MatrixXd> dlogPdMudMu(Nrepresenters);
  for(size_t i=0;i<Nrepresenters;++i)
    dlogPdMudMu[i] = dlogPdMudMu_std[i];

  dH_MC_local * EdH_class = new dH_MC_local(zbel,logP,dlogPdMu,dlogPdSigma,dlogPdMudMu,T,lmb,xmin,xmax,invertsign,gp);

  // // Double-check: (double-checked: they are the same)
  // std::cout << "dlogPdMu.block(0,0,10,10) = " << dlogPdMu.block(0,0,10,10) << std::endl;
  // std::cout << "dlogPdSigma.block(0,0,10,10) = " << dlogPdSigma.block(0,0,10,10) << std::endl;
  // std::cout << "dlogPdMudMu[0].block(0,0,10,10) = " << dlogPdMudMu[0].block(0,0,10,10) << std::endl;
  // std::cout << "dlogPdMudMu[1].block(0,0,10,10) = " << dlogPdMudMu[1].block(0,0,10,10) << std::endl;

  // TODO: remove this in the final code. It just serves for testing purposes.
  // x_test must be the same as the one in test_dH_MC_local.m
  Eigen::VectorXd x_test = Eigen::VectorXd(Dim);
  x_test(0) = 0.45;

  // Using the function itself:
  // TODO: We call dHdx_local() here, for testing purposes. 
  // In the final code, this class must be called from the optimizer.
  double dH = 0.0;
  Eigen::VectorXd ddHdx = Eigen::VectorXd::Zero(Dim);
  EdH_class->dHdx_local(x_test,dH,ddHdx);

  std::cout << "Using the function directly:" << std::endl;
  std::cout << "dH = " << dH << std::endl;
  std::cout << "ddHdx = " <<  std::endl << ddHdx << std::endl;
  std::cout << std::endl;

  // Entropy search wrapper:
  // bfgs_optimizer::ObjectiveFunction * EdH_objective = new WrapperEdH(EdH_class);
  DummyFunction * EdH_objective = new WrapperEdH(EdH_class);

  double dH_w = 0.0;
  Eigen::VectorXd ddHdx_w = Eigen::VectorXd::Zero(Dim);
  EdH_objective->evaluate(x_test,&dH_w,&ddHdx_w);

  std::cout << " ** Using the wrapper:" << std::endl;
  std::cout << "    dH_w = " << dH_w << std::endl;
  std::cout << "    ddHdx_w = " << std::endl << ddHdx_w << std::endl;
  std::cout << std::endl;


  // Continuous function for plotting:
  // Vectors:
  Eigen::VectorXd z_plot = Eigen::VectorXd::Zero(Ndiv_plot);
  Eigen::VectorXd dH_plot = Eigen::VectorXd::Zero(Ndiv_plot);
  Eigen::MatrixXd ddHdx_plot = Eigen::MatrixXd::Zero(Ndiv_plot,Dim);
  Eigen::VectorXd ddHdx_plot_i = Eigen::VectorXd::Zero(Dim);
  double dz = (xmax_s - xmin_s)/(Ndiv_plot-1);
  Eigen::VectorXd z_plot_i = Eigen::VectorXd::Zero(Dim);
  for(size_t i=0;i<Ndiv_plot;++i){
    z_plot(i) = xmin_s + dz * i;
    z_plot_i(0) = z_plot(i);
    EdH_objective->evaluate(z_plot.row(i),&dH_plot(i),&ddHdx_plot_i);
    ddHdx_plot.row(i) = ddHdx_plot_i;
  }

  // Save the data to a file:
  std::cout << "Writing data to files" << std::endl;
  std::string path_to_file("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/test_dH_output.yaml");
  YAML::Node node_to_write;
  node_to_write["z_plot"] = z_plot;
  node_to_write["dH_plot"] = dH_plot;
  node_to_write["ddHdx_plot"] = ddHdx_plot;

  // Write to file:
  std::ofstream fout(path_to_file);
  fout << node_to_write;
  fout.close();
  std::cout << "Writing succesful!" << std::endl;

  // Calling the optimizer:
  bfgs_optimizer::BFGS bfgs(EdH_objective,10);

  // Initial guess:
  Eigen::VectorXd x_init  = Eigen::VectorXd(Dim);
  Eigen::VectorXd x_final = Eigen::VectorXd(Dim);

  x_init << x_test;

  x_final = bfgs.minimize(x_init);

  std::cout << " ** Optimization:" << std::endl;
  std::cout << "    x_init = " << x_init << std::endl;
  std::cout << "    x_final = " << x_final << std::endl;
  std::cout << "[WARNING]: Still needs to be tested with actual non-zero entries logP, dlogPdM, etc. "<< std::endl;
  std::cout << std::endl;

  // Change sign:
  EdH_objective->change_output_sign(true);

  double dH_new_sign = 0.0;
  Eigen::VectorXd ddHdx_new_sign = Eigen::VectorXd::Zero(Dim);
  EdH_objective->evaluate(x_test,&dH_new_sign,&ddHdx_new_sign);

  std::cout << "Using the function directly:" << std::endl;
  std::cout << "dH_new_sign = " << dH_new_sign << std::endl;
  std::cout << "ddHdx_new_sign = " <<  ddHdx_new_sign.transpose() << std::endl;
  std::cout << std::endl;

  // Call update function:
  EdH_class->update_variables(zbel,logP,dlogPdMu,dlogPdSigma,dlogPdMudMu,lmb,gp);


}
