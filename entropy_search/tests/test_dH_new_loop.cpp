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

  size_t Dim = 1;
  // size_t Nrepresenters; 
  // size_t Nsubsamples;
  size_t hyperparam_dim; // lengthscales + signal variance + measurement noise
  size_t Nll;
  // // size_t T;
  size_t Ndiv_plot = 200; // for plotting
  // size_t Ndata_init;
  // size_t Nwarm_starts;
  // size_t MaxEvals;
  // bool   write2file;
  std::string which_kernel;
  // bool    learn_hypers;
  double xmin_s = 0.0;
  double xmax_s = 1.0;
  double lengthscale_s = 0.1;
  double prior_std_s = 2.0;
  double prior_std_n = 0.1;

  size_t numiter = 9;
  std::string path2read("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/tests/matlab/test_dH_new_loop/test_ESloop_" + std::to_string(numiter+1) + ".yaml");
  std::string path2write("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/tests/matlab/test_dH_new_loop/test_ESloop_" + std::to_string(numiter+1) + "_output.yaml");

  // Load yaml files to mimic matlab's ES variables generated in the loop:
  std::cout << "Loading files..." << std::endl;
  YAML::Node node = YAML::LoadFile(path2read);
  Eigen::MatrixXd zb_mat = node["zb"].as<Eigen::MatrixXd>();
  std::cout << "[DBG1]" << std::endl;
  Eigen::MatrixXd logP_mat = node["logP"].as<Eigen::MatrixXd>();
  std::cout << "[DBG2]" << std::endl;
  Eigen::MatrixXd dlogPdM_mat = node["dlogPdM"].as<Eigen::MatrixXd>();
  std::cout << "[DBG3]" << std::endl;
  Eigen::MatrixXd dlogPdV_mat = node["dlogPdV"].as<Eigen::MatrixXd>();
  std::cout << "[DBG4]" << std::endl;
  std::vector<Eigen::MatrixXd> ddlogPdMdM_mat = node["ddlogPdMdM"].as<std::vector<Eigen::MatrixXd>>();
  std::cout << "[DBG5]" << std::endl;
  Eigen::MatrixXd lmb_mat = node["lmb"].as<Eigen::MatrixXd>();


  double X_mat_set_d;
  double Y_mat_set_d;
  Eigen::MatrixXd Y_mat_set_arr;

  Eigen::MatrixXd X_mat_set_par;
  Eigen::VectorXd Y_mat_set_par;

  if(numiter == 1){
    X_mat_set_d = node["X_set"].as<double>();
    Y_mat_set_d = node["Y_set"].as<double>();

    X_mat_set_par = Eigen::MatrixXd(1,1); X_mat_set_par(0,0) = X_mat_set_d;
    Y_mat_set_par = Eigen::VectorXd(1); Y_mat_set_par(0) = Y_mat_set_d;
  }
  else{
    X_mat_set_par = node["X_set"].as<Eigen::MatrixXd>();

    Y_mat_set_arr = node["Y_set"].as<Eigen::MatrixXd>();
    Y_mat_set_par = Y_mat_set_arr.col(0);
  }
  // std::cout << "[DBG6]" << std::endl;
  // double S0_mat = node["S0"].as<double>();
  // std::cout << "[DBG7]" << std::endl;
  // double x_next_mat = node["x_next"].as<double>();
  // std::cout << "[DBG8]" << std::endl;
  // double EdH_next_mat = node["EdH_next"].as<double>();
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
  Eigen::VectorXd logP_mat_par = logP_mat.col(0);
  Eigen::VectorXd lmb_mat_par = lmb_mat.col(0);
  std::cout << "Succesfully parsed!" << std::endl;

  // Compute some other parameters:
  Nll = Dim;
  hyperparam_dim = Nll + 1 + 1;

  // Initialize the GP:
  gp = new libgp::GaussianProcess(Dim,"CovSum ( CovSEard, CovNoise)");
  // Initialize the hyperparameters:
  Eigen::VectorXd hyperparams(hyperparam_dim);
  hyperparams << log(lengthscale_s), log(prior_std_s), log(prior_std_n); // lengthscale, signal std, noise std
  gp->covf().set_loghyper(hyperparams);

  size_t Ndata_set = Y_mat_set_par.size();
  for(size_t i=0;i<Ndata_set;++i)
    gp->add_pattern(X_mat_set_par.row(i).data(),Y_mat_set_par(i));


  std::cout << "X_mat_set_par = \n" << X_mat_set_par << std::endl;
  std::cout << "Y_mat_set_par = " << Y_mat_set_par << std::endl;

  // std::cout << "zbel.rows() = " << zbel.rows() << std::endl;
  // std::cout << "zbel.cols() = " << zbel.cols() << std::endl;
  // std::cout << "logP.size() = " << logP.size() << std::endl;
  // std::cout << "dlogPdMu.rows() = " << dlogPdMu.rows() << std::endl;
  // std::cout << "dlogPdMu.cols() = " << dlogPdMu.cols() << std::endl;
  // std::cout << "dlogPdSigma.rows() = " << dlogPdSigma.rows() << std::endl;
  // std::cout << "dlogPdSigma.cols() = " << dlogPdSigma.cols() << std::endl;
  // std::cout << "dlogPdMudMu_std.size() = " << dlogPdMudMu_std.size() << std::endl;
  // std::cout << "dlogPdMudMu_std[0].rows() = " << dlogPdMudMu_std[0].rows() << std::endl;
  // std::cout << "dlogPdMudMu_std[0].cols() = " << dlogPdMudMu_std[0].cols() << std::endl;
  // std::cout << "lmb.size() = " << lmb.size() << std::endl;
  // std::cout << "[DBG]: @EntropySearch::run - Paused for debugging" <<std::endl;

  // std::chrono::seconds dura(2);
  // std::this_thread::sleep_for(dura);

  // // TODO: We construct the class. In the final code, this class must be constructed
  // // in the main file. In mex files, the function mexFunction() is like the main() function.
  // std::vector<Eigen::MatrixXd> dlogPdMudMu(Nrepresenters);
  // for(size_t i=0;i<Nrepresenters;++i)
  //   dlogPdMudMu[i] = dlogPdMudMu_std[i];

  Eigen::MatrixXd zbel = zb_mat;
  Eigen::VectorXd logP = logP_mat;
  Eigen::VectorXd lmb = lmb_mat_par;
  Eigen::MatrixXd dlogPdMu = dlogPdM_mat;
  Eigen::MatrixXd dlogPdSigma = dlogPdV_mat;
  std::vector<Eigen::MatrixXd> dlogPdMudMu = ddlogPdMdM_mat;
  size_t T = 200;
  Eigen::VectorXd xmin = Eigen::VectorXd::Ones(Dim) * xmin_s;
  Eigen::VectorXd xmax = Eigen::VectorXd::Ones(Dim) * xmax_s;
  bool invertsign = false;

  dH_MC_local * EdH_class = new dH_MC_local(zbel,logP,dlogPdMu,dlogPdSigma,dlogPdMudMu,T,lmb,xmin,xmax,invertsign,gp);


  // // Double-check: (double-checked: they are the same)
  // std::cout << "dlogPdMu.block(0,0,10,10) = " << dlogPdMu.block(0,0,10,10) << std::endl;
  // std::cout << "dlogPdSigma.block(0,0,10,10) = " << dlogPdSigma.block(0,0,10,10) << std::endl;
  // std::cout << "dlogPdMudMu[0].block(0,0,10,10) = " << dlogPdMudMu[0].block(0,0,10,10) << std::endl;
  // std::cout << "dlogPdMudMu[1].block(0,0,10,10) = " << dlogPdMudMu[1].block(0,0,10,10) << std::endl;

  // TODO: remove this in the final code. It just serves for testing purposes.
  // x_test must be the same as the one in test_dH_MC_local.m
  Eigen::VectorXd x_test = Eigen::VectorXd(1);
  x_test(0) = 0.45;

  // Using the function itself:
  // TODO: We call dHdx_local() here, for testing purposes. 
  // In the final code, this class must be called from the optimizer.
  // double dH = 0.0;
  // Eigen::VectorXd ddHdx = Eigen::VectorXd::Zero(Dim);
  // EdH_class->dHdx_local(x_test,dH,ddHdx);

  // std::cout << "Using the function directly:" << std::endl;
  // std::cout << "dH = " << dH << std::endl;
  // std::cout << "ddHdx = " <<  std::endl << ddHdx << std::endl;
  // std::cout << std::endl;

  // Entropy search wrapper:
  // bfgs_optimizer::ObjectiveFunction * EdH_objective = new WrapperEdH(EdH_class);
  DummyFunction * EdH_objective = new WrapperEdH(EdH_class);

  double dH_w = 0.0;
  Eigen::VectorXd ddHdx_w = Eigen::VectorXd::Zero(Dim);
  EdH_objective->evaluate(x_test,&dH_w,&ddHdx_w);

  double dz = (xmax_s - xmin_s)/(Ndiv_plot-1);
  Eigen::VectorXd z_plot = Eigen::VectorXd::Zero(Ndiv_plot);
  Eigen::VectorXd z_plot_i = Eigen::VectorXd::Zero(Dim);
  Eigen::VectorXd foo = Eigen::VectorXd::Zero(Dim);
  Eigen::VectorXd dH_plot = Eigen::VectorXd::Zero(Ndiv_plot);
  for(size_t i=0;i<Ndiv_plot;++i){
    z_plot(i) = xmin_s + dz * i;
    z_plot_i(0) = z_plot(i);
    EdH_objective->evaluate(z_plot_i,&dH_plot(i),&foo);
  }
  dH_plot.array() = -dH_plot.array();

  std::cout << "Writing data to files... " << std::endl;
  YAML::Node node_to_write;
  node_to_write.SetStyle(YAML::EmitterStyle::Block);
  node_to_write["dH_plot"] = dH_plot;

  // Write to file:
  std::ofstream fout(path2write);
  fout << node_to_write;
  fout.close();
  std::cout << "Writing succesful!" << std::endl;

}
