// Copyright (c) 2015 Max Planck Society

/*!@file
 * @author  Alonso Marco <alonso.marco@tuebingen.mpg.de>
 *
 * @date    2015-11-20
 *
 * @details 
 * This class implements the MATLAB function dH_MC_local()
 *
 */

#ifndef INCLUDE_DH_MC_LOCAL_H
#define INCLUDE_DH_MC_LOCAL_H

#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include "LogLoss.hpp"
#include "logsumexp.hpp"
#include "GP_innovation_local.hpp"
#include "MathTools.hpp"
#include "erfinv.h"


/**
 * Class that wraps the dHdx_local function from the Matlab implementation as an object. 
 * A pointer to this object must be passed to the optimizer in the main function.*/
class dH_MC_local
{

public:

  /**
   * Default constructor
  */
  dH_MC_local(){};

  /** 
   * Constructor of the class. It basically constructs the class GPInnovationLocal, whose function
   * efficient_innovation() is eventually used to compute the expected change in entropy.
   * @param zbel an NbxD Matrix where D is the input dimensionality, and Nb is the number of representer points.
   * TODO @param gp pointer to a Gaussian process (libGP) 
   * @param logP an Nbx1 Vector Nb is the number of representer points.
   * @param dlogPdM an NbxNb Vector Nb is the number of representer points.
   * @param dlogPdV an Nbx(0.5*Nb(Nb+1)) Vector Nb is the number of representer points.
   * @param ddlogPdMdM
   * @param T an integer with the number of samples in entropy prediction
   * @param lmb an Nbx1 Vector, where Nb is the number of representer points.
   * @param xmin an Dx1 Vector, where D is the input dimensionality
   * @param xmax an Dx1 Vector, where D is the input dimensionality
   * @param invertsign a boolean flag
   * @param LossFunc: it is not passed in to the function. Instead, the function LogLoss 
            inside the class LogLoss_class is invoked.
   */
  dH_MC_local(const Eigen::MatrixXd zbel, 
                    Eigen::VectorXd logP, 
                    Eigen::MatrixXd dlogPdM, 
                    Eigen::MatrixXd dlogPdV, 
                    std::vector<Eigen::MatrixXd> ddlogPdMdM,
                    const size_t T,
                    Eigen::VectorXd lmb,
                    Eigen::VectorXd xmin,
                    Eigen::VectorXd xmax,
                    bool invertsign,
                    libgp::GaussianProcess * gp);

  ~dH_MC_local();

  /**
   * This is the actual function that computes the expected change in Entropy and its derivative at each point x
   * given the log values of p_min and its derivatives, and the innovation function.
   * When class dH_MC_local() is constructed, all the private variables are accessible for this function. Then, 
   * the optimizer can acess the function with the evaluation point x as input, without the need of all the other parameters. 
   * @param x an 1xD Vector, where D is the input dimensionality.
   * @param dH a scalar that represents the expected change in entropy at location x
   * @param ddHdx a Dx1 vector that represents the gradient of the expected change in entropy at location x
   */
  void dHdx_local(const Eigen::VectorXd x, double & dH, Eigen::VectorXd & ddHdx);

  /**
   * Actual function that computes dH. This function is called 2*D+1 times from dH_MC_local().
   * @param x an 1xD Vector, where D is the input dimensionality.
   */
  double get_dH(const Eigen::VectorXd x);

  void change_sign(bool invertsign);

  
  void update_variables(Eigen::MatrixXd zbel,
                        Eigen::VectorXd logP,
                        Eigen::MatrixXd dlogPdM,
                        Eigen::MatrixXd dlogPdV,
                        std::vector<Eigen::MatrixXd> ddlogPdMdM,
                        Eigen::VectorXd lmb,
                        libgp::GaussianProcess * gp);

private:
  GPInnovationLocal * dGP;
  LogLoss           LogLoss_;
  logsumexp         logsumexp_;

  Eigen::VectorXd logP;
  Eigen::MatrixXd dlogPdM;
  Eigen::MatrixXd dlogPdV;
  std::vector<Eigen::MatrixXd> ddlogPdMdM;
  Eigen::VectorXd lmb;
  Eigen::VectorXd xmin;
  Eigen::VectorXd xmax;
  bool invertsign;
  Eigen::VectorXd W;


};

#endif

