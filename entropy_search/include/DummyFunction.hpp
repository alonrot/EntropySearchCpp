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
#ifndef __DUMMY_FUNCTION_H__
#define __DUMMY_FUNCTION_H__

#include <cmath>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include "optimizer/bfgs_optimizer.h" // #include "optimizer/objective_function.h" included here
#include "dH_MC_local.hpp"
#include "gp.h"
#include "ReadYamlParameters.hpp"

class DummyFunction : public bfgs_optimizer::ObjectiveFunction {
  public:
    DummyFunction(){};
    virtual ~DummyFunction() {}
    virtual std::string function_name(void) const = 0;
    virtual void update_something(Eigen::VectorXd new_param){};
    virtual void update_gp(libgp::GaussianProcess * gp){};
    virtual void change_output_sign(bool invert_sign){};
    virtual void update_variables(Eigen::MatrixXd zbel,
                                  Eigen::VectorXd logP,
                                  Eigen::MatrixXd dlogPdM,
                                  Eigen::MatrixXd dlogPdV,
                                  std::vector<Eigen::MatrixXd> ddlogPdMdM,
                                  Eigen::VectorXd lmb,
                                  libgp::GaussianProcess * gp){};
};

class DummyFunctionChild : public DummyFunction {

  public:
    DummyFunctionChild(Eigen::VectorXd param);
    virtual ~DummyFunctionChild() {}
    void evaluate(const Eigen::VectorXd& x, double* function_value, Eigen::VectorXd* derivative) const;
    std::string function_name(void) const;
    void update_something(Eigen::VectorXd new_param);

  private:
    Eigen::VectorXd param;
    int D;
};


class WrapperGP : public DummyFunction {

  public:
    WrapperGP(libgp::GaussianProcess * gp, Eigen::VectorXd xmin, Eigen::VectorXd xmax) : gp(gp), xmin(xmin), xmax(xmax){};
    ~WrapperGP() {}
    void evaluate(const Eigen::VectorXd& x, double* function_value, Eigen::VectorXd* derivative) const;
    std::string function_name(void) const;
    void update_gp(libgp::GaussianProcess * gp);

  private:
    libgp::GaussianProcess * gp;
    Eigen::VectorXd xmin;
    Eigen::VectorXd xmax;
};

class WrapperEdH : public DummyFunction {

  public:
    // Constructor: inline initialization of class members
    WrapperEdH(dH_MC_local * dHdx) : dHdx(dHdx){}; // TODO: Should we compute ddHdx before or after constraining dH?
    ~WrapperEdH() {};
    void evaluate(const Eigen::VectorXd& x, double* function_value, Eigen::VectorXd* derivative) const;
    std::string function_name(void) const;
    void change_output_sign(bool invert_sign);
    void update_something(Eigen::VectorXd new_param){};
    void update_variables(Eigen::MatrixXd zbel,
                          Eigen::VectorXd logP,
                          Eigen::MatrixXd dlogPdM,
                          Eigen::MatrixXd dlogPdV,
                          std::vector<Eigen::MatrixXd> ddlogPdMdM,
                          Eigen::VectorXd lmb,
                          libgp::GaussianProcess * gp);


  private:
    dH_MC_local * dHdx;
};

// Testing the bfgs_optimizer::ObjectiveFunction
class DummyFunctionChildBFGS : public bfgs_optimizer::ObjectiveFunction{

  public:
    DummyFunctionChildBFGS(Eigen::VectorXd param, double xmin_s, double xmax_s);
    ~DummyFunctionChildBFGS() {}
    void evaluate(const Eigen::VectorXd& x, double* function_value, Eigen::VectorXd* derivative) const;

  private:
    Eigen::VectorXd param;
    Eigen::VectorXd xmin;
    Eigen::VectorXd xmax;
    int D;
};

#endif /* __DUMMY_FUNCTION_H__ */