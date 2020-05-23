// Copyright (c) 2014 Max Planck Society

/*!@file
 * @author  Simon Bartels <sbartels@tuebingen.mpg.de>
 *
 * @date    2015-10-27
 *
 * @details 
 * This class implements the Matlab GPInnovationLocal functionality.
 *
 */

#ifndef __INCLUDE_EXPECTED_IMPROVEMENT_LOCAL_H__
#define __INCLUDE_EXPECTED_IMPROVEMENT_LOCAL_H__

#include <Eigen/Dense>
#include "gp.h"
#include "MathTools.hpp"
#include "DummyFunction.hpp"
#include <cmath>
#include <iostream>

class ExpectedImprovement : public DummyFunction {

  public:
    ExpectedImprovement(libgp::GaussianProcess * gp, Eigen::VectorXd xmin, Eigen::VectorXd xmax, bool invertsign = false);
    ~ExpectedImprovement() {}
    void evaluate(const Eigen::VectorXd& x, double* function_value, Eigen::VectorXd* derivative) const;
    std::string function_name(void) const;
    void update_something(Eigen::VectorXd new_param){};
    void update_gp(libgp::GaussianProcess * gp);

  private:
    double normpdf(double muX_min, double mux, double sx) const;
    double normcdf(double muX_min, double mux, double sx) const;
    double update_fmin(libgp::GaussianProcess * gp);
    double update_noise(libgp::GaussianProcess * gp);
    libgp::GaussianProcess * gp;
    Eigen::VectorXd xmin;
    Eigen::VectorXd xmax;
    double fmin;
    int my_sign;
    double sn2;
};

#endif // __INCLUDE_EXPECTED_IMPROVEMENT_LOCAL_H__
