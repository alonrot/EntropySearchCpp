// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_noise.h"
#include <cmath>

namespace libgp
{
  
  CovNoise::CovNoise() {}
  
  CovNoise::~CovNoise() {}
  
  bool CovNoise::init(int n)
  {
    input_dim = n;
    param_dim = 1;
    loghyper.resize(param_dim);
    loghyper.setZero();
    return true;
  }
  
  double CovNoise::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    // if (&x1 == &x2) // amarcovalle: This doesn't work in many cases. For example, get(x,x.transpose()), with Eigen::VectorXd x returns 0.

    // This instead:
    Eigen::VectorXd comp_vec(this->input_dim);
    for(size_t i=0;i<this->input_dim;++i)
      comp_vec(i) = x1(i) == x2(i);

    if(comp_vec.prod())
      return s2;
    else 
      return 0.0;
  }
  
  void CovNoise::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    if (&x1 == &x2) grad(0) = 2*s2;
    else grad(0) = 0.0;
  }

  void CovNoise::compute_dkdx(const Eigen::VectorXd & x, const Eigen::Ref<const Eigen::VectorXd> & kZx, const Eigen::Ref<const Eigen::MatrixXd> & Z, Eigen::Ref<Eigen::MatrixXd> dkZxdx)
  {
    dkZxdx.setZero();
  }
  
  void CovNoise::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    s2 = exp(2*loghyper(0));
  }
  
  std::string CovNoise::to_string()
  {
    return "CovNoise";
  }
  
  double CovNoise::get_threshold()
  {
    return 0.0;
  }
  
  void CovNoise::set_threshold(double threshold) {}

}