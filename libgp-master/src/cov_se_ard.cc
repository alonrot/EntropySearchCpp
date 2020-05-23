// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_se_ard.h"
#include <cmath>

namespace libgp
{
  
  CovSEard::CovSEard() {}
  
  CovSEard::~CovSEard() {}
  
  bool CovSEard::init(int n)
  {
    input_dim = n;
    param_dim = n+1;
    ell.resize(input_dim);
    loghyper.resize(param_dim);
    return true;
  }
  
  double CovSEard::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {  
    double z = (x1-x2).cwiseQuotient(ell).squaredNorm();
    return sf2*exp(-0.5*z);
  }
  
  void CovSEard::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    Eigen::VectorXd z = (x1-x2).cwiseQuotient(ell).array().square();  
    double k = sf2*exp(-0.5*z.sum());
    grad.head(input_dim) = z * k;
    grad(input_dim) = 2.0 * k;
  }

  void CovSEard::compute_dkdx(const Eigen::VectorXd & x, const Eigen::Ref<const Eigen::VectorXd> & kZx_row, const Eigen::Ref<const Eigen::MatrixXd> & Z_col, Eigen::Ref<Eigen::MatrixXd> dkZxdx_col)
  {

    // Simon's approach:
    // // dkZxdx = kZx.rowwise() - x.transpose(); 
    // dkZxdx.array().rowwise() /= ell.transpose().array();
    // dkZxdx.array().rowwise() /= ell.transpose().array(); // TODO (Simon): this can be more efficient
    // dkZxdx.array().colwise() *= -kZx.array();

    // amarcovalle's approach:
      // Notes:
      // kZx_row is a scalar
      // dkZxdx_col is a matrix with size [D 1]:

      // Computation:
      Eigen::VectorXd ell2 = ell.array().square();
      dkZxdx_col = (x - Z_col).cwiseQuotient(ell2); // (x-Z)./(ell.^2)
      dkZxdx_col *= -kZx_row;
  }
  
  void CovSEard::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    for(size_t i = 0; i < input_dim; ++i) ell(i) = exp(loghyper(i));
    sf2 = exp(2*loghyper(input_dim));
  }
  
  std::string CovSEard::to_string()
  {
    return "CovSEard";
  }
}

