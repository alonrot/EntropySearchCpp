// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_matern5_iso.h"
#include <cmath>

namespace libgp
{
  
  CovMatern5iso::CovMatern5iso() {}
  
  CovMatern5iso::~CovMatern5iso() {}
  
  bool CovMatern5iso::init(int n)
  {
    input_dim = n;
    param_dim = 2;
    loghyper.resize(param_dim);
    loghyper.setZero();
    sqrt5 = sqrt(5);
    return true;
  }
  
  double CovMatern5iso::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    double z = ((x1-x2)*sqrt5/ell).norm();
    return sf2*exp(-z)*(1+z+z*z/3);
  }
  
  void CovMatern5iso::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    double z = ((x1-x2)*sqrt5/ell).norm();
    double k = sf2*exp(-z);
    double z_square = z*z;
    grad << k*(z_square + z_square*z)/3, 2*k*(1+z+z_square/3);
  }

  void CovMatern5iso::compute_dkdx(const Eigen::VectorXd & x, const Eigen::Ref<const Eigen::VectorXd> & kZx_row, const Eigen::Ref<const Eigen::MatrixXd> & Z_col, Eigen::Ref<Eigen::MatrixXd> dkZxdx_col)
  {
    // Input parameter kZx_row not used: the expression can be compressed so that we can get rid of it.

    // Computation:
    chi_dist = chi * (x-Z_col).norm();
    dkZxdx_col = (x-Z_col).array() * (-sf2 / 3 * std::exp(-chi_dist) * chi*chi * (1 + chi_dist));
  }
  
  void CovMatern5iso::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    ell = exp(loghyper(0));
    sf2 = exp(2*loghyper(1));
    chi = sqrt5 / ell;
  }
  
  std::string CovMatern5iso::to_string()
  {
    return "CovMatern5iso";
  }
  
}
