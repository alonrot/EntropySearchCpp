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
#include <Eigen/Dense>
#include <cmath>
#include <limits>

#define DBL_EPSILON 2.2204460492503131e-16

static inline double max(double a, double b) {
  return a < b ? b : a;
}

static inline double log_relative_Gauss(double z, double &e, int &exit_flag) {

  double logphi, logPhi;

  if (z < -6) {

    e = 1;
    logPhi = -1.0e12;
    exit_flag = -1;

  } else if (z > 6) {

    e = 0;
    logPhi = 0;
    exit_flag = 1;

  } else {

    logphi = -0.5 * (z * z + log(M_PI * 2));
    logPhi = log(0.5 * erfc(-z / M_SQRT2));
    e = exp(logphi - logPhi);
    exit_flag = 0;

  }
  return logPhi;
}

void lt_factor(int s, int l, Eigen::VectorXd M, Eigen::MatrixXd V, double mp, double p, double gam,
        Eigen::VectorXd &Mnew, Eigen::MatrixXd &Vnew, double &pnew, double &mpnew, double &logS, double &d) {

  // rank 1 projected cavity parameters
  Eigen::VectorXd Vc = (V.col(l) - V.col(s)) / M_SQRT2;
  double cVc = (V(l, l) - 2 * V(s, l) + V(s, s)) / 2;
  double cM = (M(l) - M(s)) / M_SQRT2;

  double cVnic = max(0, cVc / (1 - p * cVc));

  double cmni = cM + cVnic * (p * cM - mp);

  // rank 1 calculation: step factor
  double z = cmni / sqrt(cVnic);

  double e;
  int exit_flag;
  double lP = log_relative_Gauss(z, e, exit_flag);

  double alpha, beta, r, dp, dmp;

  switch (exit_flag) {

    case 0:

      alpha = e / sqrt(cVnic);
      beta = alpha * (alpha * cVnic + cmni);
      r = beta / (1 - beta);

      // new message
      pnew = r / cVnic;
      mpnew = r * (alpha + cmni / cVnic) + alpha;

      // update terms
      dp = max(-p + DBL_EPSILON, gam * (pnew - p));
      dmp = max(-mp + DBL_EPSILON, gam * (mpnew - mp));
      d = max(dmp, dp); // for convergence measures

      pnew = p + dp;
      mpnew = mp + dmp;

      // project out to marginal
      Vnew = V - dp / (1 + dp * cVc) * (Vc * Vc.transpose());
      Mnew = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc;

      // normalization constant
      //logS  = lP - 0.5 * (log(beta) - log(pnew)) + (alpha * alpha) / (2*beta);

      // there is a problem here, when z is very large
      logS = lP - 0.5 * (log(beta) - log(pnew) - log(cVnic)) + (alpha * alpha) / (2 * beta) * cVnic;

      break;

    case -1: // impossible combination

      d = NAN;

      //Mnew = 0;
      //Vnew = 0;

      pnew = 0;
      mpnew = 0;
      logS = -INFINITY;
      break;

    case 1: // uninformative message

      pnew = 0;
      mpnew = 0;

      // update terms
      dp = -p; // at worst, remove message
      dmp = -mp;
      d = max(dmp, dp); // for convergence measures

      // project out to marginal
      Vnew = V - dp / (1 + dp * cVc) * (Vc * Vc.transpose());
      Mnew = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc;

      logS = 0;
      break;
  }
}

