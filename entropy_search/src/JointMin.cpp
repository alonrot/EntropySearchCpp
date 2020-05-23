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
#include "JointMin.hpp"

JointMin::JointMin(){}
JointMin::~JointMin(){}

double
JointMin::fixNaN(double x) {

    return std::isnan(x) ? 0 : x;
}

double
JointMin::fixLogInf(double x) {

    return std::isinf(x) ? -500 : x;
}

Eigen::VectorXd 
JointMin::logsumexp(Eigen::MatrixXd x) {

  Eigen::VectorXd y = x.colwise().maxCoeff();

  Eigen::VectorXd s = (x.rowwise() - y.transpose()).array().exp().colwise().sum().log();

  return y + s;
}

double 
JointMin::logdet(Eigen::MatrixXd M) {

  Eigen::LLT<Eigen::MatrixXd> llt(M);

  // D = 2 * sum(log(diag(chol(M))));

  Eigen::VectorXd res = ((Eigen::MatrixXd) llt.matrixU()) // Calc chol())
          .diagonal().array().log() // Log of diagonal
          .colwise().sum(); // Calc sum of logs

  return 2 * res[0];
}

double 
JointMin::max(double a, double b) {
  return a < b ? b : a;
}

double 
JointMin::log_relative_Gauss(double z, double &e, int &exit_flag) {

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

    logphi = -0.5 * (z * z + log(M_PI * 2)); // Const function call gets optimized away
    logPhi = log(0.5 * erfc(-z * M_SQRT1_2));
    e = exp(logphi - logPhi);
    exit_flag = 0;
  }
  return logPhi;
}

void 
JointMin::lt_factor(const int s, const int l, const Eigen::VectorXd M, const Eigen::MatrixXd V, const double mp, const double p, const double gam,
        Eigen::VectorXd &Mnew, Eigen::MatrixXd &Vnew, double &pnew, double &mpnew, double &logS, double &d) {

  // rank 1 projected cavity parameters

  // modif: amarcovalle
    // Eigen::VectorXd Vc = Eigen::VectorXd(M.size());
    // Vc = V.col(l) - V.col(s);
    // Vc = Vc * M_SQRT1_2;
  // before
    Eigen::VectorXd Vc = (V.col(l) - V.col(s)) * M_SQRT1_2;
  // endmodif

  double cVc = (V(l, l) - 2 * V(s, l) + V(s, s)) / 2;
  double cM = (M(l) - M(s)) * M_SQRT1_2;

  double cVnic = JointMin::max(0, cVc / (1 - p * cVc));

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
      dp = JointMin::max(-p + DBL_EPSILON, gam * (pnew - p));
      dmp = JointMin::max(-mp + DBL_EPSILON, gam * (mpnew - mp));
      d = JointMin::max(dmp, dp); // for convergence measures

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
      d = JointMin::max(dmp, dp); // for convergence measures

      // project out to marginal
      Vnew = V - dp / (1 + dp * cVc) * (Vc * Vc.transpose());
      Mnew = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc;

      logS = 0;
      break;
  }
}

double 
JointMin::min_factor(const Eigen::VectorXd Mu, const Eigen::MatrixXd Sigma, const int k, const double gam,
        Eigen::VectorXd & dlogZdMu, Eigen::VectorXd & dlogZdSigma, Eigen::MatrixXd & dlogZdMudMu) {

  int D = Mu.size();

  double logZ;

  // messages (in natural parameters)
  Eigen::VectorXd logS = Eigen::VectorXd::Zero(D - 1); // normalization constant (determines zeroth moment)
  Eigen::VectorXd MP = Eigen::VectorXd::Zero(D - 1); // mean times precision (determines first moment)
  Eigen::VectorXd P = Eigen::VectorXd::Zero(D - 1); // precision (determines second moment)  

  // TODO: check if copy is really necessary here
  // marginal:
  Eigen::VectorXd M(Mu);
  Eigen::MatrixXd V(Sigma);

  double mpm;
  double s;
  double rSr;
  double dts;

  //Eigen::VectorXd dMdMu;
  //Eigen::VectorXd dMdSigma;
  //Eigen::VectorXd dVdSigma;
  Eigen::MatrixXd _dlogZdSigma;

  Eigen::MatrixXd R;
  Eigen::VectorXd r;

  Eigen::MatrixXd IRSR;
  Eigen::MatrixXd A;
  Eigen::MatrixXd A_;
  Eigen::VectorXd b;
  Eigen::VectorXd Ab;

  Eigen::VectorXd btA;

  Eigen::MatrixXd C;

  double pnew;
  double mpnew;

  double Diff = 0, diff = 0;

  int l, count = 0;

  // mvmin = Eigen::VectorXd(2);


  // amarcovalle: test
  // size_t Ntop = 5;
  // std::cout << "Mu = " << Mu.head(Ntop).transpose() << std::endl;
  // std::cout << "Sigma = " << Sigma.block(0,0,Ntop,Ntop) << std::endl;

  while (true) {

    count++;

    Diff = 0;

    for (int i = 0; i < D - 1; i++) {

      if (i < k)
        l = i;
      else
        l = i + 1;

      lt_factor(k, l, M, V, MP[i], P[i], gam, // IN
              M, V, pnew, mpnew, logS[i], diff); // OUT

      // Write back vector elements
      P[i] = pnew;
      MP[i] = mpnew;

      if (std::isnan(diff))
        goto done; // found impossible combination

      Diff = Diff + std::abs(diff);
    }

    if (count > 50) {
      #ifdef BUILD_MEX_DH
        mexPrintf("EP iteration ran over iteration limit. Stopped.\n");
      #else
        printf("EP iteration ran over iteration limit. Stopped.\n");
      #endif
      goto done;
    }
    if (Diff < 1.0e-3) {
      goto done;
    }
  }

done:

  if (std::isnan(diff)) {

    logZ = -INFINITY;
    dlogZdMu = Eigen::VectorXd::Zero(D);
    dlogZdSigma = Eigen::VectorXd::Zero(0.5 * (D * (D + 1)));
    dlogZdMudMu = Eigen::MatrixXd::Zero(D, D);
    //mvmin << Mu(k), Sigma(k, k);
    //dMdMu = Eigen::VectorXd::Zero(D);
    //dMdSigma = Eigen::VectorXd::Zero(0.5 * (D * (D + 1)));
    //dVdSigma = Eigen::VectorXd::Zero(0.5 * (D * (D + 1)));

  } else {

    // evaluate log Z:

    // C = eye(D) ./ sqrt(2); C(k,:) = -1/sqrt(2); C(:,k) = [];
    C = Eigen::MatrixXd::Zero(D, D - 1);
    for (int i = 0; i < D - 1; i++) {

      C(i + (i >= k), i) = M_SQRT1_2;
      C(k, i) = -M_SQRT1_2;
    }

    R = C.array().rowwise() * P.transpose().array().sqrt();
    r = (C.array().rowwise() * MP.transpose().array()).rowwise().sum();
    mpm = (MP.array() * MP.array() / P.array()).FIXNAN().sum();
    s = logS.sum();

    IRSR = R.transpose() * Sigma * R;
    IRSR.diagonal().array() += 1; // Add eye()

    rSr = r.dot(Sigma * r);

    A_ = R * IRSR.llt().solve(R.transpose());
    A = 0.5 * (A_.transpose() + A_); // ensure symmetry.

    b = (Mu + Sigma * r);
    Ab = A * b;

    dts = logdet(IRSR);
    logZ = 0.5 * (rSr - b.dot(Ab) - dts) + Mu.dot(r) + s - 0.5 * mpm;

    // if(k == 4){
    //   size_t Ntop = 5;
    //   std::cout << "rSr = " << rSr << std::endl;
    //   std::cout << "s = " << s << std::endl;
    //   std::cout << "dts = " << dts << std::endl;
    //   std::cout << "mpm = " << mpm << std::endl;
    //   std::cout << "b = " << b.head(Ntop).transpose() << std::endl;
    //   std::cout << "Ab = " << Ab.head(Ntop).transpose() << std::endl;
    //   std::cout << "Mu = " << Mu.head(Ntop).transpose() << std::endl;
    //   std::cout << "r = " << r.head(Ntop).transpose() << std::endl;
    // }

    if (true /*TODO: needs derivative? */) {

      dlogZdSigma = Eigen::VectorXd(0.5 * (D * (D + 1)));

      btA = b.transpose() * A;

      dlogZdMu = r - Ab;
      dlogZdMudMu = -A;

      _dlogZdSigma = -A - 2 * r * Ab.transpose() + r * r.transpose() + btA * Ab.transpose();


      Eigen::MatrixXd diag = _dlogZdSigma.diagonal().asDiagonal();

      // amarcovalle: Solved important bug here:
      // Before, the expression was:
      //
      //      _dlogZdSigma = 0.5 * (_dlogZdSigma + _dlogZdSigma.transpose() - diag);
      //
      // which suffers from aliasing, as explained here:
      // https://eigen.tuxfamily.org/dox/group__TopicAliasing.html
      // "In Eigen, aliasing refers to assignment statement in which the same matrix 
      // (or array or vector) appears on the left and on the right of the assignment operators.""
      // "Aliasing is harmless with coefficient-wise computations; this includes scalar 
      // multiplication and matrix or array addition."
      // However, in other cases, the function eval() should be used to correct for this.
      // The expression above is wrong, and it is solved as follows:
      _dlogZdSigma = 0.5 * (_dlogZdSigma + _dlogZdSigma.transpose().eval() - diag);

      // dlogZdSigma = dlogZdSigma(logical(triu(ones(D,D))));
      for (int x = 0, i = 0; x < D; x++) {

        for (int y = 0; y <= x; y++) {
          dlogZdSigma[i++] = _dlogZdSigma(y, x);
        }
      }
    }

    // // amarcovalle [DBG]:
    // if(k == 0 || k == 1){
    //   std::cout << "[DBG]: @min_factor(), k = " << k << ", _dlogZdSigma.block(0,0,8,8) = " << std::endl;
    //   std::cout << _dlogZdSigma.block(0,0,8,8) << std::endl;
    // }

    // // amarcovalle [DBG]:
    // if(k == 0){
    //   std::cout << "[DBG]: @min_factor() - dlogZdSigma.head(8) = " << std::endl;
    //   std::cout << dlogZdSigma.head(8) << std::endl;
    // }

  }

  return logZ;
}

void 
JointMin::joint_min(const Eigen::VectorXd Mu, const Eigen::MatrixXd Sigma,
        Eigen::VectorXd & logP, Eigen::MatrixXd & dlogPdMu, 
        Eigen::MatrixXd & dlogPdSigma, std::vector<Eigen::MatrixXd> & dlogPdMudMu) {



  // size_t Ntop = 5;
  // std::cout << "@joint_min() this->Mu = " << Mu.head(Ntop).transpose() << std::endl;
  // std::cout << "@joint_min() this->Sigma = " << std::endl << Sigma.block(0,0,Ntop,Ntop) << std::endl;

  // printf("\n");
  // printf("[DBG]: @joint_min(): 1\n");

  Eigen::VectorXd dlPdM;
  Eigen::VectorXd dlPdS;
  Eigen::MatrixXd dlPdMdM;

  double gam = 1;
  int D = Mu.size();

  // // added: amarcovalle
  //   dlPdM     = Eigen::VectorXd(D);
  //   dlPdS     = Eigen::VectorXd( 0.5 * D * (D+1) );
  //   dlPdMdM   = Eigen::MatrixXd(D , D);
  // // added end

  // printf("[DBG]: @joint_min(): Mu.size() = %i\n",D);

  Eigen::MatrixXd gg = Eigen::MatrixXd::Zero(D, D); // amarcovalle: This has to be initialized to zero
  // Eigen::MatrixXd gg = Eigen::MatrixXd(D, D); // amarcovalle: old way, leading to different results over iterations

  // commented: amarcovalle
    logP = Eigen::VectorXd(D);
  // commented end

  // commented: amarcovalle
    dlogPdMu = Eigen::MatrixXd(D, D);
    dlogPdSigma = Eigen::MatrixXd(D, D * (D + 1) / 2);
  // commented end

  // printf("[DBG]: @joint_min(): after things 1\n");

  // commented: amarcovalle
    // *dlogPdMudMu = new Eigen::MatrixXd[D]; // Create an array of matrizes
  // commented end

  // printf("[DBG]: @joint_min(): after things 2\n");

  for (int k = 0; k < D; k++) {

#ifdef DEBUG_PRINTF
    if (k % 10 == 0)
      DEBUG_PRINTF('#');
#endif
    
    logP(k) = min_factor(Mu, Sigma, k, gam, // IN
            dlPdM, dlPdS, dlPdMdM); // OUT

      dlogPdMu.row(k) = dlPdM;
      dlogPdSigma.row(k) = dlPdS;

      dlogPdMudMu[k] = dlPdMdM;

      // if( k == 3 ){
      //   size_t Ntop = 5;
      //   std::cout << "logP(k) = " << logP(k) << std::endl;
      //   std::cout << "dlPdM.head(Ntop) = " << dlPdM.head(Ntop).transpose() << std::endl;
      //   std::cout << "dlPdS.head(Ntop) = " << dlPdS.head(Ntop).transpose() << std::endl;
      //   std::cout << "dlogPdMudMu[k].block(0,0,Ntop,Ntop) = " << dlogPdMudMu[k].block(0,0,Ntop,Ntop) << std::endl;
      // }

  }

  // std::cout << "dlogPdMudMu[0].block(0,0,8,8) = " << std::endl << dlogPdMudMu[0].block(0,0,8,8) << std::endl; // Same over iterations
  // std::cout << "dlogPdMudMu[1].block(0,0,8,8) = " << std::endl << dlogPdMudMu[1].block(0,0,8,8) << std::endl; // Same over iterations


  // Sanity check for INF values
  logP = logP.unaryExpr(std::ptr_fun(fixLogInf));

  // re-normalize at the end, to smooth out numerical imbalances:
  double Z = logP.array().exp().sum(); // amarcovalle: Z is correctly computed

  Eigen::VectorXd Zm = (dlogPdMu.array().colwise() * logP.array().exp()).colwise().sum() /Z;
  Eigen::VectorXd Zs = (dlogPdSigma.array().colwise() * logP.array().exp()).colwise().sum() /Z;

  Eigen::MatrixXd Zij = Zm * Zm.transpose(); // amarcovalle: Zij is correctly computed

  for (int i = 0; i < D; i++) {

    for (int j = i; j < D; j++) {

      // Eigen::MatrixXd Mj = (*dlogPdMudMu)[j]; // amarcovalle: not needed
      // Eigen::MatrixXd Mj = dlogPdMudMu[j]; // amarcovalle: not needed

      for (int k = 0; k < D; k++) {
        // gg(i, j) -= (dlogPdMu(k, i) * dlogPdMu(k, j) + Mj(k, i)) * exp(logP(k)); // Robert's way, incorrect. amarcovalle: Using Mj(k, i) here is incorrect
        // gg(i, j) += ((dlogPdMu(k, i) * dlogPdMu(k, j) + Mj(k, i)) * exp(logP(k))) / Z; // Robert's way, incorrect. amarcovalle: modified to match with the code belo
        gg(i, j) += ( ( dlogPdMu(k, i) * dlogPdMu(k, j) + dlogPdMudMu[k](i,j) ) * exp(logP(k)) ) / Z; // Correct way
      }

      // Hesse Matrix is symmetric:
      gg(j, i) = gg(i, j); 
    }
  }

  Eigen::MatrixXd adds(D,D);
  adds = -gg+Zij;

  // std::cout << "[DBG1]: @joint_min() - gg.block(0,0,8,8) = " << std::endl; //  Different over iterations
  // std::cout << gg.block(0,0,8,8) << std::endl;

  // std::cout << "[DBG1]: @joint_min() - Zij.block(0,0,8,8) = " << std::endl; //  Same over iterations
  // std::cout << Zij.block(0,0,8,8) << std::endl;

  // std::cout << "[DBG1]: @joint_min() - dlogPdMu.block(0,0,8,8) = " << std::endl; //  Same over iterations
  // std::cout << dlogPdMu.block(0,0,8,8) << std::endl;

  // std::cout << "[DBG1]: @joint_min() - logP.transpose() = " << std::endl; //  Same over iterations
  // std::cout << logP.transpose() << std::endl;

  for (int i = 0; i < D; i++) {
    // (*dlogPdMudMu)[i].array() *= gg.array(); // Robert's way (this is a bug)
    dlogPdMudMu[i].array() += adds.array();
  }

  // std::cout << "dlogPdMudMu[0].block(0,0,8,8) = " << std::endl << dlogPdMudMu[0].block(0,0,8,8) << std::endl;
  // std::cout << "(*dlogPdMudMu)[1].block(0,0,8,8) = " << std::endl << (*dlogPdMudMu)[1].block(0,0,8,8) << std::endl;


  dlogPdMu = dlogPdMu.array().rowwise() - Zm.transpose().array();
  dlogPdSigma = dlogPdSigma.array().rowwise() - Zs.transpose().array();
  
  logP = logP.array() - logsumexp(logP)(0, 0);

}
