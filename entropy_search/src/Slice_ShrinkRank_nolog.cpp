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
#include <cmath>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include "DummyFunction.hpp"
#include "optimizer/objective_function.h"

// For Debugging:
    #include <chrono>
    #include <thread>

#define DBL_EPSILON 2.22045e-16

// #include "matrix.h"

/* Note from amarcovalle */
// This function has not been tested yet. A sanity check would be to compare its results with those from Slice_ShrinkRank_nolog.m.
// However, creating a mex wrapper for this function has already been tried (in Slice_ShrinkRank_nolog_cpp.cpp) without success,
// as it invoves passing a matlab function handle, and reading from that handle inside this c++, which is challenging.
// Therefore, the aim for this function to be tested is to realize numerical benchmarks and compare the results with
// those obtained from Slice_ShrinkRank_nolog.m/

static inline Eigen::VectorXd ProjNullSpace(Eigen::MatrixXd J, Eigen::VectorXd v, size_t width) {

  // width > 0 ? v - J * J' * v : v)

  // Eigen::MatrixXd t = J.block(0, 0, J.rows(), width);
  Eigen::MatrixXd t = J.leftCols(width); // amarcovalle: more intuitive

  if (width > 0) {
    return v - t * t.transpose() * v;
  } else {
    return v;
  }
}

Eigen::VectorXd SliceShrinkRank_nolog(Eigen::VectorXd V, bfgs_optimizer::ObjectiveFunction * fun, double s0) {

  // TODO: input V as const



  // Initialize random generator
    // std::default_random_engine generator; // Robert
    // double number = distribution(generator); // Robert
    // Same as MATLAB's default random generator for rand(), randi() and randn(), which is Mersenne Twister with seed 0:
    std::random_device rd;
    std::mt19937 generator(rd()); 
    std::normal_distribution<double> distribution(0.0, 1.0);
    std::uniform_real_distribution<double> uniform_dist(0,1);

    
  size_t dim = V.size(); // Dimensions

  // [f,~]= P(xx');
  double f;
  Eigen::VectorXd df = Eigen::VectorXd(dim); // Not used, actually
  // fun->evaluate_function(V,f,df); // with DummyFunction * fun
  fun->evaluate(V,&f,&df); // with bfgs_optimizer::ObjectiveFunction * fun

  // Special case: when f is zero, assign a small non-zero value
  if(f == 0){
    // std::cout << "[DBG]: @SliceShrinkRank_nolog() - Numerical zero in f" << std::endl;
    f = std::exp(-500); // Numerical zero, such that log(f) = -500;
  }


  // // amarcovalle: test
  // if(V(0) == 0.6){
  //   double ff;
  //   Eigen::VectorXd dff = Eigen::VectorXd(dim); // Not used, actually
  //   fun->evaluate(V,&ff,&dff); // with bfgs_optimizer::ObjectiveFunction * fun
  //   std::cout << "@SliceShrinkRank_nolog() ff = " << ff << std::endl;
  //   std::cout << "@SliceShrinkRank_nolog() dff = " << dff << std::endl;
  // }


  // f = 0.5;

  // MATLAB:
  // logf = log(f);
  // logy = log(rand()) + logf;

  // Treat exception: this slice sampler is not meant for f < 0:
  if (f < 0){
    // std::cout << "@SliceShrinkRank_nolog() fun(V) < 0 " << std::endl;
    // std::cout << "@SliceShrinkRank_nolog() return V " << std::endl;
    return V;
  }

  // c++:
  double logf = log(f);
  // double logy = log(0.5) + logf; // Robert
  double logy = log(uniform_dist(generator)) + logf; // amarcovalle


  const double theta = 0.95; // TODO: Consider decreasing this factor, to reduce faster the step size
  // const double theta = 0.8; // TODO: Consider decreasing this factor, to reduce faster the step size

  double s = s0;
  double sum = 1.0 / s;

  double threshold_stop = DBL_EPSILON;

  // Used matrix width
  size_t cUsed = 0; // c
  size_t JUsed = 0; // J
  size_t sUsed = 1; // invS size

  // Eigen::MatrixXd c = Eigen::MatrixXd(dim, dim);
  // Eigen::Matrix<double,3,Eigen::Dynamic> c; // amarcovalle: this grows dynamically in columns, so changed accordingly


  // This number can probably be computed as an upper bound, that responds to s0*(theta)^Nmax_iters < DBL_EPSILON
  // A conservative factor 1.5 is introduced to overestimate Nmax_iters
  size_t Nmax_iters = std::ceil(2 * std::log(threshold_stop/s0) / std::log(theta));
  Eigen::MatrixXd c = Eigen::MatrixXd::Zero(dim,Nmax_iters);

  Eigen::MatrixXd J = Eigen::MatrixXd(dim, dim);
  // Eigen::VectorXd invS = Eigen::VectorXd(dim);
  Eigen::VectorXd invS; // amarcovalle: this grows dynamically

  // invS.conservativeResize(sUsed,Eigen::NoChange);
  invS.conservativeResize(sUsed);
  invS(0) = sum; // Initialize first slot with 1/s0

  // std::cout << "invS = " << std::endl << invS << std::endl;

  double fk, logfk, sx;

  Eigen::VectorXd mx;
  Eigen::VectorXd dlogfk;
  Eigen::VectorXd dfk;
  Eigen::VectorXd g;
  Eigen::VectorXd xk;

  Eigen::VectorXd random = Eigen::VectorXd(dim);

  while (true) {

    if(sUsed >= Nmax_iters)
      throw std::runtime_error("@SliceShrinkRank_nolog: invS grew till its maximum size");

    // random.setRandom();
    for(int i = 0;i < random.size();i++)
      random(i) = distribution(generator);

    // c.block(0, cUsed++, dim, 1) = ProjNullSpace(J, V + s * random, JUsed); // Robert: c++ indexing, c(0:dim-1,cUsed:cUsed+1-1), with cUsed = val;
    c.col(cUsed) = ProjNullSpace(J, V + s * random, JUsed); // amarcovalle (avoid using block(), as it is nonituitive, and only columns are changing)

    sx = 1.0 / sum;

    // mx = sx * ((c.block(0, 0, dim, cUsed).colwise() - V) * invS.head(sUsed).asDiagonal()).rowwise().sum(); // Robert:  c++ indexing, c(0:dim-1,0:cUsed-1), with cUsed = val+1;
    mx = sx * ((c.leftCols(cUsed+1).colwise() - V) * invS.head(sUsed).asDiagonal()).rowwise().sum(); // amarcovalle (avoid using block(), as it is nonituitive, and only columns are changing)

    ++cUsed; // amarcovalle: This should be done at the end (not compatible with Robert's way)


    // random.setRandom();
    for(int i = 0;i < random.size();i++)
      random(i) = distribution(generator);

    xk = V + ProjNullSpace(J, mx + sx * random, JUsed);

    
    // std::cout << "mx = " << mx << std::endl;
    // std::cout << "random = " << random << std::endl;
    // std::cout << "sx = " << sx << std::endl;
    // std::cout << "JUsed = " << JUsed << std::endl;
    // std::cout << "sx = " << sx << std::endl;
    // std::cout << "sum = " << sum << std::endl;
    // std::cout << "s0 = " << s0 << std::endl;

    // [f,df]= P(xx');
    dfk = Eigen::VectorXd(dim);
    // fun->evaluate_function(xk,fk,dfk);
    fun->evaluate(xk,&fk,&dfk);
    // std::cout << "xk = " << xk << std::endl;
    // std::cout << "fk = " << fk << std::endl;
    // std::cout << "dfk = " << std::endl << dfk << std::endl;

    if(fk == 0){
      // std::cout << "[DBG]: @SliceShrinkRank_nolog() - Numerical zero in fk" << std::endl;
      fk = std::exp(-500); // Numerical zero, so that log(fk) = -500
    }



    // fk = 0.5;
    logfk = log(fk);
    dlogfk = dfk / fk;



    // if(logfk == INFINITY || logfk == -INFINITY){
    //   std::cout << "logfk = " << logfk << std::endl;
    //   std::cout << "fk = " << fk << std::endl;
    //   std::cout << "dfk = " << dfk << std::endl;
    //   std::cout << "dlogfk = " << std::endl << dlogfk << std::endl;
    //   std::cout << "logy = " << logy << std::endl;
    //   std::cout << "g = " << g << std::endl;
    //   std::cout << "[DBG]: @SliceShrinkRank_nolog() - Paused for debugging" <<std::endl;
    //   std::chrono::seconds dura(1);
    //   std::this_thread::sleep_for(dura);
    // }


    if (logfk > logy) { // accept
      // std::cout << "Accepted" << std::endl;
      return xk;
    } 
    else { // shrink

      // std::cout << "Shrink" << std::endl;

      g = ProjNullSpace(J, dlogfk, JUsed); // Returns zero if dlogfk = zero vector

      // amarcovalle: added here the condition g.norm() != 0, which will happen
      // if dfk = zero vector
      if (JUsed < dim - 1 && g.transpose() * dlogfk > 0.5 * g.norm() * dlogfk.norm() && g.norm() != 0) { 

        J.block(0, JUsed++, dim, 1) = g / g.norm();

        // s = s;

      } 
      else {

        s = theta * s;

        // TODO (amarcovalle): get here the true EPS number. Look at mxgeteps.c
        // double DBL_EPSILON = 2.22045e-16;

        if (s < threshold_stop) { // TODO: consider increasing threshold_stop if the slice sampler takes too long

          std::cout << "Bug found: contracted down to zero step size, still not accepted, iter = " << sUsed << "(max: " << Nmax_iters << ")" << std::endl;

          return V;

        }
      }
      sum = sum + 1.0 / s;

      // sUsed++; invS.conservativeResize(sUsed,Eigen::NoChange); invS(sUsed-1) = 1.0 / s; // amarcovalle: push back
      sUsed++; invS.conservativeResize(sUsed); invS(sUsed-1) = 1.0 / s; // amarcovalle: push back
      // invS(sUsed++) = 1.0 / s; Doesn't work, as invS grows dynamically

      // std::cout << "[DBG]: invS = " << std::endl << invS << std::endl;
      // std::cout << "[DBG]: sUsed = " << std::endl << sUsed << std::endl;
      // std::cout << "[DBG]: cUsed = " << cUsed << std::endl;

      if(logfk == INFINITY || logfk == -INFINITY){
        std::cout << "logf = " << logf << std::endl;
        std::cout << "f = " << f << std::endl;
        std::cout << "logy = " << logy << std::endl;
        std::cout << "logfk = " << logfk << std::endl;
        std::cout << "fk = " << fk << std::endl;
        std::cout << "dfk = " << dfk << std::endl;
        std::cout << "dlogfk = " << std::endl << dlogfk << std::endl;
        std::cout << "logy = " << logy << std::endl;
        std::cout << "g = " << g << std::endl;
        std::cout << "J = " << J << std::endl;
        std::cout << "s = " << s << std::endl;
        std::cout << "sum = " << sum << std::endl;
        std::cout << "invS = " << invS.transpose() << std::endl;
        std::cout << "[DBG]: @SliceShrinkRank_nolog() - Paused for debugging" <<std::endl;
        std::chrono::seconds dura(1);
        std::this_thread::sleep_for(dura);
      }


    }
  }
}

