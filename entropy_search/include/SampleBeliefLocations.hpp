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
#ifndef __SAMPLE_BELIEF_LOCATIONS_H__
#define __SAMPLE_BELIEF_LOCATIONS_H__

#include <cmath>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include "DummyFunction.hpp"
#include "ExpectedImprovement.hpp"

class SampleBeliefLocations {

  public:
    SampleBeliefLocations(){};
    SampleBeliefLocations(libgp::GaussianProcess * gp, double xmin_s, double xmax_s, size_t Nb, size_t Nsub);
    virtual ~SampleBeliefLocations() {}
    void sample(Eigen::Ref<Eigen::MatrixXd> zbel, Eigen::Ref<Eigen::VectorXd> lmb, Eigen::MatrixXd BestGuesses);
    void update_gp(libgp::GaussianProcess * gp);

  private:
  	double S0;
  	size_t D;
  	size_t Nb;
    size_t Nsub;
    size_t N;
    double xmin_s; // Scalar limits for the domain, assuming it's squared. 
    double xmax_s; // Scalar limits for the domain, assuming it's squared.

    // Domain limits:
    Eigen::VectorXd xmin;
    Eigen::VectorXd xmax;
    
  	// Same generator as in Slice_ShrinkRank_nolog.cpp:
  	std::mt19937 generator;
  	std::uniform_real_distribution<double> uni_dis;
    // bfgs_optimizer::ObjectiveFunction * EI_objective;
    DummyFunction * EI_objective;

};

// Slice sampler (not worth to make a class just for this):
extern Eigen::VectorXd SliceShrinkRank_nolog(Eigen::VectorXd V, bfgs_optimizer::ObjectiveFunction * fun, double s0);

#endif /* __SAMPLE_BELIEF_LOCATIONS_H__ */