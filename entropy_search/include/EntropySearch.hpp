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
#include "insetup.hpp"
#include "JointMin.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include "DummyFunction.hpp"
#include "dH_MC_local.hpp"
#include "SampleBeliefLocations.hpp"
#include "MathTools.hpp"
#include "LoggingTools.hpp"
#include "ReadYamlParameters.hpp"

class EntropySearch{

public:
	EntropySearch(std::shared_ptr<INSetup> in);
	~EntropySearch();

	OutResults run(void);
	void posterior(	const Eigen::MatrixXd zb, const size_t Nrep,
									Eigen::VectorXd & Mb, Eigen::MatrixXd & Vb);
	void update_BestGuesses_list(	Eigen::VectorXd logP, Eigen::VectorXd lmb, Eigen::MatrixXd zb, 
																size_t ind_eval, double threshold, bool replace);
	Eigen::VectorXd FindGlobalGPMinimum(	bfgs_optimizer::ObjectiveFunction * GPmean, 
																				Eigen::MatrixXd BestGuesses, double xmin_s, double xmax_s);
	Eigen::MatrixXd get_warm_starts(double xmin_s, double xmax_s, size_t Nwarm_starts, size_t Nsub, 
																	size_t D, double S0, bfgs_optimizer::ObjectiveFunction * fun);

	/* Banners */
	void banner_init(libgp::GaussianProcess * gp);
	void banner_iter(size_t numiter);
	void banner_final(size_t MaxEvals, Eigen::MatrixXd global_min_esti_x, 
										Eigen::VectorXd global_min_esti_mux, Eigen::VectorXd global_min_esti_varx);

private:
	JointMin  bel_min;
	std::shared_ptr<INSetup> in;

	dH_MC_local dH;
	DummyFunction * EdH_objective;
	DummyFunction * GPmean;
	SampleBeliefLocations sample_bl;

	Eigen::MatrixXd BestGuesses;

  Eigen::VectorXd logP;
  Eigen::MatrixXd dlogPdMu;
  Eigen::MatrixXd dlogPdSigma;
  std::vector<Eigen::MatrixXd> dlogPdMudMu;

  Eigen::VectorXd Mb;
  Eigen::MatrixXd Vb; // This must be a matrix????

  Eigen::MatrixXd zb;
  Eigen::VectorXd lmb;

  Eigen::MatrixXd global_min_esti_x;
  Eigen::VectorXd global_min_esti_mux;
  Eigen::VectorXd global_min_esti_varx;

  bool first_time;

  Eigen::VectorXd xmin;
  Eigen::VectorXd xmax;

};

// Slice sampler (not worth to make a class just for this):
extern Eigen::VectorXd SliceShrinkRank_nolog(Eigen::VectorXd V, bfgs_optimizer::ObjectiveFunction * fun, double s0);