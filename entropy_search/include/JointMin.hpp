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
#include <iostream>
#include <vector>

#define FIXNAN() unaryExpr(std::ptr_fun(fixNaN))

#define DBL_EPSILON 2.2204460492503131E-16


class JointMin{

public:
	JointMin();
	~JointMin();
	void joint_min(const Eigen::VectorXd Mu, const Eigen::MatrixXd Sigma,
	        Eigen::VectorXd & logP, Eigen::MatrixXd & dlogPdMu, 
	        Eigen::MatrixXd & dlogPdSigma, std::vector<Eigen::MatrixXd> & dlogPdMudMu);
	
private:
	static double fixNaN(double x);
	static double fixLogInf(double x);
	Eigen::VectorXd logsumexp(Eigen::MatrixXd x);
	double logdet(Eigen::MatrixXd M);
	double max(double a, double b);
	double log_relative_Gauss(double z, double &e, int &exit_flag);
	void   lt_factor(const int s, const int l, const Eigen::VectorXd M, const Eigen::MatrixXd V, const double mp, const double p, const double gam,
        Eigen::VectorXd &Mnew, Eigen::MatrixXd &Vnew, double &pnew, double &mpnew, double &logS, double &d);
	double min_factor(const Eigen::VectorXd Mu, const Eigen::MatrixXd Sigma, const int k, const double gam,
	        Eigen::VectorXd & dlogZdMu, Eigen::VectorXd & dlogZdSigma, Eigen::MatrixXd & dlogZdMudMu);

};