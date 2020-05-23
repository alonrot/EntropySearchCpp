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
#include "ExpectedImprovement.hpp"

#define DBL_EPSILON 2.22045e-16
#define INF_ 1e300

ExpectedImprovement::ExpectedImprovement(libgp::GaussianProcess * gp, Eigen::VectorXd xmin, Eigen::VectorXd xmax, bool invertsign) {

	this->gp = gp;
	this->xmin = xmin;
	this->xmax = xmax;
	// this->N = this->gp->get_sampleset_size();
// std::cout << "[DBG1]: @ExpectedImprovement::ExpectedImprovement()" << std::endl;
	this->my_sign = +1;
	if(invertsign)
		this->my_sign = -1;
// std::cout << "[DBG2]: @ExpectedImprovement::ExpectedImprovement()" << std::endl;
	// // Look for the minimum again:
	// std::vector<double> Y_vec = this->gp->get_sampleset()->y();
	// Eigen::VectorXd Y(this->N);
	// Y = Eigen::Map<Eigen::VectorXd>(Y_vec.data(),this->N);
	// this->fmin = Y.minCoeff(); // TODO: Replace this->fmin by min(M), where M is the posterior mean at the sample locations X
	this->fmin = ExpectedImprovement::update_fmin(gp);
// std::cout << "[DBG3]: @ExpectedImprovement::ExpectedImprovement()" << std::endl;
	// // Noise:
	// this->sn2 = (this->gp->covf().get_loghyper())(this->gp->covf().get_param_dim()-1); //amarcovalle
	// this->sn2 = exp(2 * this->sn2);
	this->sn2 = ExpectedImprovement::update_noise(gp);
	// std::cout << "[DBG4]: @ExpectedImprovement::ExpectedImprovement()" << std::endl;
}

double
ExpectedImprovement::update_fmin(libgp::GaussianProcess * gp){

	// Look for the minimum again:
	std::vector<double> Y_vec = gp->get_sampleset()->y();

	// Find the minimum of Y_vec:
	size_t N = gp->get_sampleset_size();
	if(N == 0)
		return 0.0;

	Eigen::VectorXd Y(N);
	Y = Eigen::Map<Eigen::VectorXd>(Y_vec.data(),N);
	double fmin = Y.minCoeff(); // TODO: Replace this->fmin by min(M), where M is the posterior mean at the sample locations X

	return fmin;

}

double
ExpectedImprovement::update_noise(libgp::GaussianProcess * gp){

	// Noise:
	double sn2 = (gp->covf().get_loghyper())(gp->covf().get_param_dim()-1);
	sn2 = exp(2 * sn2);

	return sn2;
}

void 
ExpectedImprovement::update_gp(libgp::GaussianProcess * gp){

	// Copy gp structure:
	this->gp = gp;

	// Update other internal variables:
	this->fmin = ExpectedImprovement::update_fmin(gp);
	this->sn2 = ExpectedImprovement::update_noise(gp);

}

std::string ExpectedImprovement::function_name(void) const {
	return "ExpectedImprovement";
}

void
ExpectedImprovement::evaluate(const Eigen::VectorXd& x, double* EI, Eigen::VectorXd* dEI) const{

	// Some parameters:
	int D = x.size();
	int N = this->gp->get_sampleset_size();
	libgp::SampleSet * sampleset = this->gp->get_sampleset(); // X in matlab (a.w.a Z around this c++ code)
	Eigen::VectorXd x_in = x;

	// Numerical zero:
	// double numerical_zero = std::exp(-500); // This number is way too low
	double numerical_zero = DBL_EPSILON;

	///////////////////////////////
	// Assign a super high value //
	///////////////////////////////
  if ( (x.array() < this->xmin.array()).any() || (x.array() > this->xmax.array()).any() )
  {
    *EI = numerical_zero;
    (*dEI).setZero();
    // std::cout << "[WARN]: EI(x), with x out of boundaries" << std::endl;
    return;
  }	


	// if isempty(GP.x)
	//     kxx  = feval(GP.covfunc{:},GP.hyp.cov,x);
	//     dkxx = feval(GP.covfunc_dx{:},GP.hyp.cov,x);
	//     dkxx = reshape(dkxx,[size(dkxx,2),size(dkxx,3)]);
	    
	//     s    = sqrt(kxx);
	//     dsdx = 0.5 / s * dkxx';
	    
	//     z    = fm / s;                          % assuming zero mean
	    
	//     phi   = exp(-0.5 * z.^2) ./ sqrt(2*pi); % faster than Matlabs normpdf
	//     Phi   = 0.5 * erfc(-z ./ sqrt(2));      % faster than Matlabs normcdf

	    
	//     f  = sign * (fm * Phi + s * phi);
	//     df = sign * (dsdx * phi);
	//     return;
	// end

  if(gp->get_sampleset_size() == 0){
  	// Assume fm = +Inf. Then, phi = 0, Phi = 1
  	// std::cout<< "@ExpectedImprovement: GP data set is empty. Returning Inf" << std::endl;
  	*EI = this->my_sign * INF_;
  	(*dEI).setZero();
  }


  // Construct GP wrapper, to obtain GPmean and GPmean derivative:
  // bfgs_optimizer::ObjectiveFunction * GPmean = new WrapperGP(this->gp,this->xmin,this->xmax);

  // Posterior mean and its derivative:
 //  double mu = 0.0;
 //  Eigen::VectorXd dmu = Eigen::VectorXd::Zero(D);
 //  GPmean->evaluate(x,&mu,&dmu); // This assumes prior zero mean (and thus, zero derivative)
	// std::cout << "mu = " << mu << std::endl;

 	// Compute posterior mean:
  double mu = gp->f(x.data());

  // Posterior std:
  double s = std::sqrt(this->gp->covf().get(x,x));

  // kXx  = feval(GP.covfunc{:},GP.hyp.cov,GP.x,x);
  Eigen::VectorXd kXx(N);
  gp->covf().getX(sampleset, x.transpose(),  kXx);

	// dkxX = feval(GP.covfunc_dx{:},GP.hyp.cov,x,GP.x);
  Eigen::MatrixXd dkXx(D,N);
  gp->covf().compute_dkdx(x, kXx, sampleset, dkXx); // Compute first d/dx{k(X,x)}, with size(dkXxdx) = [D N], and then transpose and change sign
  Eigen::MatrixXd dkxX(N,D);
  dkxX = -dkXx.transpose(); // dkxX [N D]

  // Compute cholesky factorization of the Gram matrix, i.e., chol(k(X,X))
  Eigen::MatrixXd cK(N,N); // In matlab, the noise is included, for this case
  cK = gp->getCholesky().topLeftCorner(N,N);

  // kXx  = feval(GP.covfunc{:},GP.hyp.cov,GP.x,x);
  gp->covf().getX(sampleset, x.transpose(),  kXx);
  Eigen::VectorXd cKkXx = gp->getCholesky().topLeftCorner(N, N).transpose().triangularView<Eigen::Upper>().solve(kXx);

  // Get dkxx:
  Eigen::VectorXd dkxx = Eigen::VectorXd::Zero(D);
  Eigen::VectorXd kxx(1);
  gp->covf().get(x.transpose(),x.transpose(),kxx);
  kxx = kxx.array() - this->sn2; // Remove the noise
  gp->covf().compute_dkdx(x, kxx, x, dkxx);

  // Posterior dstd:
	// for d = 1:D
	//     dsdx(d) = 0.5 / s * (dkxx(1,d) - 2 * dkxX(:,d)' * (GP.cK \ (GP.cK' \ kXx)));
	// end
  Eigen::VectorXd dsdx = Eigen::VectorXd::Zero(D);
  dsdx = (dkxx - 2 * dkxX.transpose() * cK.inverse() * cKkXx).array() * 0.5 / s; // Check this. Really dkxX.transpose(), or directly dkXx?

  // dmudx(d) = dmdx(d) + (dkxX(:,d)' * alpha);
  Eigen::VectorXd dmudx = Eigen::VectorXd::Zero(D);
  Eigen::VectorXd dmdx = Eigen::VectorXd::Zero(D); // Prior derivative of the prior mean. Since we assume prior mean = 0, this is also zero.
  dmudx = dmdx + dkXx * gp->get_alpha(); // Posterior derivative mean, not sure if dkXx(=-dkXx.transpose()), or dkXx.transpose()

	// phi   = exp(-0.5 * z.^2) ./ sqrt(2*pi) / s; % faster than Matlabs normpdf (amarcovalle: corrected; the final /s was missing)
  double phi = normpdf(this->fmin,mu,s);

	// Phi   = 0.5 * erfc(-z ./ sqrt(2));          % faster than Matlabs normcdf, equal to 0.5 * (1 + erf(-z ./ sqrt(2)));  
  double Phi = normcdf(this->fmin,mu,s);

	// f  = sign * ((fm - m) * Phi + s * phi);
  *EI = this->my_sign * ( (this->fmin - mu) * Phi  + s * phi ); // This assumes prior zero mean (and thus, zero derivative)

	// df = sign * (-dmudx    * Phi + dsdx * phi);
  *dEI = this->my_sign * ( -dmudx.array() * Phi   + dsdx.array() * phi ); // This assumes prior zero mean (and thus, zero derivative)

	// if sign * f < 0 
	//     f  = 0;
	//     df = zeros(size(x,2),1);
	//     display('Here?');
	// end

  // DBG:
  if(std::isinf(std::log(*EI)))
  	throw std::runtime_error("@ExpectedImprovement: std::log(*EI) = Inf");

	if(this->my_sign * (*EI) <= 0){
		// std::cout<< "@ExpectedImprovement: this->my_sign * (*EI) <= 0, returning zero" << std::endl;
		*EI = numerical_zero;
		(*dEI).setZero();
	}

}

double
ExpectedImprovement::normpdf(double muX_min, double mux, double sx) const{

	double z = (muX_min - mux) / sx;

	return (1.0/(sx*std::sqrt(2.0*M_PI))) * std::exp( -0.5 * std::pow(z,2) );

}

double ExpectedImprovement::normcdf(double muX_min, double mux, double sx) const {

	double z = (muX_min - mux) / sx;

   return 0.5 * std::erfc(-z * M_SQRT1_2);
}