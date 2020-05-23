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
#include "DummyFunction.hpp"

#define DBL_EPSILON 2.22045e-16
#define INF_ 1e300

DummyFunctionChild::DummyFunctionChild(Eigen::VectorXd param){

	// Copy parameters:
	this->param = param;

	// Dimension:
	this->D = param.size();

}

void
DummyFunctionChild::evaluate(const Eigen::VectorXd& x, double* f, Eigen::VectorXd* df) const {

	Eigen::VectorXd x2 = x.array().square();
	*f = x2.dot(this->param);

	// f = x.array().square() * (this->param.array());
	*df = 2 * (x.array() * this->param.array());

}

std::string
DummyFunctionChild::function_name(void) const {
	return "DummyFunctionChild";
}

void
DummyFunctionChild::update_something(Eigen::VectorXd new_param){
	this->param = new_param;
}

std::string
WrapperGP::function_name(void) const {
	return "WrapperGP";
}

void
WrapperGP::evaluate(const Eigen::VectorXd& x, double* f, Eigen::VectorXd* df) const{

	// Some parameters:
	size_t D = x.size();
	int N = this->gp->get_sampleset_size();
	libgp::SampleSet * sampleset = this->gp->get_sampleset(); // X in matlab (a.w.a Z around this c++ code)
	Eigen::VectorXd x_in = x;

	///////////////////////////////
	// Assign a super high value //
	///////////////////////////////
	  if ( (x.array() < this->xmin.array()).any() || (x.array() > this->xmax.array()).any() )
	  {
	    // TODO: get here the true EPS number. Look at mxgeteps.c
	    *f = INF_; // Inf
	    *df = Eigen::VectorXd(x.size()).setZero();
	    // std::cout << "[WARN]: f(x), with x out of boundaries" << std::endl;
	    return;
	  }	

	//////////////////////////////////////////////////////////////////
	// Project back to the boundaries, in case x is out of the domain
	//////////////////////////////////////////////////////////////////
		// bool limits_hit = false;
		// for(size_t i=0;i<D;++i){

		// 	if(x_in(i) < this->xmin(i)){
		// 		x_in(i) = this->xmin(i);
		// 		limits_hit = true;
		// 	}

		// 	if(x_in(i) > this->xmax(i)){
		// 		x_in(i) = this->xmax(i);
		// 		limits_hit = true;
		// 	}
		// }
		// if(limits_hit){
	 //    std::cout << "[WARN]: f(x), with x out of boundaries" << std::endl;
	 //    std::cout << "x_in = " << x_in << std::endl;
		// }

	// Compute GP posterior mean:
	*f = this->gp->f(x_in.data());

	// Kernel derivative:
		// kx  = feval(GP.covfunc{:},GP.hyp.cov,x,GP.x); // This is a vector because x is a single point
		Eigen::VectorXd kXx(N); // Same as kx in matlab
		(this->gp->covf()).getX(sampleset,x_in.transpose(),kXx); // kXx is here a column vector [N 1], because x is a single point

		// dkx = feval(GP.covfunc_dx{:},GP.hyp.cov,x,GP.x); // this is a vector of vectors, size(dkx) = [1,N,D] (D row vectors)
    Eigen::MatrixXd dkXx(D,kXx.size()); // second (by amarcovalle)
    // Eigen::MatrixXd dkXx(kXx.size(), D); // first (by Simon)
    this->gp->covf().compute_dkdx(x_in, kXx, sampleset, dkXx); // Tested in test_ker_derivative.cpp

  // GP Mean derivative:
	  // Get alpha only once, as it has to be used several times to compute df:
		Eigen::VectorXd alpha = this->gp->get_alpha();

		// Get GP mean derivative:
    Eigen::VectorXd df_aux(D); 
    for(size_t i=0;i<D;++i)
    {
    	(*df)(i) = alpha.dot(dkXx.row(i));
    }


	// amarcovalle: Simon included GP derivatives w.r.t the input dimensions, only for the SEard kernel
	// df = this->gp->



    //kbx  = feval(GP.covfunc{:},GP.hyp.cov,zbel,x);
    // Eigen::VectorXd kbx(zbel.rows());
    // (gp->covf()).get(zbel, x.transpose(), kbx); x is a VectorXd, by definition column; therefore, we need to transpose it.

    // //% derivatives of kernel values
    // //dkxb = feval(GP.covfunc_dx{:},GP.hyp.cov,x,zbel);
    // Eigen::MatrixXd dkbx(kbx.size(), D);
    // gp->covf().compute_dkdx(x, kbx, zbel, dkbx);

    // //dkxx = feval(GP.covfunc_dx{:},GP.hyp.cov,x);
    // Eigen::VectorXd dkxx(D);
    // gp->covf().compute_dkdx(x, kxx, x, dkxx);



		// This needs to be parsed:
			// function [f,df] = GPmeanderiv(x,GP,alpha)


				// // dkx = feval(GP.covfunc_dx{:},GP.hyp.cov,x,GP.x); // this is probably a vector of vectors, maybe size(dkx) = [1,N,D] (D row vectors)
		  //   // Eigen::MatrixXd dkXx(kXx.size(), D); // first
		  //   Eigen::MatrixXd dkXx(D,kXx.size()); // second
		  //   std::cout << "dkXx = " << dkXx << std::endl;
				// std::cout << "[DBG1]: @WrapperGP::evaluate() " << std::endl;
				// libgp::SampleSet * sampleset = new libgp::SampleSet(D);
				// sampleset = this->gp->get_sampleset();
		  //   this->gp->covf().compute_dkdx(x, kXx, sampleset, dkXx); // This does not work
				// std::cout << "[DBG2]: @WrapperGP::evaluate() - this->gp->covf().compute_dkdx(x, kx, this->gp->get_sampleset(), dkXx);" << std::endl;
				// std::cout << "[DBG3]: dkXx = " << std::endl << dkXx << std::endl;
		  //   Eigen::VectorXd df_aux(D); 
		  //   for(size_t i=0;i<D;++i)
		  //   {
		  //   	df_aux(i) = alpha.dot(dkXx.row(i));
		  //   	// df(i) = (kbx.col(i)).dot(alpha); // maybe this?
		  //   }
				// std::cout << "[DBG4]: @WrapperGP::evaluate() " << std::endl;
		  //   *df = df_aux;
		  //   std::cout << "[DBG6]: @WrapperGP::evaluate(), df_aux = " << df_aux << std::endl;
		  //   std::cout << "[DBG6]: @WrapperGP::evaluate(), *df = " << *df << std::endl;

				// f  = kx * alpha;
				// df = zeros(size(x,2),1);
				// for d = 1:size(x,2)
				//     df(d) = dkx(:,:,d) * alpha;
				// end

			// end
		// Since x is a single point, we can use (I think, but not sure cuz size(x,2)=D~=1) Simon's functions (unprepared for when x is a collection of points rowwise)
				// std::cout << "kXx = " << kXx << std::endl;

}

void
WrapperGP::update_gp(libgp::GaussianProcess * gp){

	this->gp = gp;

}

DummyFunctionChildBFGS::DummyFunctionChildBFGS(Eigen::VectorXd param, double xmin_s, double xmax_s){

	// Copy parameters:
	this->param = param;

	// Dimension:
	this->D = param.size();

	// Range:
	this->xmin = Eigen::VectorXd::Ones(this->D) * xmin_s;
	this->xmax = Eigen::VectorXd::Ones(this->D) * xmax_s;

}

void
DummyFunctionChildBFGS::evaluate(const Eigen::VectorXd& x, double* f, Eigen::VectorXd* df) const {

	//////////////////////////
	// Assign a zero values //
	//////////////////////////
  if ( (x.array() < this->xmin.array()).any() || (x.array() > this->xmax.array()).any() )
  {
    *f = 0.0;
    *df = Eigen::VectorXd(x.size()).setZero();
    // std::cout << "[WARN]: f(x), with x out of boundaries" << std::endl;
    return;
  }

	Eigen::VectorXd x2 = x.array().square();
	*f = x2.dot(this->param);

	*df = 2 * (x.array() * this->param.array());

}

void
WrapperEdH::evaluate(const Eigen::VectorXd& x, double* function_value, Eigen::VectorXd* derivative) const{

	this->dHdx->dHdx_local(x,*function_value,*derivative);

}

void
WrapperEdH::change_output_sign(bool invert_sign){

	this->dHdx->change_sign(invert_sign);

}

void 
WrapperEdH::update_variables( Eigen::MatrixXd zbel,
				                      Eigen::VectorXd logP,
				                      Eigen::MatrixXd dlogPdM,
				                      Eigen::MatrixXd dlogPdV,
				                      std::vector<Eigen::MatrixXd> ddlogPdMdM,
				                      Eigen::VectorXd lmb,
				                      libgp::GaussianProcess * gp){

	// Update variables of dH_MC_local:
	this->dHdx->update_variables(zbel,logP,dlogPdM,dlogPdV,ddlogPdMdM,lmb,gp);
}


std::string
WrapperEdH::function_name(void) const {
	return "WrapperEdH";
}



