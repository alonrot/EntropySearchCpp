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
#include "SampleBeliefLocations.hpp"

SampleBeliefLocations::SampleBeliefLocations(libgp::GaussianProcess * gp, double xmin_s, double xmax_s, size_t Nb, size_t Nsub) {

  // Get parameters:
  this->D = gp->get_input_dim();

  // Assume the interval limits are all the same for all dimensions:
  this->xmin_s = xmin_s;
  this->xmax_s = xmax_s;
  this->xmin   = Eigen::VectorXd::Ones(this->D) * xmin_s;
  this->xmax   = Eigen::VectorXd::Ones(this->D) * xmax_s;

  // Initialization:
  this->S0 = 0.5 * (xmax - xmin).norm();
  // std::cout << "this->S0 = " << this->S0 << std::endl;
  this->Nb = Nb;
  this->Nsub = Nsub;
  this->N = gp->get_sampleset_size(); // N is the number of training points

  std::random_device rd;
  this->generator = std::mt19937(rd()); 
  this->uni_dis = std::uniform_real_distribution<double>(xmin_s,xmax_s);

  // Input function: EI
  this->EI_objective = new ExpectedImprovement(gp,this->xmin,this->xmax);
}

void
SampleBeliefLocations::update_gp(libgp::GaussianProcess * gp){

  this->EI_objective->update_gp(gp); 
  this->N = gp->get_sampleset_size();

}

void
SampleBeliefLocations::sample(Eigen::Ref<Eigen::MatrixXd> zbel, Eigen::Ref<Eigen::VectorXd> lmb, Eigen::MatrixXd BestGuesses){

  // Error checking:
  if(zbel.cols()!=(int)this->D)
    throw std::runtime_error("zbel must be NbxD, D is incorrect");

  if(zbel.rows()!=(int)this->Nb)
    throw std::runtime_error("zbel must be NbxD, Nb is incorrect");

  if(lmb.size()!=(int)this->Nb)
    throw std::runtime_error("lmb must be of dimenson Nb");    

  Eigen::VectorXd xx_in   = Eigen::VectorXd::Zero(this->D);
  Eigen::VectorXd xx_out  = Eigen::VectorXd::Zero(this->D);
  double expected_impro = 0.0;
  Eigen::VectorXd foo(this->D);
  size_t NBG = BestGuesses.rows();
  // std::cout << "NBG = " << NBG << std::endl;
  // std::cout << "@SampleBeliefLocations()" << std::endl;

  // Code the first-iteration special case:
  // If there is no data, or if NBG == 1, maybe?
  // if isempty(GP.x)  % if there are no previous evaluations, just sample uniformly
      
  //     D  = size(xmax,2);
  //     zb = bsxfun(@plus,bsxfun(@times,(xmax - xmin),rand(Nb,D)),xmin);
  //     mb = -log(prod(xmax-xmin)) * ones(Nb,1);
  //     return;
  // end

  // If there is no training data yet we cannot call EI, therefore, we return random.
  // If there is only one BestGuess in the list BestGuesses, this is because we are in the
  // first iteration of ES, and thus, it contains zeroes, and thus, we cannot use it as
  // a warm start for the slice sampler. Therefore, we return also random.
  // if(this->N == 0 || this->NBG == 1){
  if(this->N == 0){

    // Randomly sample zbel:
    MathTools::sample_unif_mat(this->xmin_s,this->xmax_s,zbel);

    // Compute lmb:
    lmb = -Eigen::VectorXd::Ones(this->Nb) * std::log((this->xmax - this->xmin).prod());

    // std::cout << "@SampleBeliefLocations: empty GP data set, returning log(Volume)" << std::endl;

    return;

  }

  // std::cout << "BestGuesses = " << std::endl << BestGuesses << std::endl;

  for(size_t i=0;i<this->Nb;++i){

    // Use BestGuesses as warm starts until we get out of them, but start the list from top
    if( i < NBG )
      xx_in = BestGuesses.row(NBG-1-i);
    else
      MathTools::sample_unif_vec(this->xmin_s,this->xmax_s,xx_in);

    // std::cout << "xx_in = " << xx_in.transpose() << std::endl;
    // Call slice sampler, and subsample:
    for(size_t k=0;k<this->Nsub;++k){
      
      xx_out = SliceShrinkRank_nolog(xx_in,this->EI_objective,this->S0);
      // std::cout << "xx_out = " << xx_out.transpose() << std::endl;

      // If they are the same, the slice sampler couldn't improve this sample.
      // Therefore, consider randomly sampling a new one:
      // Comparing xx_in == xx_out up to some precision parameter, which if not passed, is set to 0
      // If the outcome of the slice sampler is out of bounds, resample:
      if( xx_in.isApprox(xx_out) || MathTools::outside_dom_vec(this->xmin_s,this->xmax_s,xx_out) ){
        MathTools::sample_unif_vec(this->xmin_s,this->xmax_s,xx_out);
        // std::cout << "[WARNING]: @SampleBeliefLocations::sample() - Randomly sampling the input to the slice sampler" << std::endl;
        // std::cout << "[WARNING]: @SampleBeliefLocations::sample() - What about just projecting back to the domain?" << std::endl;
      }

      // Restart the sampling from the same point:
      xx_in = xx_out;
    }

    // Add to zbel:
    zbel.row(i) = xx_out;

    // Add to lmb:
    this->EI_objective->evaluate(xx_out,&expected_impro,&foo);
    // std::cout << "expected_impro = " << expected_impro << std::endl;
    lmb(i) = std::log(expected_impro);
  }

  return;
}