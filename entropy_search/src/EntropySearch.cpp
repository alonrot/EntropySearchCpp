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
#include "EntropySearch.hpp"

// For Debugging:
    #include <chrono>
    #include <thread>

EntropySearch::EntropySearch(std::shared_ptr<INSetup> in)
{

	// Initialize pointer to input structure:
	this->in = in;

	// Assume the boundaries of the domain are equal for all dimensions:
	this->xmin 					= Eigen::VectorXd::Ones(this->in->Dim) * this->in->xmin_s;
	this->xmax 					= Eigen::VectorXd::Ones(this->in->Dim) * this->in->xmax_s;

	// Construct the needed classes:
	this->bel_min       = JointMin();

	// Locations of the representers:
	this->zb 						= Eigen::MatrixXd::Zero(in->Nrepresenters,in->Dim);
	this->lmb 					= Eigen::VectorXd::Zero(in->Nrepresenters);

	// Mean and variance on a grid on the representer points:
	this->Mb 						= Eigen::VectorXd::Zero(in->Nrepresenters);
	this->Vb 						= Eigen::MatrixXd::Zero(in->Nrepresenters,in->Nrepresenters);

	// log(p_min) at the representer points and derivatives:
	this->logP 					= Eigen::VectorXd::Zero(in->Nrepresenters);
	this->dlogPdMu      = Eigen::MatrixXd::Zero(in->Nrepresenters,in->Nrepresenters);
	this->dlogPdSigma   = Eigen::MatrixXd::Zero(in->Nrepresenters,0.5*(in->Nrepresenters)*(in->Nrepresenters+1));
	this->dlogPdMudMu 	= std::vector<Eigen::MatrixXd>(this->in->Nrepresenters);
	for(size_t i=0;i<this->in->Nrepresenters;++i)
		this->dlogPdMudMu[i] = Eigen::MatrixXd::Zero(this->in->Nrepresenters,this->in->Nrepresenters);

	// Set the first BestGuess on the list to random:
	this->BestGuesses   = Eigen::MatrixXd(1,in->Dim);
	MathTools::sample_unif_mat(this->in->xmin_s,this->in->xmax_s,this->BestGuesses);

	this->first_time = true;

	// Sample belief locations:
	this->sample_bl = SampleBeliefLocations(this->in->gp,this->in->xmin_s,this->in->xmax_s,this->in->Nrepresenters,this->in->Nsubsamples);

	// TODO: The next pointers cannot be made smart pointers for compatibility reasons with the optimizer bfgs_optimizer::BFGS
	// GPmean function:
	this->GPmean = new WrapperGP(this->in->gp,this->xmin,this->xmax);

	// Change in entropy:
	this->dH = dH_MC_local(this->zb,this->logP,this->dlogPdMu,this->dlogPdSigma,this->dlogPdMudMu,this->in->T,lmb,this->xmin,this->xmax,true,this->in->gp); // originally false
	this->EdH_objective = new WrapperEdH(&this->dH);

	// Logging out variables:
	this->global_min_esti_x 		= Eigen::MatrixXd::Zero(this->in->MaxEval,this->in->Dim);
	this->global_min_esti_mux 	= Eigen::VectorXd::Zero(this->in->MaxEval);
	this->global_min_esti_varx 	= Eigen::VectorXd::Zero(this->in->MaxEval);

}

EntropySearch::~EntropySearch()
{
	delete this->GPmean;
	delete this->EdH_objective;
}

void 
EntropySearch::posterior(	const Eigen::MatrixXd zb, const size_t Nrep, 
													Eigen::VectorXd & Mb, Eigen::MatrixXd & Vb) {
	
	// Sample set size:
	size_t Nss = this->in->gp->get_sampleset_size();

	if( Nss == 0 ){

		// Compte posterior mean:
		for(size_t i = 0 ; i < Nrep ; ++i){
				Mb(i) = this->in->gp->f(zb.row(i).data()); // This updates internally k_star
		}
		this->in->gp->covf().get(zb,zb,Vb);

		size_t ind_noise = this->in->gp->covf().get_param_dim() - 1;
		double sn2 = std::exp(2*this->in->gp->covf().get_loghyper()(ind_noise));
		// std::cout << "sn2 = " << sn2 << std::endl;
		Eigen::MatrixXd sn2_diag = sn2 * Eigen::MatrixXd::Identity(Nrep,Nrep);
		Vb = Vb - sn2_diag; // Subtract noise

		// // Fix covariance:
		// Eigen::MatrixXd mat_fix = 1e-1 * Eigen::MatrixXd::Identity(Nrep,Nrep);
		// Vb = Vb + mat_fix;

		return;
	}

	// Get cholesky decomposition:
	// TODO: what happens when the dataset is empty? Does gplib handle this?
	Eigen::MatrixXd chol_kXX = this->in->gp->getCholesky().transpose();
	// Eigen::MatrixXd chol_kXX = this->in->gp->getCholesky();

	// Eigen::MatrixXd chol_kXX = 

	// Compute k(zb,X)
	Eigen::MatrixXd kzbX = Eigen::MatrixXd::Zero(Nrep,Nss);

	for(size_t i = 0 ; i < Nrep ; ++i){
		
		// Compte posterior mean:
		Mb(i) = this->in->gp->f(zb.row(i).data()); // This updates internally k_star

		// Get k_star = k(zb.row(i),X), with dimensions 1xD, and store it as a row vector:
		kzbX.row(i) = this->in->gp->get_k_star(); // k_star was updated when calling gp->f()
	}

	// Prior variance matrix, whose entries (i,j) are k(zb.row(i),zb.row(j))
	Eigen::MatrixXd kzbzb = Eigen::MatrixXd::Zero(Nrep,Nrep);
	this->in->gp->covf().get(zb,zb,kzbzb);

	// We perform: A_ = kzbX * inv(chol_kXX)
	Eigen::MatrixXd A_ = chol_kXX.topLeftCorner(Nss,Nss).triangularView<Eigen::Upper>().solve<Eigen::OnTheRight>(kzbX);
	// Eigen::MatrixXd A_ = chol_kXX.topLeftCorner(Nss,Nss).triangularView<Eigen::Lower>().transpose().solve<Eigen::OnTheRight>(kzbX);




  // // chol(K)	
  // L =  K.adjoint().llt().matrixL();	
  // bt = (L.triangularView<Lower>().solve(Kstar)).adjoint();
  // a = L.triangularView<Lower>().solve(y);
  // // Mean
  // M = bt * a;
  // btb = MatrixXd(kk,kk).setZero().selfadjointView<Lower>().rankUpdate(bt);
  // // covariance
  // C = Kstarstar - btb;

	// Remove the noise: (TODO: the covariance is not exactly the same)
	size_t ind_noise = this->in->gp->covf().get_param_dim() - 1;
	double sn2 = std::exp(2*this->in->gp->covf().get_loghyper()(ind_noise));
	// std::cout << "sn2 = " << sn2 << std::endl;
	// this->Vb = this->Vb - sn2 * Eigen::MatrixXd::Identity(this->in->Nrepresenters,this->in->Nrepresenters);


	// We compute the posterior:
	Eigen::MatrixXd sn2_diag = sn2 * Eigen::MatrixXd::Identity(Nrep,Nrep);
	kzbzb = kzbzb - sn2_diag; // Subtract noise
	Vb = kzbzb - A_ * A_.transpose();
	// Eigen::MatrixXd mat_fix = 1e-1 * Eigen::MatrixXd::Identity(Nrep,Nrep);
	// Vb = Vb + mat_fix;

	// std::cout << "sn2_diag = " << std::endl << sn2_diag.block(0,0,5,5) << std::endl;
	// std::cout << "Vb = " << std::endl << Vb.block(0,0,5,5) << std::endl;

	// Expensive way: kzbzb - (kzX / chol_kXX) * (kzX / chol_kXX)';
	// Vb = kzbzb - kzbX * (chol_kXX.inverse() * ( chol_kXX.inverse().transpose() * kzbX.transpose() ) );

	// proj_alon = kbx - kXb.transpose() * ( cK.inverse() * ( (cK.transpose().inverse()) * kXx ) );

	// This is obviously wrong:
  // (in->gp->covf()).get(zb, zb, Vb);
	// size_t Ntop = 5;
	// std::cout << "kzbX.block(0,0,Ntop,Nss) = " << std::endl << kzbX.block(0,0,Ntop,Nss) << std::endl;
	// std::cout << "A_.block(0,0,Ntop,Nss) = " << std::endl << A_.block(0,0,Ntop,Nss) << std::endl;
	// std::cout << "chol_kXX.block(0,0,Nss,Nss) = " << std::endl << chol_kXX.block(0,0,Nss,Nss) << std::endl;
	// std::cout << "kzbzb.block(0,0,Ntop,Ntop) = " << std::endl << kzbzb.block(0,0,Ntop,Ntop) << std::endl;	
	// std::cout << "Vb.block(0,0,Ntop,Ntop) = " << std::endl << Vb.block(0,0,Ntop,Ntop) << std::endl;

	return;

}

OutResults
EntropySearch::run(void) {

	// Box width:
	double S0 = 0.5 * (this->xmax - this->xmin).norm();

	double thres_warm_start = 0.1;
	double thres_replace = 0.5; // sqrt(2.5e-1)
	Eigen::VectorXd foo = Eigen::VectorXd::Zero(this->in->Dim);

	// Seeding with nanoseconds:
  std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
  std::srand(nanoseconds.count());

  YAML::Node node_to_write;
  node_to_write.SetStyle(YAML::EmitterStyle::Block);


  // DBG:
  // Eigen::MatrixXd x_next_vec = Eigen::MatrixXd::Zero(this->in->MaxEval,this->in->Dim);

	// Copying the pointer to the input structure:
	// cout << "D = " << this->in->Dim << endl;

	// cout << "number of data points N = " << this->in->gp->get_sampleset_size() << endl;
	// cout << "this->xmax: " << this->xmax << endl;
	// cout << "S0: " << S0 << endl;

	// While loop:
	int numiter = 0; 
	bool converged = false;

	EntropySearch::banner_init(this->in->gp);


	while ( !converged && numiter < this->in->MaxEval )
	{

		EntropySearch::banner_iter(numiter+1);

		// Update sample_belief_locations with the new gp object:
		// [zb,lmb]   = SampleBeliefLocations(GP,in.xmin,in.xmax,in.Nb,BestGuesses,in.PropFunc);
		this->sample_bl.update_gp(this->in->gp);
		
		// this->sample_bl.update_gp(this->in->gp); // This updates, internally, the EI function, but doesn't compile: problem using virtual inheritance and const members
		this->sample_bl.sample(this->zb,this->lmb,this->BestGuesses);

	  if(MathTools::isNaN_vec(this->lmb)){
			std::cout << "this->lmb has infs " << std::endl;
			std::cout << "this->lmb = " << std::endl << this->lmb.transpose() << std::endl;
			std::cout << "this->zb = " << std::endl << this->zb << std::endl;
			std::cout << "this->BestGuesses = " << std::endl << this->BestGuesses << std::endl;
      std::cout << "[DBG]: @EntropySearch::run - Paused for debugging" <<std::endl;
      std::chrono::seconds dura(1);
      std::this_thread::sleep_for(dura);
	  }

	  Eigen::VectorXd x_most_informative_mat = Eigen::VectorXd::Zero(this->in->Dim);
	  double y_new_mat;
	  double EdH_max_mat;
	  Eigen::VectorXd Mb_mat(this->in->Nrepresenters);
	  Eigen::MatrixXd Vb_mat(this->in->Nrepresenters,this->in->Nrepresenters);
	  Eigen::VectorXd fooo(this->in->Nrepresenters);
	  if(this->in->read_for_test_SampleBeliefLocations_flag){
	  	std::string path2read("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/tests/matlab/test_sample_belief_locations_new/test_sbl_" + std::to_string(numiter+1) + ".yaml");
			LoggingTools::read_for_test_SampleBeliefLocations(path2read,this->in->Dim, // in
																												// this->zb,this->lmb,x_most_informative_mat,EdH_max_mat); // out
																												this->zb,this->lmb,Mb_mat,Vb_mat,x_most_informative_mat,y_new_mat,EdH_max_mat); // out
		}

		// std::cout << "this->zb.head(8).transpose() = " << this->zb.col(0).head(8).transpose() << std::endl;
		// std::cout << "this->lmb.head(8).transpose() = " << this->lmb.head(8).transpose() << std::endl;



			// // std::cout << "zb_mat_par = " << std::endl << zb_mat_par.head(8).transpose() << std::endl;
			// // std::cout << "logP_mat_par = " << std::endl << logP_mat_par.head(8).transpose() << std::endl;
			// // std::cout << "lmb_mat_par = " << std::endl << lmb_mat_par.head(8).transpose() << std::endl;
			// // std::cout << "dlogPdM_mat = " << std::endl << dlogPdM_mat.block(0,0,8,8) << std::endl;
			// // std::cout << "dlogPdV_mat = " << std::endl << dlogPdV_mat.block(0,0,8,8) << std::endl;
			// // std::cout << "S0_mat = " << std::endl << S0_mat << std::endl;
			// // std::cout << "x_next_mat = " << std::endl << x_next_mat << std::endl;
			// // std::cout << "EdH_next_mat = " << std::endl << EdH_next_mat << std::endl;
		 // //  std::cout << "ddlogPdMdM_mat[0].block(0,0,8,8) = " << std::endl << ddlogPdMdM_mat[0].block(0,0,8,8) << std::endl;
		 // //  std::cout << "ddlogPdMdM_mat[1].block(0,0,8,8) = " << std::endl << ddlogPdMdM_mat[1].block(0,0,8,8) << std::endl;

	  // Quick test:
	  // Eigen::VectorXd xx_in(3); Eigen::VectorXd xx_out(3);
	  // xx_in = xx_in.setOnes()*0.67; xx_out = xx_out.setOnes()*0.67;
	  // if(xx_in.isApprox(xx_out))
		 //  std::cout << "xx_in.isApprox(xx_out) = " << xx_in.isApprox(xx_out) << std::endl;
	  // std::cout << "DBL_EPSILON = " << DBL_EPSILON << std::endl;
   //  std::cout << "[DBG]: @EntropySearch::run - Paused for debugging" <<std::endl;
   //  std::chrono::seconds dura(1);
   //  std::this_thread::sleep_for(dura);
// std::cout <<   "[DBG1]" << std::endl;
	  // Eigen::VectorXd xx_a(3); Eigen::VectorXd xx_b(3); Eigen::VectorXd xx_c(3);
	  // xx_a.setRandom();
	  // std::cout << "xx_a = " << std::endl << xx_a.transpose() << std::endl;
	  // xx_b.setRandom();
	  // std::cout << "xx_b = " << std::endl << xx_b.transpose() << std::endl;
	  // MathTools::sample_unif_vec(this->in->xmin_s,this->in->xmax_s,xx_c);
	  // std::cout << "xx_c = " << std::endl << xx_c.transpose() << std::endl;
	  // MathTools::sample_unif_vec(this->in->xmin_s,this->in->xmax_s,xx_c);
	  // std::cout << "xx_c = " << std::endl << xx_c.transpose() << std::endl;
   //  std::cout << "[DBG]: @EntropySearch::run - Paused for debugging" <<std::endl;
   //  std::chrono::seconds dura(1);
   //  std::this_thread::sleep_for(dura);

		// [Mb,Vb] = GP_moments(GP,zb);
		EntropySearch::posterior(	this->zb,this->in->Nrepresenters, // in
															this->Mb,this->Vb); // out
// std::cout <<   "[DBG1.5]" << std::endl;
		if(this->in->read_for_test_SampleBeliefLocations_flag){
			this->Mb = Mb_mat;
			this->Vb = Vb_mat;
		}
// std::cout <<   "[DBG2]" << std::endl;
		// // Remove the noise: (TODO: the covariance is not exactly the same)
		// size_t ind_noise = this->in->gp->covf().get_param_dim() - 1;
		// double sn2 = std::exp(2*this->in->gp->covf().get_loghyper()(ind_noise));
		// // this->Vb = this->Vb - sn2 * Eigen::MatrixXd::Identity(this->in->Nrepresenters,this->in->Nrepresenters);

		if(this->Vb.diagonal().minCoeff() < 0){
			std::cout << "this->Vb.diagonal().minCoeff() = " << this->Vb.diagonal().transpose() << std::endl;
			throw std::runtime_error("@ES() this->Vb has negative variances in the diagonal");
		}
// std::cout <<   "[DBG3]" << std::endl;
		// std::cout << "this->Mb is not exactly the same" << std::endl;

		// size_t Ntop = 5;
		// std::cout << "this->Mb = " << this->Mb.head(Ntop).transpose() << std::endl;
		// std::cout << "this->Vb = " << std::endl << this->Vb.block(0,0,Ntop,Ntop) << std::endl;

	  if(MathTools::isNaN_vec(this->Mb))
			std::cout << "this->Mb has nans " << std::endl;

	  if(MathTools::isNaN_mat(this->Vb))
			std::cout << "this->Vb has nans " << std::endl;

		// std::cout << "this->Mb = " << std::endl << this->Mb.transpose() << std::endl;

		// cout << "[DBG]: @while() posterior computed, Mb(0) = " << Mb(0) << endl;

		// if(this->in->read_matrices_to_file_test_sample_belief_locations_new){
			// std::cout <<   "[DBG4]" << std::endl;
		// 	// Load files:
		// 	std::cout << "Loading files..." << std::endl;
		// 	std::string path2read("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/tests/matlab/test_sample_belief_locations_new/test_sbl_" + std::to_string(numiter+1) + ".yaml");
		// 	YAML::Node node_to_read = YAML::LoadFile(path2read);

		// 	// Overwrite existing variables:
		//   this->Mb 	= node_to_read["Mb_mat"].as<Eigen::VectorXd>();
		//   this->Vb 	= node_to_read["Vb_mat"].as<Eigen::MatrixXd>();
		//   this->lmb = node_to_read["lmb_mat"].as<Eigen::VectorXd>();
		//   this->zb 	= node_to_read["zb_mat"].as<Eigen::MatrixXd>();
		// }

	  // Belief over the minimum on the sampled set:
	  bel_min.joint_min(this->Mb,this->Vb, // in
	  									this->logP,this->dlogPdMu,this->dlogPdSigma,this->dlogPdMudMu); // out
// std::cout <<   "[DBG5]" << std::endl;
	 //  size_t Ntop = 5;
	 //  size_t ind3Darray = 30;
		// std::cout << "this->logP = " << std::endl << this->logP.head(Ntop).transpose() << std::endl;
		// std::cout << "this->dlogPdMu = " << std::endl << this->dlogPdMu.block(0,0,Ntop,Ntop) << std::endl;
		// std::cout << "this->dlogPdSigma = " << std::endl << this->dlogPdSigma.block(0,0,Ntop,Ntop) << std::endl;
		// std::cout << "this->dlogPdMudMu[ind3Darray] = " << std::endl << this->dlogPdMudMu[ind3Darray].block(0,0,Ntop,Ntop) << std::endl;
		// std::cout << "this->dlogPdMudMu[ind3Darray+1] = " << std::endl << this->dlogPdMudMu[ind3Darray+1].block(0,0,Ntop,Ntop) << std::endl;

		// if(this->in->read_matrices_to_file_test_sample_belief_locations_new){
			
		// 	// Load files:
		// 	std::cout << "Loading files..." << std::endl;
		// 	std::string path2write("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/tests/matlab/test_sample_belief_locations_new/test_sbl_" + std::to_string(numiter+1) + "_output_cpp_loop.yaml");
		// 	YAML::Node node_to_write = YAML::LoadFile(path2write);

		// 	// Overwrite existing variables:
		// 	node_to_write["logP_cpp_loop"] = this->logP;
		// 	node_to_write["dlogPdMu_cpp_loop"] = this->dlogPdMu;
		// 	node_to_write["dlogPdSigma_cpp_loop"] = this->dlogPdSigma;
		// 	node_to_write["dlogPdMudMu_cpp_loop"] = this->dlogPdMudMu;

		//   // Write to file:
		//   std::ofstream fout(path2write);
		//   fout << node_to_write;
		//   fout.close();
		//   std::cout << "Writing succesful!" << std::endl;
		// }



	  ///////////////////////////////////
	  // Used for test_dH_new_loop.cpp
	  ///////////////////////////////////
	  // Eigen::VectorXd x_most_informative_mat;
	  // double EdH_max_mat;
	  if(this->in->read_for_test_dH_MC_local_flag && this->in->Dim <= 2){
			std::string path2read("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/tests/matlab/test_dH_new_loop/test_ESloop_" + std::to_string(numiter+1) + ".yaml");
			LoggingTools::read_for_test_dH_MC_local(path2read,this->in->Dim,this->zb,this->lmb,this->logP,this->dlogPdMu,
																							this->dlogPdSigma,this->dlogPdMudMu,this->Mb,this->Vb,x_most_informative_mat,y_new_mat,EdH_max_mat);
		}
			//  // Load yaml files to mimic matlab's ES variables generated in the loop:
			//  std::cout << "Loading files..." << std::endl;
			//  std::string path2file("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/tests/matlab/test_dH_new_loop/test_ESloop_" + std::to_string(numiter+1) + ".yaml");
			//  YAML::Node node = YAML::LoadFile(path2file);
			//  Eigen::MatrixXd zb_mat = node["zb"].as<Eigen::MatrixXd>();
			//  // std::cout << "[DBG1]" << std::endl;
			//  Eigen::MatrixXd Mb_mat = node["Mb"].as<Eigen::MatrixXd>();
			//  // std::cout << "[DBG2]" << std::endl;
			//  // std::cout << "Mb = " << Mb.head(8).transpose() << std::endl;
			//  Eigen::MatrixXd Vb_mat = node["Vb"].as<Eigen::MatrixXd>();
			//  // std::cout << "[DBG3]" << std::endl;
			//  // std::cout << "Vb = " << Vb.block(0,0,8,8) << std::endl;
			//  Eigen::MatrixXd logP_mat = node["logP"].as<Eigen::MatrixXd>();
			//  // std::cout << "[DBG2]" << std::endl;
			//  Eigen::MatrixXd dlogPdM_mat = node["dlogPdM"].as<Eigen::MatrixXd>();
			//  // std::cout << "[DBG3]" << std::endl;
			//  Eigen::MatrixXd dlogPdV_mat = node["dlogPdV"].as<Eigen::MatrixXd>();
			//  // std::cout << "[DBG4]" << std::endl;
			//  std::vector<Eigen::MatrixXd> ddlogPdMdM_mat = node["ddlogPdMdM"].as<std::vector<Eigen::MatrixXd>>();
			//  // std::cout << "[DBG5]" << std::endl;
			//  Eigen::MatrixXd lmb_mat = node["lmb"].as<Eigen::MatrixXd>();
			//  // std::cout << "[DBG6]" << std::endl;
			//  // double S0_mat = node["S0"].as<double>();
			//  // std::cout << "[DBG7]" << std::endl;
			//  x_next_mat = node["x_next"].as<double>();
			//  // std::cout << "[DBG8]" << std::endl;
			//  EdH_next_mat = node["EdH_next"].as<double>();
			//  // std::cout << "[DBG9]" << std::endl;
			//  std::cout << "Succesfully loaded!" << std::endl;
// std::cout <<   "[DBG6]" << std::endl;
			// // std::cout << "zb_mat.cols() = " << std::endl << zb_mat.cols() << std::endl;
			// // std::cout << "zb_mat.rows() = " << std::endl << zb_mat.rows() << std::endl;
			// // std::cout << "logP_mat.cols() = " << std::endl << logP_mat.cols() << std::endl;
			// // std::cout << "logP_mat.rows() = " << std::endl << logP_mat.rows() << std::endl;
			// // std::cout << "lmb_mat.cols() = " << std::endl << lmb_mat.cols() << std::endl;
			// // std::cout << "lmb_mat.rows() = " << std::endl << lmb_mat.rows() << std::endl;
			// // std::cout << "dlogPdM_mat.cols() = " << std::endl << dlogPdM_mat.cols() << std::endl;
			// // std::cout << "dlogPdM_mat.rows() = " << std::endl << dlogPdM_mat.rows() << std::endl;
			// // std::cout << "dlogPdV_mat.cols() = " << std::endl << dlogPdV_mat.cols() << std::endl;
			// // std::cout << "dlogPdV_mat.rows() = " << std::endl << dlogPdV_mat.rows() << std::endl;
			// std::cout << "Parsing appropiately..." << std::endl;
		 //  Eigen::VectorXd zb_mat_par = zb_mat.col(0);
		 //  Eigen::VectorXd logP_mat_par = logP_mat.col(0);
		 //  Eigen::VectorXd lmb_mat_par = lmb_mat.col(0);
			// std::cout << "Succesfully parsed!" << std::endl;
			// this->Mb 					= Mb.col(0);
			// // std::cout << "this->Mb = " << this->Mb.head(8).transpose() << std::endl;
			// this->Vb 					= Vb;
			// this->zb 					= zb_mat_par;
			// this->lmb 				= lmb_mat_par;
			// this->logP 				= logP_mat_par;
			// this->dlogPdMu 		= dlogPdM_mat;
			// this->dlogPdSigma = dlogPdV_mat;
			// this->dlogPdMudMu = ddlogPdMdM_mat;
	  // }
	  /////////////////////////////////////////

		// // Write to a file to test dH_MC_local:
		// if(this->in->write_matrices_to_file){
		// 	std::cout << "Writing data to files... " << std::endl;
		// 	std::string path_to_file("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/test_joint_min_input_output.yaml");
		// 	node_to_write["Mb"] = this->Mb;
		// 	node_to_write["Vb"] = this->Vb;
		// 	node_to_write["logP"] = this->logP;
		// 	node_to_write["dlogPdMu"] = this->dlogPdMu;
		// 	node_to_write["dlogPdSigma"] = this->dlogPdSigma;
		// 	std::vector<Eigen::MatrixXd> dlogPdMudMu_to_store(this->in->Nrepresenters);
		// 	for(size_t i=0;i<this->in->Nrepresenters;++i)
		// 		dlogPdMudMu_to_store[i] = this->dlogPdMudMu[i];
		// 	node_to_write["dlogPdMudMu"] = dlogPdMudMu_to_store;

		//   // std::cout << "this->dlogPdMudMu[0].block(0,0,4,4) = " << std::endl << this->dlogPdMudMu[0].block(0,0,4,4) << std::endl;
// std::cout <<   "[DBG7]" << std::endl;
		// 	// Store current GP data:
		// 	size_t Ndata_set = this->in->gp->get_sampleset_size();
		// 	Eigen::MatrixXd x_data_set = Eigen::MatrixXd::Zero(Ndata_set,this->in->Dim);
		// 	Eigen::VectorXd y_data_set = Eigen::VectorXd::Zero(Ndata_set);
		// 	for(size_t i=0;i<Ndata_set;++i){
		// 		x_data_set.row(i) = this->in->gp->get_sampleset()->x(i);
		// 		y_data_set(i) = this->in->gp->get_sampleset()->y(i);
		// 	}
		// 	node_to_write["x_data_set"] = x_data_set;
		// 	node_to_write["y_data_set"] = y_data_set;

		//   // Write to file:
		//   std::ofstream fout(path_to_file);
		//   fout << node_to_write;
		//   fout.close();
		//   std::cout << "Writing succesful!" << std::endl;
		// }

	  if(MathTools::isNaN_vec(this->logP))
			std::cout << "this->logP has nans " << std::endl;
		  // std::cout << "this->logP = " << std::endl << this->logP.transpose() << std::endl;

	  if(MathTools::isNaN_mat(this->dlogPdMu))
			std::cout << "this->dlogPdMu has nans " << std::endl;
		  // std::cout << "this->dlogPdMu = " << std::endl << this->dlogPdMu.transpose() << std::endl;

	  if(MathTools::isNaN_mat(this->dlogPdSigma))
			std::cout << "this->dlogPdSigma has nans " << std::endl;

	  if(MathTools::isNaN_3Darray(this->dlogPdMudMu,this->in->Dim))
			std::cout << "this->dlogPdMudMu has nans " << std::endl;

// std::cout <<   "[DBG8]" << std::endl;
	  // if(MathTools::isNaN_mat(this->dlogPdSigma))
		 //  std::cout << "this->dlogPdSigma = " << std::endl << this->dlogPdSigma.transpose() << std::endl;


		// if(this->dlogPdMu.array().isNaN().matrix().any())
		//   std::cout << "this->dlogPdMu = " << std::endl << this->dlogPdMu << std::endl;

	  // std::cout << "this->dlogPdSigma = " << std::endl << this->dlogPdSigma << std::endl;
	  // std::cout << "this->dlogPdMudMu = " << std::endl << this->dlogPdMudMu << std::endl;

	  // cout << "[DBG]: @EntropySearch::run() joint_min correctly computed" << endl;
	  // cout << "[DBG]: @EntropySearch::run() logP(Nb-1) = " << logP(in->Nrepresenters-1) << endl;
	  // cout << "[DBG]: @EntropySearch::run() logP(0) = " << logP(0) << endl;

		// out.Hs(numiter) = - sum(exp(logP) .* (logP + lmb));


		// Obtain the maximum probability:
		size_t ind_max_BG;
		(logP + lmb).maxCoeff(&ind_max_BG);

		// // Store the best current guess as start point for later optimization:
		EntropySearch::update_BestGuesses_list(this->logP,this->lmb,this->zb,ind_max_BG,thres_warm_start,false);

		// 	// Obtain the maximum probability:
		// 	(logP + lmb).maxCoeff(&this->ind_max_BG);
		// 	// cout << "ind_max_BG = " << ind_max_BG << endl;

		// 	// Update Best Guesses: is this far from all the best guesses? If so, then add it in.
		// 	for(size_t i=0 ; i < this->in->Nll ; ++i)
		// 		this->ell(i) = exp(this->in->gp->covf().get_loghyper()(i));

		// 	if ( BestGuesses.rows() == 1 && first_time )
		// 	{
		// 		BestGuesses.row(0) = zb.row(ind_max_BG);
		// 		first_time = false; 
// std::cout <<   "[DBG9]" << std::endl;				
		// 	}
		// 	else
		// 	{
		
		// 		double dist = 0.2;

		// 		// dist = min(sqrt(sum(bsxfun(@minus,zb(bli,:)./ell,bsxfun(@rdivide,BestGuesses,ell)).^2,2)./D));
		// 		Eigen::VectorXd dist1 = Eigen::VectorXd(in->Dim);
		// 		Eigen::MatrixXd dist2 = Eigen::MatrixXd(BestGuesses.rows(),in->Dim);
		// 		Eigen::MatrixXd dist3 = Eigen::MatrixXd(BestGuesses.rows(),in->Dim);
		// 		Eigen::VectorXd dist4 = Eigen::VectorXd(BestGuesses.rows());
		// 		Eigen::VectorXd aux = zb.row(ind_max_BG);
		// 		dist1 = aux.array() / ell.array();
		// 		dist2 = (BestGuesses.array().rowwise()) / ell.transpose().array();
		// 		dist3 = dist2.array().rowwise() - dist1.transpose().array();
		// 		dist3 = (-dist3.array()).pow(2);
		// 		dist4 = dist3.rowwise().sum();
		// 		dist4 = dist4.array()/(this->in->Dim);
		// 		dist4 = dist4.array().sqrt();
		// 		dist  = dist4.minCoeff();

		// 		// cout << "dist = " << dist << endl;

		// 		if ( dist > 0.1 )
		// 		{
		// 			// BestGuesses(size(BestGuesses,1)+1,:) = zb(bli,:);
		// 			int rows_extended = BestGuesses.rows() + 1;
		// 			BestGuesses.conservativeResize(rows_extended,Eigen::NoChange);
		// 			BestGuesses.row(rows_extended - 1) = zb.row(ind_max_BG);
		// 		}
				
		// 	}

			// % store the best current guess as start point for later optimization.

// std::cout <<   "[DBG10]" << std::endl;
			// std::cout << "[DBG]: @EntropySearch::run() BestGuesses [" << this->BestGuesses.rows() << ",";
			// std::cout << this->BestGuesses.cols() << "], = " << std::endl;
			// std::cout << this->BestGuesses << std::endl;

		  if(MathTools::isNaN_mat(this->BestGuesses))
				std::cout << "this->BestGuesses has nans " << std::endl;

			// Change sign before:
			this->EdH_objective->change_output_sign(true);

			// // Write to a file to test dH_MC_local:
			// if(this->in->write_matrices_to_file){
			// 	std::cout << "Writing data to files" << std::endl;
			// 	std::string path_to_file("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/test_dH_input.yaml");
			// 	node_to_write["zb"] = this->zb;
			// 	node_to_write["logP"] = this->logP;
			// 	node_to_write["dlogPdMu"] = this->dlogPdMu;
			// 	node_to_write["dlogPdSigma"] = this->dlogPdSigma;
			// 	node_to_write["lmb"] = this->lmb;
			// 	std::vector<Eigen::MatrixXd> dlogPdMudMu_to_store(this->in->Nrepresenters);
			// 	for(size_t i=0;i<this->in->Nrepresenters;++i)
			// 		dlogPdMudMu_to_store[i] = this->dlogPdMudMu[i];
			// 	node_to_write["dlogPdMudMu"] = dlogPdMudMu_to_store;

			// 	// Store current GP data:
			// 	size_t Ndata_set = this->in->gp->get_sampleset_size();
			// 	Eigen::MatrixXd x_data_set = Eigen::MatrixXd::Zero(Ndata_set,this->in->Dim);
			// 	Eigen::VectorXd y_data_set = Eigen::VectorXd::Zero(Ndata_set);
			// 	for(size_t i=0;i<Ndata_set;++i){
			// 		x_data_set.row(i) = this->in->gp->get_sampleset()->x(i);
			// 		y_data_set(i) = this->in->gp->get_sampleset()->y(i);
			// 	}
			// 	node_to_write["x_data_set"] = x_data_set;
			// 	node_to_write["y_data_set"] = y_data_set;

			//   // Write to file:
			//   std::ofstream fout(path_to_file);
			//   fout << node_to_write;
			//   fout.close();
			//   std::cout << "Writing succesful!" << std::endl;
			// }
// std::cout <<   "[DBG11]" << std::endl;
			// Update EdH_objective through dH_MC_local:
			this->EdH_objective->update_variables(this->zb,this->logP,this->dlogPdMu,this->dlogPdSigma,this->dlogPdMudMu,this->lmb,this->in->gp);

			// double dz_test = (this->in->xmax_s - this->in->xmin_s)/(this->in->Ndiv_plot-1);
			// double dH_plot_i_test = 0.0;
			// Eigen::VectorXd z_plot_test = Eigen::VectorXd::Zero(this->in->Ndiv_plot);
			// Eigen::VectorXd z_plot_i_test = Eigen::VectorXd::Zero(1);
			// Eigen::VectorXd dH_plot_test = Eigen::VectorXd::Zero(this->in->Ndiv_plot);
			// for(size_t i=0;i<this->in->Ndiv_plot;++i){
			// 	z_plot_test(i) = this->in->xmin_s + dz_test * i;
			// 	z_plot_i_test(0) = z_plot_test(i);
			// 	EdH_objective->evaluate(z_plot_i_test,&dH_plot_i_test,&foo);
			// 	dH_plot_test(i) = dH_plot_i_test;
			// }
			// dH_plot_test.array() = -dH_plot_test.array();

			// size_t Ntop = 5;
			// std::cout << "z_plot_test = " << std::endl << z_plot_test.head(Ntop).transpose() << std::endl;
			// std::cout << "dH_plot_test = " << std::endl << dH_plot_test.head(Ntop).transpose() << std::endl;

		 //  size_t Ntop = 5;
		 //  size_t ind3Darray = 30;
		 //  std::cout << "this->zb = " << std::endl << this->zb.col(0).head(Ntop).transpose() << std::endl;
			// std::cout << "this->lmb = " << std::endl << this->lmb.head(Ntop).transpose() << std::endl;
			// std::cout << "this->logP = " << std::endl << this->logP.head(Ntop).transpose() << std::endl;
			// std::cout << "this->dlogPdMu = " << std::endl << this->dlogPdMu.block(0,0,Ntop,Ntop) << std::endl;
			// std::cout << "this->dlogPdSigma = " << std::endl << this->dlogPdSigma.block(0,0,Ntop,Ntop) << std::endl;
			// std::cout << "this->dlogPdMudMu[ind3Darray] = " << std::endl << this->dlogPdMudMu[ind3Darray].block(0,0,Ntop,Ntop) << std::endl;
			// std::cout << "this->dlogPdMudMu[ind3Darray+1] = " << std::endl << this->dlogPdMudMu[ind3Darray+1].block(0,0,Ntop,Ntop) << std::endl;


		// Construct EdH_objective to be used in the optimizer:
			// this->dH = new dH_MC_local(zb,logP,dlogPdMu,dlogPdSigma,dlogPdMudMu,this->in->T,lmb,this->xmin,this->xmax,true,this->in->gp); // originally false
			// this->dH_p = new dH_MC_local(zb,logP,dlogPdMu,dlogPdSigma,dlogPdMudMu,this->in->T,lmb,this->xmin,this->xmax,true,this->in->gp);

			// Wrapper, to make it compatible with the type of function that bfgs expects:
			// this->EdH_objective = new WrapperEdH(this->dH);
			// this->EdH_p_objective = new WrapperEdH(this->dH_p);

		// Call to Slice_ShrinkRank_nolog() (coded!) to obtain the warm starts for the local optimization below:
			// Eigen::VectorXd xx_warm 	= Eigen::VectorXd::Zero(in->Dim);
			// Eigen::VectorXd xx_init 		= Eigen::VectorXd::Zero(in->Dim);
			// MathTools::sample_unif_vec(this->in->xmin_s,this->in->xmax_s,xx_init);
			// xx_warm = SliceShrinkRank_nolog(xx_init,this->EdH_p_objective, S0); // We need bfgs_optimizer::ObjectiveFunction
			// xx_warm << 5.0,3.0,0.1;
// std::cout <<   "[DBG1]: @EntropySearch::run() x_most_informative = "  << std::endl;
			// Get warm starts for the local optimization of EdH_p_objective:
			Eigen::MatrixXd x_warm_starts = 
			EntropySearch::get_warm_starts(this->in->xmin_s,this->in->xmax_s,this->in->Nwarm_starts,this->in->Nsubsamples,this->in->Dim,S0,this->EdH_objective); // originally EdH_p_objective
// std::cout <<   "[DBG2]: @EntropySearch::run() x_most_informative = " << std::endl;
// std::cout <<   "[DBG12]" << std::endl;
			// std::cout << "[DBG]: @EntropySearch::run() xx_init = " << xx_init.transpose() << std::endl;
			// std::cout << "[DBG]: @EntropySearch::run() xx_warm = " << xx_warm.transpose() << std::endl;

			// Change sign before:
			this->EdH_objective->change_output_sign(false); // amarcovalle: originally uncommented
// std::cout <<   "[DBG13]" << std::endl;
			// Find the most informative point by running local optimization:
			bfgs_optimizer::BFGS bfgs_EdH(this->EdH_objective,this->in->Nline_searches); // We need bfgs_optimizer::ObjectiveFunction
			Eigen::MatrixXd x_most_informative_list = Eigen::MatrixXd(this->in->Nwarm_starts,this->in->Dim);
			Eigen::VectorXd EdH_list = Eigen::VectorXd(this->in->Nwarm_starts);
		  Eigen::VectorXd x_most_informative = Eigen::VectorXd(this->in->Dim);
			for(size_t i=0;i<this->in->Nwarm_starts;++i){
			  x_most_informative_list.row(i) = bfgs_EdH.minimize(x_warm_starts.row(i));
			  this->EdH_objective->evaluate(x_most_informative_list.row(i),&EdH_list(i),&foo);
			}

			size_t ind_min;
			EdH_list.minCoeff(&ind_min);
			x_most_informative = x_most_informative_list.row(ind_min);
			double EdH_max;
			this->EdH_objective->evaluate(x_most_informative,&EdH_max,&foo);
			EdH_max = -EdH_max;
// std::cout <<   "[DBG14]" << std::endl;
			// [DBG]: Overwrite:
			if( (this->in->read_for_test_dH_MC_local_flag || this->in->read_for_test_SampleBeliefLocations_flag) && this->in->Dim <= 2 ){
				x_most_informative = x_most_informative_mat;
				EdH_max = -EdH_max_mat;
			}

			Eigen::VectorXd dH_plot;
			Eigen::VectorXd z_plot_single;
			Eigen::MatrixXd z_plot;
			Eigen::VectorXd z_plot_i;
			double dz;
			size_t Nel;
			if( this->in->Dim <= 2 && (this->in->write2pyplot || this->in->write2file) ){
				dz = (this->in->xmax_s - this->in->xmin_s)/(this->in->Ndiv_plot-1);
				Nel = std::pow(this->in->Ndiv_plot,this->in->Dim);
				dH_plot = Eigen::VectorXd::Zero(Nel);
				z_plot_single = Eigen::VectorXd::Zero(this->in->Ndiv_plot);
				z_plot = Eigen::MatrixXd::Zero(Nel,this->in->Dim);
				z_plot_i = Eigen::VectorXd::Zero(this->in->Dim);
				// TODO: move this to MathTools:
				for(size_t i=0;i<this->in->Ndiv_plot;++i){
					z_plot_single(i) = this->in->xmin_s + dz * i;
				}

				if(this->in->Dim == 1){
					
					// Export true function:
					z_plot = z_plot_single;
					for(size_t i=0;i<Nel;++i){

						z_plot_i = z_plot.row(i);

						// GP posterior mean and std:
						this->EdH_objective->evaluate(z_plot_i,&dH_plot(i),&foo);
					}

					dH_plot.array() = -dH_plot.array();
				}
				else if(this->in->Dim == 2){
					for(size_t i=0;i<this->in->Ndiv_plot;++i){
						z_plot.block(i*this->in->Ndiv_plot,0,this->in->Ndiv_plot,1) = Eigen::VectorXd::Ones(this->in->Ndiv_plot)*z_plot_single(i);
						z_plot.block(i*this->in->Ndiv_plot,1,this->in->Ndiv_plot,1) = z_plot_single;
					}
				}

			}


		// Load exploration status to be plotted:
	  if(this->in->write2pyplot && this->in->Dim <= 2){
			LoggingTools::write_for_pyplot(this->in->path2data_logging_absolute,
																				this->in->Dim,
																				numiter,
																				this->in->plot_true_function,
																				this->in->Ndiv_plot,
																				this->in->gp,
																				this->in->cost_function,
																				z_plot,
																				dH_plot,
																				EdH_max,
																				x_most_informative);
	  }


			// // DBG:
			// std::cout << "x_most_informative = " << x_most_informative << std::endl;
			// std::cout << "EdH_list(ind_min) = " << EdH_list(ind_min) << std::endl;
			// std::cout << "EdH_list = " << EdH_list << std::endl;
			// double EdH_max;
			// this->EdH_objective->evaluate(x_most_informative.row(0),&EdH_max,&foo);
			// std::cout << "@ES() [DBG1]: EdH_max = " << EdH_max << std::endl;
			// EdH_max = -EdH_max;
			// std::cout << "@ES() [DBG2]: EdH_max = " << EdH_max << std::endl;
			// // Test:
			// Eigen::VectorXd x_test(1);
			// x_test(0) = 0.6; 
			// Eigen::VectorXd x_test_sol = bfgs_EdH.minimize(x_test);

			// double EdH_test = 0;
			// Eigen::VectorXd EdH_test_der(1);
			// this->EdH_objective->evaluate(x_test,&EdH_test,&EdH_test_der);
			// std::cout << "@ES() x_test = " << x_test << std::endl;
			// std::cout << "@ES() EdH_test = " << EdH_test << std::endl;
			// std::cout << "@ES() EdH_test_der = " << EdH_test_der << std::endl;

		  // std::cout << "EdH local optimization suceeded!" << std::endl;

			// // Compute print locations:
			// size_t Nel = pow(this->in->Ndiv_plot,this->in->Dim);
			// Eigen::VectorXd dH_plot = Eigen::VectorXd::Zero(Nel);
			// if(this->in->Dim == 1){
			// 	double dz = (this->in->xmax_s - this->in->xmin_s)/(this->in->Ndiv_plot-1);
			// 	double dH_plot_i = 0.0;
			// 	Eigen::VectorXd z_plot = Eigen::VectorXd::Zero(this->in->Ndiv_plot);
			// 	Eigen::VectorXd z_plot_i = Eigen::VectorXd::Zero(1);
			// 	for(size_t i=0;i<Nel;++i){
			// 		z_plot(i) = this->in->xmin_s + dz * i;
			// 		z_plot_i(0) = z_plot(i);
			// 		this->EdH_objective->evaluate(z_plot_i,&dH_plot_i,&foo);
			// 		dH_plot(i) = dH_plot_i;
			// 	}
			// 	dH_plot.array() = -dH_plot.array();
			// }

			
			// // DBG:
			// Eigen::VectorXd x_dbg(1);
			// x_dbg(0) = 0.54321;
			// double dH_plot_dbg = 0.0;
			// this->EdH_objective->evaluate(x_dbg,&dH_plot_dbg,&foo);

			// size_t Ntop = 5;
			// std::cout << "z_plot = " << std::endl << z_plot.head(Ntop).transpose() << std::endl;
			// std::cout << "dH_plot = " << std::endl << dH_plot.head(Ntop).transpose() << std::endl;

			// std::cout << "@ES()1 z_plot = " << z_plot.transpose() << std::endl;
			// std::cout << "@ES()1 dH_plot = " << dH_plot.transpose() << std::endl;
			// for(size_t i=0;i<this->in->Ndiv_plot;++i){
			// 	z_plot(i) = this->in->xmin_s + dz * i;
			// 	z_plot_i(0) = z_plot(i);
			// 	this->EdH_objective->evaluate(z_plot_i,&dH_plot(i),&foo);
			// 	dH_plot(i) = -dH_plot(i);
			// }
			// std::cout << "@ES()2 z_plot = " << z_plot.head(8).transpose() << std::endl;
			// std::cout << "@ES()2 dH_plot = " << dH_plot.head(8).transpose() << std::endl;

		  // Project back inside the domain, if needed:
		  if(MathTools::outside_dom_vec(this->in->xmin_s,this->in->xmax_s,x_most_informative))
		  	throw std::runtime_error("x_most_informative has to be projected back to the domain...");

		  MathTools::project_to_boundaries(this->in->xmin_s,this->in->xmax_s,x_most_informative);
		  // std::cout << "[DBG4]: @EntropySearch::run() x_most_informative = " << x_most_informative << std::endl;

		// Get an experimental value/observation:
		  std::cout << "    x_next = [ " << x_most_informative.transpose() << " ]" << std::endl;
		  double y_new = this->in->cost_function->get_value(x_most_informative);
		  std::cout << "    y = f(x_next) = " << y_new << std::endl;

		  // DBG:
		  if(this->in->read_for_test_SampleBeliefLocations_flag || this->in->read_for_test_dH_MC_local_flag){
		  	y_new = y_new_mat;
		  }
		  this->in->gp->add_pattern(x_most_informative.data(),y_new);
		  // std::cout << "    f:            = " << this->in->cost_function->function_name() << std::endl;

		  // // [DBG]: Test: save data over iterations to reproduce the search in the matlab version
		  // if(true){
		  // 	x_next_vec.row(numiter) = x_most_informative;
		  // }

// std::cout << "[DBG1]: @EntropySearch::run()" << std::endl;
		// // Call to FindGlobalGPMinimum() (not coded?)
		// // [out.FunEst(numiter,:),FunVEst] = FindGlobalGPMinimum(BestGuesses,GP,in.xmin,in.xmax);
		//   bfgs_optimizer::BFGS bfgs_GP(this->GPmean,10); // We need bfgs_optimizer::ObjectiveFunction
		//   Eigen::VectorXd x_init_gp = Eigen::VectorXd(in->Dim);
		//   Eigen::VectorXd x_final_gp = Eigen::VectorXd(in->Dim);
		//   MathTools::sample_unif_vec(this->in->xmin_s,this->in->xmax_s,x_init_gp);
		//   x_final_gp = bfgs_GP.minimize(x_init_gp);
		//   // std::cout << "GPmean local optimization suceeded!" << std::endl;

		  Eigen::VectorXd x_GPmean_min = Eigen::VectorXd::Zero(this->in->Dim);
		  // std::cout << "[DBG2]: @EntropySearch::run()" << std::endl;
		  // this->GPmean = new WrapperGP(this->in->gp,this->xmin,this->xmax);
		  this->GPmean->update_gp(this->in->gp);
		  // std::cout << "[DBG3]: @EntropySearch::run()" << std::endl;
			x_GPmean_min = EntropySearch::FindGlobalGPMinimum(this->GPmean,this->BestGuesses,this->in->xmin_s,this->in->xmax_s);
// std::cout << "[DBG4]: @EntropySearch::run()" << std::endl;
		// Update global minimum estimates:
			this->global_min_esti_x.row(numiter) = x_GPmean_min;
			this->global_min_esti_mux(numiter) = this->in->gp->f(x_GPmean_min.data());
			this->global_min_esti_varx(numiter) = this->in->gp->var(x_GPmean_min.data());

		// Update BestGuesses"
			EntropySearch::update_BestGuesses_list(logP,lmb,zb,numiter,thres_replace,true);

		// Optimize hyperparameters:

		// Store some variables:
	  if(this->in->write2file && this->in->Dim <= 2){
	  	LoggingTools::log_variables(this->in->Dim,this->in->xmin_s,this->in->xmax_s,numiter,ind_min,this->in->Ndiv_plot,
	  															this->in->gp, dH_plot, EdH_max, this->in->cost_function,
	  															this->in->plot_true_function,
	  															this->Mb,this->Vb,this->zb,
	  															x_most_informative,this->global_min_esti_x,this->global_min_esti_mux,this->global_min_esti_varx);
	  }


	  if(this->in->write2pyplot && this->in->Dim <= 2){
			LoggingTools::write_for_pyplot(this->in->path2data_logging_absolute,
																				this->in->Dim,
																				numiter,
																				this->in->plot_true_function,
																				this->in->Ndiv_plot,
																				this->in->gp,
																				this->in->cost_function,
																				z_plot,
																				dH_plot,
																				EdH_max,
																				x_most_informative);
	  }

  // Progress log:
  std::cout << std::endl;
  std::cout << "    List of global minimums" << std::endl;
  std::cout << "    =======================" << std::endl;
  for(size_t i=0;i<numiter+1;++i){
    std::cout << "     " << i+1 << ") [" << this->global_min_esti_x.row(i) << "]" << 
    ", mu(x_bg) = " << this->global_min_esti_mux(i) <<
    ", var(x_bg) = " << this->global_min_esti_varx(i) << std::endl;
  }

  std::cout << std::endl;
  std::cout << "    List of evaluations" << std::endl;
  std::cout << "    =======================" << std::endl;
  for(size_t i=0;i<in->gp->get_sampleset_size();++i){
    std::cout << "     " << i+1 << ") X = [" << in->gp->get_sampleset()->x(i).transpose() << "]" << 
    ", y(X) = " << in->gp->get_sampleset()->y(i) << std::endl;
  }


  // Log data to a yaml file:
	LoggingTools::write_progress(this->in->path2data_logging_absolute,
															this->in->Dim,
															this->in->gp,
															this->global_min_esti_x,	
															this->global_min_esti_mux,
															this->global_min_esti_varx);

	  // Update counter:
		++numiter;
	}


	// Display information:
	EntropySearch::banner_final(this->in->MaxEval,this->global_min_esti_x,this->global_min_esti_mux,this->global_min_esti_varx);

	// Output structure:
	OutResults out_results;
	// out_results.in = std::make_shared<INSetup>();
	out_results.in = this->in;
	out_results.global_min_esti_x = this->global_min_esti_x;
	out_results.global_min_esti_mux = this->global_min_esti_mux;
	out_results.global_min_esti_varx = this->global_min_esti_varx;

	return out_results;

	// // Write to file:
	// if(this->in->write2file){
	// 	std::ofstream * file = new std::ofstream("data.txt");
	// 	for(size_t i=0;i<this->in->MaxEval;++i){
	// 		MathTools::write2file_mat(file,this->global_min_esti_x,"Matrix");
	// 	}
	// 	file->close();
	// }

	// if(true){
	// 	  YAML::Node node_test;
	// 	  node_test.SetStyle(YAML::EmitterStyle::Block);

	// 		std::cout << "Writing data to files... " << std::endl;
	// 		std::string path_to_file("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/test_next_points.yaml");
	// 		node_test["x_next_vec"] = x_next_vec;

	// 	  std::ofstream fout(path_to_file);
	// 	  fout << node_test;
	// 	  fout.close();
	// 	  std::cout << "Writing succesful!" << std::endl;
	// }

}


// void
// EntropySearch::get_EdH_plot(Eigen::VectorXd & dH_plot, MatrixXd & z_plot, 
// 														Eigen::VectorXd & x_next, double & EdH_next,
// 														size_t Dim, size_t Ndiv_plot.
// 														DummyFunction * EdH_objective){

// 	Eigen::VectorXd z_plot_i = Eigen::VectorXd::Zero(Dim);
// 	Eigen::VectorXd foo = Eigen::VectorXd::Zero(Dim);
// 	size_t Nel = std::pow(Ndiv_plot,Dim);
// 	Eigen::VectorXd x_next;
// 	size_t ind_min;
// 	double EdH_min;

// 	// Export true function:
// 	for(size_t i=0;i<Nel;++i){

// 		z_plot_i = z_plot.row(i);

// 		EdH_objective->evaluate(z_plot_i,&dH_plot(i),&foo);
// 	}

// 	dH_plot.minCoeff(&ind_min);
// 	x_next = dH_plot.row(ind_min);
// 	this->EdH_objective->evaluate(x_next,&EdH_min,&foo);
// 	EdH_next = -EdH_min;

// }

void 
EntropySearch::update_BestGuesses_list( Eigen::VectorXd logP,
																							Eigen::VectorXd lmb,
																							Eigen::MatrixXd candidates,
																							size_t ind_for_cand,
																							double threshold,
																							bool replace){

	// Replace, extend or leave unaltered the BestGuesses:
	//
	// If the candidate is far away from the BestGuesses, 
	// i.e., dist > threshold, then this candidate does not
	// improve any of the current BestGuesses, but still is a good candidate, 
	// so we add it to the list.
	//
	// If the candidate is close enough to one of the BestGuesses,
	// i.e., dist < threshold, then replace this BestGuess by the candidate,
	// only when asked by the user.

	// TODO: Try to pass BestGuesses by reference, but without using Eigen::Ref<>

	size_t ind_dist_min;

	// Lengthscales:
	Eigen::VectorXd ell(in->Nll);
	for(size_t i=0 ; i < this->in->Nll ; ++i)
		ell(i) = exp(this->in->gp->covf().get_loghyper()(i));

	// Update Best Guesses: is this far from all the best guesses? If so, then add it in.
	if ( this->BestGuesses.rows() == 1 && this->first_time ) {
		this->BestGuesses.row(0) = candidates.row(ind_for_cand);
		this->first_time = false;
		return;
	}

	// dist = min(sqrt(sum(bsxfun(@minus,zb(bli,:)./ell,bsxfun(@rdivide,BestGuesses,ell)).^2,2)./D));
	Eigen::VectorXd dist1 = Eigen::VectorXd(in->Dim);
	Eigen::MatrixXd dist2 = Eigen::MatrixXd(this->BestGuesses.rows(),in->Dim);
	Eigen::MatrixXd dist3 = Eigen::MatrixXd(this->BestGuesses.rows(),in->Dim);
	Eigen::VectorXd dist4 = Eigen::VectorXd(this->BestGuesses.rows());
	Eigen::VectorXd aux = candidates.row(ind_for_cand);
	dist1 = aux.array() / ell.array();
	dist2 = (this->BestGuesses.array().rowwise()) / ell.transpose().array();
	dist3 = dist2.array().rowwise() - dist1.transpose().array();
	dist3 = (-dist3.array()).pow(2);
	dist4 = dist3.rowwise().sum();
	dist4 = dist4.array()/(this->in->Dim);
	dist4 = dist4.array().sqrt();
	double dist = dist4.minCoeff(&ind_dist_min);

	// Extend the bestguesses or replace the one at ind_for_cand
	if ( dist > threshold ) { // Grow, and add the candidate to the best guesses
		// BestGuesses(size(BestGuesses,1)+1,:) = zb(bli,:); and BestGuesses(size(BestGuesses,1)+1,:) = out.FunEst(numiter,:);
		size_t rows_extended = this->BestGuesses.rows() + 1;
		this->BestGuesses.conservativeResize(rows_extended,Eigen::NoChange);
		this->BestGuesses.row(rows_extended - 1) = candidates.row(ind_for_cand);
	}
	else { // Dont' grow, but replace if asked
		if(replace){
			// BestGuesses(ci,:)  = out.FunEst(numiter,:);
			this->BestGuesses.row(ind_dist_min) = candidates.row(ind_for_cand);
		}
	}

	return;
}

Eigen::VectorXd
EntropySearch::FindGlobalGPMinimum(bfgs_optimizer::ObjectiveFunction * GPmean, Eigen::MatrixXd BestGuesses, double xmin_s, double xmax_s){

// std::cout << "[DBG1]: @EntropySearch::FindGlobalGPMinimum() " << std::endl;
	size_t NBG 	= BestGuesses.rows();
	size_t D 		= this->in->Dim;
	size_t ind_min;
	double y_fin;
	// std::cout << "[DBG2]: @EntropySearch::FindGlobalGPMinimum() " << std::endl;
  Eigen::VectorXd x_init = Eigen::VectorXd::Zero(D);
  Eigen::VectorXd x_fin = Eigen::VectorXd::Zero(D);
  Eigen::VectorXd foo = Eigen::VectorXd::Zero(D);
	Eigen::MatrixXd x_fin_list 	= Eigen::MatrixXd::Zero(2*NBG,D);
	Eigen::VectorXd y_fin_list 	= Eigen::VectorXd::Zero(2*NBG);
  bfgs_optimizer::BFGS bfgs_GP(GPmean,this->in->Nline_searches); // We need bfgs_optimizer::ObjectiveFunction
// std::cout << "[DBG3]: @EntropySearch::FindGlobalGPMinimum() " << std::endl;
	for(size_t i=0;i<2*NBG;++i){

		// Select initial point:
		if( i < NBG )
			x_init = BestGuesses.row(i);
		else
			MathTools::sample_unif_vec(xmin_s,xmax_s,x_init);

		// Call the optimizer:
		// [out.FunEst(numiter,:),FunVEst] = FindGlobalGPMinimum(BestGuesses,GP,in.xmin,in.xmax);
	  x_fin = bfgs_GP.minimize(x_init);

	  // Evaluate on x_fin:
	  GPmean->evaluate(x_fin,&y_fin,&foo);
		
	  // Update lists:
	  x_fin_list.row(i) = x_fin;
		y_fin_list(i) = y_fin;
	}
// std::cout << "[DBG2]: @EntropySearch::FindGlobalGPMinimum() " << std::endl;
	// Get the best in the list:
	y_fin_list.minCoeff(&ind_min);
// std::cout << "[DBG3]: @EntropySearch::FindGlobalGPMinimum() " << std::endl;
	return x_fin_list.row(ind_min);
}

Eigen::MatrixXd
EntropySearch::get_warm_starts(double xmin_s, double xmax_s, size_t Nwarm_starts, size_t Nsub, size_t D, double S0, bfgs_optimizer::ObjectiveFunction * fun){

	Eigen::MatrixXd xx_warm = Eigen::MatrixXd::Zero(Nwarm_starts,D);
	Eigen::VectorXd xx_in = Eigen::VectorXd::Zero(D);
	Eigen::VectorXd xx_out = Eigen::VectorXd::Zero(D);

	// // [DBG] Test:
 //  double ff;
 //  Eigen::VectorXd dff = Eigen::VectorXd(D); // Not used, actually
 //  Eigen::VectorXd x_test = Eigen::VectorXd(D); // Not used, actually
 //  x_test(0) = 0.6;
 //  fun->evaluate(x_test,&ff,&dff); // with bfgs_optimizer::ObjectiveFunction * fun
 //  std::cout << "@get_warm_starts() ff = " << ff << std::endl;
 //  std::cout << "@get_warm_starts() dff = " << dff << std::endl;
 //  Eigen::VectorXd x_test_out = SliceShrinkRank_nolog(x_test,fun,S0);
 //  std::cout << "@get_warm_starts() x_test = " << x_test << std::endl;
  // std::cout << "@get_warm_starts()" << std::endl;

  for(size_t i=0;i<Nwarm_starts;++i){

		MathTools::sample_unif_vec(xmin_s,xmax_s,xx_in);

    // std::cout << "xx_in = " << xx_in.transpose() << std::endl;
    // Call slice sampler, and subsample:
    for(size_t k=0;k<Nsub;++k){
      
      // std::cout << "xx_in = " << xx_in << std::endl;
      xx_out = SliceShrinkRank_nolog(xx_in,fun,S0);
      // std::cout << "xx_out = " << xx_out << std::endl;
      // std::cout << "xx_out = " << xx_out.transpose() << std::endl;

      // If they are the same, the slice sampler couldn't improve this sample.
      // Therefore, consider randomly sampling a new one:
      // Comparing xx_in == xx_out up to some precision parameter, which if not passed, is set to 0
      // If the outcome of the slice sampler is out of bounds, resample:
      if( xx_in.isApprox(xx_out) || MathTools::outside_dom_vec(xmin_s,xmax_s,xx_out) ){
        MathTools::sample_unif_vec(xmin_s,xmax_s,xx_out);
        // std::cout << "[WARNING]: @EntropySearch::get_warm_starts - Randomly sampling the input to the slice sampler" << std::endl;
        // std::cout << "[WARNING]: @EntropySearch::get_warm_starts - What about just projecting back to the domain?" << std::endl;
      }

      // Restart the sampling from the same point:
      xx_in = xx_out;
    }

    // Add to the list of warm starts:
    xx_warm.row(i) = xx_out;

  }

  return xx_warm;
}

void
EntropySearch::banner_init(libgp::GaussianProcess * gp){

	std::cout << std::endl << std::endl;
	std::cout << "/////////////////////////////////////////////////////////////////" << std::endl;
	std::cout << "                          EntropySearch                          " << std::endl;
	std::cout << "/////////////////////////////////////////////////////////////////" << std::endl;
	std::cout << std::endl;
	std::cout << "Input parameters" << std::endl;
	std::cout << "=================================================================" << std::endl;
	std::cout << " Covariance function:       " << gp->covf().to_string() << std::endl;
	std::cout << " Number of training points: " << gp->get_sampleset_size() << std::endl;
	std::cout << std::endl;

	return;
}

void
EntropySearch::banner_iter(size_t numiter){

	std::cout << std::endl;
	std::cout << "=============================" << std::endl;
	std::cout << " EntropySearch - Iteration " << numiter << std::endl;
	std::cout << "=============================" << std::endl;
	std::cout << std::endl;

	return;
}

void
EntropySearch::banner_final(size_t MaxEvals, Eigen::MatrixXd global_min_esti_x, Eigen::VectorXd global_min_esti_mux, Eigen::VectorXd global_min_esti_varx){

	// Print out information:
	std::cout << std::endl;
	std::cout << "=============================" << std::endl;
	std::cout << " EntropySearch has finished! " << std::endl;
	std::cout << "=============================" << std::endl;
	std::cout << std::endl;

	return;
}


