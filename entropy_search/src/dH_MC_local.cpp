// Copyright (c) 2015 Max Planck Society

/*!@file
 * @author  Alonso Marco <alonso.marco@tuebingen.mpg.de>
 *
 * @date    2015-11-20
 *
 * @details 
 * This class implements the MATLAB function dH_MC_local()
 *
 */
 
#include "dH_MC_local.hpp"
#define EPS_ 1e-50
// #define DBL_EPSILON 2.2204460492503131e-16 # TODO: consider using this one, instead
#define INF_ 1e300

// For Debugging:
    #include <chrono>
    #include <thread>

// // DBG:
// #include "ReadYamlParameters.hpp"

dH_MC_local::dH_MC_local(const  Eigen::MatrixXd zbel,
                                Eigen::VectorXd logP,
                                Eigen::MatrixXd dlogPdM,
                                Eigen::MatrixXd dlogPdV,
                                std::vector<Eigen::MatrixXd> ddlogPdMdM,
                                const size_t T,
                                Eigen::VectorXd lmb,
                                Eigen::VectorXd xmin,
                                Eigen::VectorXd xmax,
                                bool invertsign,
                                libgp::GaussianProcess * gp
                                )
                                : 
                                LogLoss_(), 
                                logsumexp_(){
  
  // Note: the previous procedure was a bug, i.e.,:
    // We draw samples, but only once, to ensure smooth derivatives:
    // std::default_random_engine generator;
    // std::normal_distribution<double> distribution(0.0,1.0);
    // W = Eigen::VectorXd(T);
    // for(int i = 0;i < T;i++)
    //   W(i) = distribution(generator);

  // % Variable W needed for Riemann integration when we compute the
  // % expected change in the GP mean, for which we need an unknown new
  // % measurement y, over which we marginalize:
    // if ~isfield(specs,'W')
    //     W = sqrt(2) * erfinv(2*linspace(0,1,specs.T+2)-1);
    //     specs.W = W(2:end-1);
    // end

  double lim_low = -1.0;
  double lim_up  = +1.0;
  double dz = (lim_up-lim_low)/(T+1);
  W = Eigen::VectorXd(T);
  
  for(size_t i=1;i<T+1;++i){ // Omit the instances 0 and T+2, as they correspond to -1 and 1, which yield Inf
    W(i-1)  = std::sqrt(2) * ErrorFunction::erfinv(lim_low + dz * i);
  }

  this->logP        = logP;
  this->dlogPdM     = dlogPdM;
  this->dlogPdV     = dlogPdV;
  this->ddlogPdMdM  = ddlogPdMdM;
  this->lmb         = lmb;
  this->xmin        = xmin;
  this->xmax        = xmax;
  this->invertsign  = invertsign;
  this->dGP         = new GPInnovationLocal(gp,zbel); // zb has to be [Nb D]

}

dH_MC_local::~dH_MC_local(void) {}

void
dH_MC_local::update_variables(Eigen::MatrixXd zbel,
                              Eigen::VectorXd logP,
                              Eigen::MatrixXd dlogPdM,
                              Eigen::MatrixXd dlogPdV,
                              std::vector<Eigen::MatrixXd> ddlogPdMdM,
                              Eigen::VectorXd lmb,
                              libgp::GaussianProcess * gp){

  // Update variables:
  this->logP        = logP;
  this->dlogPdM     = dlogPdM;
  this->dlogPdV     = dlogPdV;
  this->ddlogPdMdM  = ddlogPdMdM;
  this->lmb         = lmb;

  // Update GPInnovationLocal():
  this->dGP->update_variables(gp,zbel);

  // size_t Ntop = 5;
  // size_t ind3Darray = 30;
  // std::cout << "zb = " << std::endl << zbel.col(0).head(Ntop).transpose() << std::endl;
  // std::cout << "this->lmb = " << std::endl << this->lmb.head(Ntop).transpose() << std::endl;
  // std::cout << "this->logP = " << std::endl << this->logP.head(Ntop).transpose() << std::endl;
  // std::cout << "this->dlogPdMu = " << std::endl << this->dlogPdM.block(0,0,Ntop,Ntop) << std::endl;
  // std::cout << "this->dlogPdSigma = " << std::endl << this->dlogPdV.block(0,0,Ntop,Ntop) << std::endl;
  // std::cout << "this->dlogPdMudMu[ind3Darray] = " << std::endl << this->ddlogPdMdM[ind3Darray].block(0,0,Ntop,Ntop) << std::endl;
  // std::cout << "this->dlogPdMudMu[ind3Darray+1] = " << std::endl << this->ddlogPdMdM[ind3Darray+1].block(0,0,Ntop,Ntop) << std::endl;


}

void dH_MC_local::change_sign(bool invertsign){
  this->invertsign = invertsign;
}

void dH_MC_local::dHdx_local(const Eigen::VectorXd x, double & dH, Eigen::VectorXd & ddHdx)
{

  if ( (x.array() < xmin.array()).any() || (x.array() > xmax.array()).any() )
  {
    // TODO: get here the true EPS number. Look at mxgeteps.c
    dH = EPS_;
    // dH = INF_; // amarcovalle, TODO: Shouldn't this be Inf, since we are minimzing?
    ddHdx = Eigen::VectorXd(x.size()).setZero();
    // std::cout << "[DBG]: @dHdx_local() x is out of bounds" << std::endl;
    return;
  }

  // if (!this->invertsign){
  //   std::cout << "[DBG]: @dHdx_local(): " << std::endl;
  //   std::cout << "@dH_MC_local::dHdx_local - x = " << x.transpose() << std::endl;
  // }

  // Compute dH at location x:
  dH = get_dH(x);

  if(std::isnan(dH)){
    std::cout << "dH = " << dH << std::endl;
    std::cout << "@dH_MC_local::dHdx_local - x = " << x.transpose() << std::endl;
    std::cout << "[DBG]: Paused for debugging" <<std::endl;
    std::chrono::seconds dura(1);
    std::this_thread::sleep_for(dura);
  }

  // Compute gradient:
    // Gradient deviation:
    double e_ = 1e-5; // Always, matlab
    // double e_ = 1e-1;

    // dH derivative:
    int D = x.size();
    ddHdx = Eigen::VectorXd(D);

    // y is treated in MATLAB as a row vector, just as x:
    Eigen::VectorXd y = Eigen::VectorXd(D);
    double dHy1, dHy2;
    for(int d = 0; d < D ; d++){
      y = x; y(d) += e_;
      dHy1 = get_dH(y);

      y = x; y(d) -= e_;
      dHy2 = get_dH(y);

      ddHdx(d) = (dHy1 - dHy2)/(2*e_);
    }

  // Invert sign?
  if(this->invertsign){
    dH = -dH;
    ddHdx = -ddHdx;
  }

}


double dH_MC_local::get_dH(const Eigen::VectorXd x)
{

  int Nb = this->logP.size();
  int D  = x.size();
  
  // Get efficient innovation:
  Eigen::VectorXd Lx = Eigen::VectorXd::Zero(Nb);
  Eigen::MatrixXd dLxdx = Eigen::MatrixXd::Zero(Nb,D);
  this->dGP->efficient_innovation(x,Lx,dLxdx);


  if (!this->invertsign){
    if(MathTools::isNaN_vec(Lx))
      std::cout << "Lx has nans " << std::endl;

    if(MathTools::isNaN_mat(dLxdx))
      std::cout << "dLxdx has nans " << std::endl;
  }
  
  Eigen::VectorXd dMdx = Lx;
  Eigen::MatrixXd dVdx_aux = Eigen::MatrixXd(Nb,Nb);
  dVdx_aux = -Lx*Lx.transpose();
  Eigen::VectorXd dVdx = Eigen::VectorXd(0.5*Nb*(Nb+1));
  for(int j = 0, c = 0; j < dVdx_aux.cols() ; j++)
    for(int i = 0; i <= j; i++)
      dVdx[c++] = dVdx_aux(i,j);
  // dMM = dMdx*dMdx', which is the same as dMM = - (-Lx*Lx') = -dVdx_aux
  Eigen::MatrixXd dMM = Eigen::MatrixXd(Nb,Nb);
  dMM = -dVdx_aux;

  // // DBG:
  // size_t Ntop = 5;
  // if(x(0) == 0.54321){
  //   std::cout << "Lx.head(Ntop) = " << std::endl << Lx.head(Ntop).transpose() << std::endl;
  //   std::cout << "dMdx.head(Ntop) = " << std::endl << dMdx.head(Ntop).transpose() << std::endl;
  //   std::cout << "dVdx_aux.block(0,0,Ntop,Ntop) = " << std::endl << dVdx_aux.block(0,0,Ntop,Ntop) << std::endl;
  //   std::cout << "dVdx.head(Ntop) = " << std::endl << dVdx.head(Ntop).transpose() << std::endl;
  //   std::cout << "dMM.block(0,0,Ntop,Ntop) = " << std::endl << dMM.block(0,0,Ntop,Ntop) << std::endl;
  // }


// std::cout << "ddlogPdMdM.size() = " << ddlogPdMdM.size() << std::endl;
// std::cout << "ddlogPdMdM[0].rows() = " << ddlogPdMdM[0].rows() << std::endl;
// std::cout << "ddlogPdMdM[0].cols() = " << ddlogPdMdM[0].cols() << std::endl;
// std::cout << "ddlogPdMdM[1].rows() = " << ddlogPdMdM[1].rows() << std::endl;
  // std::chrono::seconds dura(2);
  // std::this_thread::sleep_for(dura);



  // trterm = sum(sum(bsxfun(@times,ddlogPdMdM,reshape(dMM,[1,size(dMM)])),3),2);
  // Eigen::VectorXd trterm = Eigen::VectorXd::Zero(Nb);
  // for(int i = 0 ; i < Nb ; i++ )
  //   for (int k = 0 ; k < Nb ; k++ )
  //     trterm(i) += this->ddlogPdMdM[k].row(i)*dMM.col(k);

  Eigen::VectorXd trterm = Eigen::VectorXd::Zero(Nb);
  for(int i = 0 ; i < Nb ; i++ )
      trterm(i) = (this->ddlogPdMdM[i].array()*dMM.array()).sum();

  // detchange = dlogPdV * dVdx + 0.5 * trterm; 
  Eigen::VectorXd detchange = Eigen::VectorXd(Nb);
  detchange = this->dlogPdV*dVdx + 0.5*trterm;

  // if(x(0) == 0.54321){
  //   std::cout << "trterm.head(Ntop) = " << std::endl << trterm.head(Ntop).transpose() << std::endl;
  //   std::cout << "this->ddlogPdMdM[0].block(0,0,Ntop,Ntop) = " << std::endl << this->ddlogPdMdM[0].block(0,0,Ntop,Ntop) << std::endl;
  //   std::cout << "this->ddlogPdMdM[1].block(0,0,Ntop,Ntop) = " << std::endl << this->ddlogPdMdM[1].block(0,0,Ntop,Ntop) << std::endl;
  //   std::cout << "detchange.head(Ntop) = " << std::endl << detchange.head(Ntop).transpose() << std::endl;
  // }
// std::cout << "trterm = " << std::endl << trterm << std::endl;
// std::cout << "trterm = " << trterm.transpose() << std::endl;
// std::cout << "dlogPdV = " << dlogPdV << std::endl;
// std::cout << "dVdx = " << dVdx.transpose() << std::endl;
  // stochange is a matrix NbxT
  // stochange = ( dlogPdM_ * dMdx ) * W;
  Eigen::MatrixXd stochange = Eigen::MatrixXd(Nb,W.size());
  stochange = ( this->dlogPdM * dMdx ) * this->W.transpose(); //  This is weird zeros because dlogPdM is weird zeros
  // In MATLAB, W is treated as a row vector. Eigen treats vectors as columns, therefore
  // we transpose it.
  // std::cout << "stochange = " << std::endl << stochange <<std::endl;
  // std::cout << "dlogPdM = " << std::endl << dlogPdM <<std::endl;
  // std::cout << "dMdx = " << std::endl << dMdx <<std::endl;

  // if(x(0) == 0.54321){
  //   std::cout << "this->dlogPdM.block(0,0,Ntop,Ntop) = " << std::endl << this->dlogPdM.block(0,0,Ntop,Ntop) << std::endl;
  //   std::cout << "stochange.block(0,0,Ntop,Ntop) = " << std::endl << stochange.block(0,0,Ntop,Ntop) << std::endl;
  //   std::cout << "this->W.head(Ntop) = " << std::endl << this->W.head(Ntop).transpose() << std::endl;
  // }

  Eigen::MatrixXd lPred = Eigen::MatrixXd(Nb,stochange.cols());

  // lPred = bsxfun(@plus,logP + detchange,stochange); 
  lPred = stochange.colwise() + (this->logP + detchange);
  // lPred = (logP + detchange) + stochange.colwise(); //Does not work
// std::cout << "lPred = " << lPred << std::endl;
// std::cout << "logP = " << logP.transpose() << std::endl;
// std::cout << "detchange = " << detchange.transpose() << std::endl;
// std::cout << "stochange = " << stochange << std::endl;

  // if(x(0) == 0.54321){
  //   std::cout << "lPred(1).block(0,0,Ntop,Ntop) = " << std::endl << lPred.block(0,0,Ntop,Ntop) << std::endl;
  //   std::cout << "this->logP.head(Ntop) = " << std::endl << this->logP.head(Ntop).transpose() << std::endl;
  // }

  // lselP = logsumexp(lPred); In MATLAB lselP is 1xNb
  Eigen::VectorXd lselP = Eigen::VectorXd(Nb);
  lselP = logsumexp_.logsumexp_f(lPred);
// std::cout << "lPred = " << lPred << std::endl;
  // lPred gets modified, but remains Nb x stochange.cols()
  // lPred = bsxfun(@minus,lPred,lselP);
  lPred = lPred.rowwise() - lselP.transpose();
// std::cout << "lPred = " << lPred << std::endl;
  // dHp = feval(LossFunc{:},logP,lmb,lPred,zbel); but zbel is not used inside LossFunc
  // dHp is returned by MATLAB as a row vector 1 x lPred.cols()
  Eigen::VectorXd dHp = Eigen::VectorXd(lPred.cols());
  dHp = LogLoss_.LogLoss_f(this->logP,this->lmb,lPred);
  // std::cout << "dHp = " << dHp.transpose() << std::endl;
  // std::cout << "logP = " << logP.transpose() << std::endl;
  // std::cout << "lmb = " << lmb.transpose() << std::endl;
  // std::cout << "lPred = " << lPred << std::endl;
  double dH;
  dH = dHp.sum()/dHp.size();

  // if(x(0) == 0.54321){
  //   std::cout << "lselP.head(Ntop) = " << std::endl << lselP.head(Ntop).transpose() << std::endl;
  //   std::cout << "lPred(2).block(0,0,Ntop,Ntop) = " << std::endl << lPred.block(0,0,Ntop,Ntop) << std::endl;
  //   std::cout << "this->logP.head(Ntop) = " << std::endl << this->logP.head(Ntop).transpose() << std::endl;
  //   std::cout << "this->lmb.head(Ntop) = " << std::endl << this->lmb.head(Ntop).transpose() << std::endl;
  //   std::cout << "dHp.head(Ntop) = " << std::endl << dHp.head(Ntop).transpose() << std::endl;
  //   std::cout << "dH = " << dH << std::endl;
  // }

  // TODO: implement this?
  // if ~isreal(dH); keyboard; end;

  // // amarcovalle [DBG]:
  // if(this->lmb(0) != 0.0){
  //   YAML::Node node_to_write;
  //   node_to_write.SetStyle(YAML::EmitterStyle::Block);
  //   std::cout << "@dH_MC_local() Writing data to files... " << std::endl;
  //   std::string path_to_file("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/test_LogLoss_input.yaml");
  //   node_to_write["logP"] = this->logP;
  //   node_to_write["lmb"] = this->lmb;
  //   node_to_write["lPred"] = lPred;

  //   std::ofstream fout(path_to_file);
  //   fout << node_to_write;
  //   fout.close();
  //   std::cout << "Writing succesful!" << std::endl;
  //   std::chrono::seconds dura(1);
  //   std::this_thread::sleep_for(dura);

  // }


  return dH;
}
