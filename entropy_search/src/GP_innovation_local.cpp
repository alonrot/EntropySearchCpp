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
#include "GP_innovation_local.hpp"

// For Debugging:
    #include <chrono>
    #include <thread>

#define DBL_EPSILON 2.22045e-16


GPInnovationLocal::GPInnovationLocal(libgp::GaussianProcess * gp, const Eigen::MatrixXd & zbel) : zbel(zbel), gp(gp) {


	//K   = feval(GP.covfunc{:},GP.hyp.cov,GP.x) + exp(2*GP.hyp.lik) * eye(size(GP.x,1));
	//cK  = chol(K);
	//kbX = feval(GP.covfunc{:},GP.hyp.cov,zbel,GP.x);

    // Dimensions of the input arguments:
    // zbel must be [Nb D]

    // Initialize some parameters:
    // size_t N = this->gp->get_sampleset_size();
    // size_t Nb = zbel.rows();
    // this->cKkXb.resize(N, Nb);

    // Make sure dimensionalities agree:
    assert((this->gp->covf()).get_input_dim() == zbel.cols());

    // Get k(X,zbel), here denominated kXb, but placed directly into cKkXb, for efficiency:
    // (this->gp->covf()).getX(this->gp->get_sampleset(), zbel, this->cKkXb);

    // Compute inv(cK) * cKkXb. solveInPlace() and solveInPlace<OnTheLeft>() are the same. Basically, A.triangularView<Eigen::Upper>().solveInPlace(B) is equivalent to A \ B , i.e., inv(A)*B
    // chol(k(X,X)) is defined as cK = this->gp->getCholesky().topLeftCorner(N, N).transpose();
    // this->gp->getCholesky().topLeftCorner(N, N).transpose().triangularView<Eigen::Upper>().solveInPlace(this->cKkXb); // This changes this->cKkXb
    // AB = A.triangularView<Eigen::Upper>().solve(B) (here AB contains the solutions)
    // A.triangularView<Eigen::Upper>().solveInPlace(B) (here B is passed by reference)

    /*
     * What follows now is a little hack. There is no easy way to retrieve the noise from the libGP. 
     * I assume the noise is the LAST parameter. 
     */

     // sn2 = (this->gp->covf().get_loghyper())(this->gp->covf().get_param_dim());
     this->sn2 = (this->gp->covf().get_loghyper())(this->gp->covf().get_param_dim()-1); //amarcovalle
     this->sn2 = exp(2 * this->sn2);
}

GPInnovationLocal::~GPInnovationLocal(){
    // zbel is detroyed where it is created as well as the GP
}

void
GPInnovationLocal::update_variables(libgp::GaussianProcess * gp, Eigen::MatrixXd zbel){

    this->gp = gp;
    this->zbel = zbel;

    // Update noise:
    this->sn2 = (this->gp->covf().get_loghyper())(this->gp->covf().get_param_dim()-1); //amarcovalle
    this->sn2 = exp(2 * this->sn2);
}

void GPInnovationLocal::efficient_innovation(const Eigen::VectorXd & x, Eigen::Ref<Eigen::VectorXd> Lx, Eigen::Ref<Eigen::MatrixXd> dLxdx){

    size_t n = gp->get_sampleset_size(); // number of data points
    // size_t D = this->zbel.size(); //input dimensionality 
    size_t D = this->zbel.cols(); //input dimensionality //amarcovalle
    size_t Nb = this->zbel.rows();

    // std::cout << "[DBG]: n = " << n << std::endl; //amarcovalle
    // std::cout << "[DBG]: D = " << D << std::endl; //amarcovalle

    // std::cout << "[DBG2]: @GPInnovationLocal::efficient_innovation() " << std::endl;

    // size(zbel) = [Nb D]

    //% kernel values
    Eigen::VectorXd kxx(1); //we need to wrap kxx as a vector even though it is a scalar
    (this->gp->covf()).get(x.transpose(),x.transpose(),kxx); // This is the same as in matlab
    // kxx(0) = (gp->covf()).get(x, x);

    //kbx  = feval(GP.covfunc{:},GP.hyp.cov,zbel,x);
    Eigen::VectorXd kbx(Nb); // k(zbel,x)
    (this->gp->covf()).get(this->zbel, x.transpose(), kbx); // This is the same as in matlab

    // std::cout << "[DBG2]: kxx = " << std::endl << kxx << std::endl;
    // std::cout << "[DBG3]: kbx = " << std::endl << kbx.transpose() << std::endl;

    //% derivatives of kernel values
    //dkxb = feval(GP.covfunc_dx{:},GP.hyp.cov,x,this->zbel);
        // Simon's way:
        // Eigen::MatrixXd dkbx(kbx.size(), D); // Simon
        // gp->covf().compute_dkdx(x, kbx, this->zbel, dkbx); // Simon

        // amarcovalle:
        Eigen::MatrixXd dkbx(D,Nb); // amarcovalle, and transpose after
        libgp::SampleSet * sset_zbel = new libgp::SampleSet(D);
        for(size_t i=0;i<Nb;++i){
            sset_zbel->add(zbel.row(i),0);
        }
        gp->covf().compute_dkdx(x, kbx, sset_zbel, dkbx);
        // std::cout << "dkbx = " << std::endl << dkbx << std::endl;
        // dkbx = dkbx.transpose();

    //dkxx = feval(GP.covfunc_dx{:},GP.hyp.cov,x); // This should be zero in matlab. In the c++ code below it is.
    Eigen::VectorXd dkxx(D);
    this->gp->covf().compute_dkdx(x, kxx, x, dkxx);

    if (n < 1){ // Case tested: same as in matlab (up to probably numerical noise)
        // kxx  = feval(GP.covfunc{:},GP.hyp.cov,x);
        // kxx(0) = (gp->covf()).get(x, x) - this->sn2; // libGP adds the noise automatically (amarcovalle: this was coded by Simon)
        kxx(0) = kxx(0) - this->sn2; // amarcovalle: kxx is computed way before this. Therefore, we don't recompute it.
        if(kxx(0) < 0){
            std::cout << "[DBG]: @GPInnovationLocal::efficient_innovation()" <<std::endl;
            std::cout << "[DBG]: kxx(0) = " << kxx(0) << " < 0" <<std::endl;
            std::cout << "[DBG]: Applying saturation: kxx(0) = 0.0" <<std::endl;
            kxx(0) = 0.0;
        }
        
        //% terms of the innovation
        double sloc   = sqrt(kxx(0));
        //proj   = kbx;
        //dvloc  = dkxx;
        //dproj  = dkxb;

        // Assign numerical zero in case sloc is zero, to ensure numerical stability
        if(sloc == 0){
            sloc = DBL_EPSILON; // Numerical zero
        }
        
        // % innovation, and its derivative
        //Lx     = proj ./ sloc;
        //dLxdx  = dproj ./ sloc - 0.5 * bsxfun(@times,proj,dvloc) ./ sloc.^3;
        Lx = kbx / sloc;
        // dLxdx = dkbx.transpose() / sloc - 0.5 * kbx * dkxx / sloc / kxx(0); //TODO: Did this work originally in Matlab?! 
        // dLxdx = dkbx / sloc - 0.5 * kbx * dkxx / sloc / kxx(0); //TODO: Did this work originally in Matlab?! // amarcovalle: removed the transpose
        return;
    }

    // % kernel values
    Eigen::VectorXd kXx(n);

    // kXx  = feval(GP.covfunc{:},GP.hyp.cov,GP.x,x);
    gp->covf().getX(gp->get_sampleset(), x.transpose(),  kXx);
    // kxx  = feval(GP.covfunc{:},GP.hyp.cov,x) + exp(GP.hyp.lik * 2);
    // kxx(0) = (gp->covf()).get(x, x); // noise is already added
    // amarcovalle: kxx is computed way before this. Therefore, we don't recompute it.
    
    //dkxX = feval(GP.covfunc_dx{:},GP.hyp.cov,x,GP.x);
    Eigen::MatrixXd dkXx(D,n);
    Eigen::MatrixXd dkxX(n,D);
    // gp->covf().compute_dkdx(x, kXx.transpose(), gp->get_sampleset(), dkxX); // Simon
    gp->covf().compute_dkdx(x, kXx, gp->get_sampleset(), dkXx); // Compute first d/dx{k(X,x)}, with size(dkXxdx) = [D N], and then transpose and change sign
    dkxX = -dkXx.transpose(); // Not needed, as dLxdx is not computed
    
    // % derivatives of kernel values
    // dkxb = reshape(dkxb,[size(dkxb,2),size(dkxb,3)]);
    // dkxX = reshape(dkxX,[size(dkxX,2),size(dkxX,3)]);
    // dkxx = reshape(dkxx,[size(dkxx,2),size(dkxx,3)]);

        // Compute cK: (i.e., chol(k(X,X)))
            // // Compute Gram matrix, i.e., k(X,X)
            // Eigen::MatrixXd X(n,D);
            // libgp::SampleSet * sset = this->gp->get_sampleset();
            // for(size_t i=0;i<n;++i)
            //     X.row(i) = sset->x(i);
            // Eigen::MatrixXd cK(n,n);
            // (this->gp->covf()).get(X,X,cK); // Do here the Gram matrix k(X,X) + std_n^2*eye*(n)

            // // Gram matrix cK contains the noise. Therefore, we subtract it:
            // Eigen::VectorXd noise_var_vec = Eigen::VectorXd::Ones(n);
            // noise_var_vec.array() = noise_var_vec.array() * this->sn2;
            // Eigen::MatrixXd noise_var_mat = noise_var_vec.asDiagonal();
            // cK = cK - noise_var_mat;
            // gp->getCholesky().topLeftCorner(n, n).transpose().triangularView<Eigen::Upper>().solveInPlace(cK); // TODO: this is wrong
            // // TODO: The matrix gp->getCholesky() is internally computed using the noise, while in matlab it's not the case.
            // // This explains the differences. In order to solve this problem, gp->getCholesky() has to be reimplemented here
            // // computing L by subtracting the noise.

        // TODO: compute cK like this, instead. Like above is wrong:
            // Compute cholesky factorization of the Gram matrix, i.e., chol(k(X,X))
            Eigen::MatrixXd cK(n,n);
            cK = gp->getCholesky().topLeftCorner(n,n).transpose();

    // % terms of the innovation
    // sloc   = sqrt(kxx - kXx' * (cK \ (cK' \ kXx)));
    Eigen::VectorXd cKkXx = gp->getCholesky().topLeftCorner(n,n).transpose().triangularView<Eigen::Upper>().solve(kXx); // kXx does not get changed here
    // cKkXx = inv(cK') * kXx
    // double sloc = std::sqrt( kxx(0) - cKkXx.dot(cKkXx) ); // amarcovalle: This gives a wrong result, even when computing cKkXx without transpose().
    double sloc_opt1_inside = kxx(0) - kXx.transpose() * ( cK.inverse() * ( (cK.transpose().inverse()) * kXx ) );
    double sloc_opt2_inside = kxx(0) - cKkXx.dot(cKkXx);
    double sloc_gpvar_inside = gp->var(x.data());
    if(sloc_gpvar_inside < 0){
        std::cout << "[DBG]: @GPInnovationLocal::efficient_innovation()" <<std::endl;
        std::cout << "[DBG]: gp->var(x.data()) = " << gp->var(x.data()) << " < 0" <<std::endl;
        std::cout << "[DBG]: Applying saturation: gp->var(x.data()) = 0.0" <<std::endl;
        sloc_gpvar_inside = 0.0;
    }

    double sloc = std::sqrt(sloc_gpvar_inside);

    // Assign numerical zero in case sloc is zero, to ensure numerical stability
    if(sloc == 0){
        sloc = DBL_EPSILON; // Numerical zero
    }

    // if(sloc_opt1_inside < 0 || sloc_opt2_inside < 0){
    //     std::cout << std::endl << std::endl;
        // std::cout << "x = " << std::endl << x.transpose()<< std::endl;
        // std::cout << "kxx = " << std::endl << kxx.transpose()<<std::endl;
        // std::cout << "cKkXx = " << std::endl << cKkXx.transpose() <<std::endl;
        // std::cout << "cKkXx.dot(cKkXx) = " << std::endl << cKkXx.dot(cKkXx) <<std::endl;
        // std::cout << "kXx = " << std::endl << kXx.transpose() <<std::endl;
        // std::cout << "cK = " << std::endl << cK <<std::endl;
        // std::cout << "cK.inverse() = " << std::endl << cK.inverse() <<std::endl;
    //     std::cout << "[DBG]: Paused for debugging" <<std::endl;
    //     std::cout << "sloc_opt1_inside = " << sloc_opt1_inside <<std::endl;
    //     std::cout << "sloc_opt2_inside = " << sloc_opt2_inside <<std::endl;
    //     std::cout << "gp->var(x.data()) = " << gp->var(x.data()) <<std::endl;
    //     std::cout << "this->zbel = " << std::endl << this->zbel <<std::endl;
    //     Eigen::MatrixXd Xset = Eigen::MatrixXd::Zero(D,n);
    //     Eigen::VectorXd Yset = Eigen::VectorXd::Zero(n);
    //     for(size_t i=0;i<n;++i){
    //         Xset.col(i) = gp->get_sampleset()->x(i);
    //         Yset(i)     = gp->get_sampleset()->y(i);
    //     }
        
    //     std::cout << "Xset = " << std::endl << Xset <<std::endl;
    //     std::cout << "Yset = " << std::endl << Yset <<std::endl;
    //     std::chrono::seconds dura(1);
    //     std::this_thread::sleep_for(dura);
    //     sloc_opt1_inside = 0.0;
    // }
    // double sloc_inside = sloc_opt1_inside;
    // double sloc_opt1 = std::sqrt( sloc_opt1_inside ); // amarcovalle: probably less efficient
    // double sloc_opt2 = std::sqrt( gp->var(x.data()) ); // Trust more libgp than the above implementation
    // std::cout << "sloc_opt1 = " << std::endl << sloc_opt1 <<std::endl;
    // std::cout << "sloc_opt2 = " << std::endl << sloc_opt2 <<std::endl;
    // double sloc = std::sqrt(sloc_inside);
    // proj   = kbx - kbX * (cK \ (cK' \ kXx));
    // Eigen::VectorXd proj = kbx - this->cKkXb.transpose() * cKkXx; // THIS WAS WRONG (not the same results as in matlab)


    // proj   = kbx - kbX * (cK \ (cK' \ kXx));

        // Compute kXb (it will be transposed afterwards):
        Eigen::MatrixXd kXb(n,Nb);
        (this->gp->covf()).getX(this->gp->get_sampleset(), zbel, kXb);

        // Compute proj:
        Eigen::VectorXd proj_simon(Nb);
        proj_simon = kbx - kXb.transpose() * cK.inverse() * cKkXx; // amarcovalle: weird results

        // amarcovalle: Recompute proj
        Eigen::VectorXd proj_alon(Nb);
        proj_alon = kbx - kXb.transpose() * ( cK.inverse() * ( (cK.transpose().inverse()) * kXx ) );

        Eigen::VectorXd proj(Nb);
        proj = proj_alon;

        // proj is meant to have negative values
        // if( MathTools::any_negative_vec(proj) ){
        //     std::cout << "proj.transpose() = " << std::endl << proj.transpose() <<std::endl;
        //     throw std::runtime_error("[DBG]: @GPInnovationLocal::efficient_innovation() - proj(i) < 0, for some i");
        // }

        // std::cout << "proj_simon = " << std::endl << proj_simon.transpose() << std::endl;
        // std::cout << "proj_alon = " << std::endl << proj_alon.transpose() << std::endl;

    // std::cout << "proj = " << std::endl << proj.transpose() << std::endl;
    // std::cout << "sloc = " << std::endl << sloc << std::endl;

    // dvloc  = (dkxx' - 2 * dkxX' * (cK \ (cK' \ kXx)))';
    // dproj  = dkxb - kbX * (cK \ (cK' \ dkxX));

    // % innovation, and its derivative
    // Lx     = proj ./ sloc;
    Lx = proj / sloc;
    // dLxdx  = dproj ./ sloc - 0.5 * bsxfun(@times,proj,dvloc) ./ sloc.^3;

    // std::cout << "Lx = " << std::endl << Lx.transpose() << std::endl;

    // // Variables to compare results with test_GP_innovation_local.m in matlab
    // std::cout << "this->zbel = " << std::endl << this->zbel <<std::endl;
    // Eigen::MatrixXd Xset = Eigen::MatrixXd::Zero(D,n);
    // Eigen::VectorXd Yset = Eigen::VectorXd::Zero(n);
    // for(size_t i=0;i<n;++i){
    //     Xset.col(i) = gp->get_sampleset()->x(i);
    //     Yset(i)     = gp->get_sampleset()->y(i);
    // }
    
    // std::cout << "Xset = " << std::endl << Xset <<std::endl;
    // std::cout << "Yset = " << std::endl << Yset <<std::endl;


    if(MathTools::isNaN_vec(proj)){
        std::cout << "[DBG]: @GPInnovationLocal::efficient_innovation(): " << std::endl;
        std::cout << "proj has nans " << std::endl;
        std::cout << "proj = " << std::endl << proj.transpose() <<std::endl;
        std::cout << "kXb = " << std::endl << kXb <<std::endl;
        std::cout << "cK = " << std::endl << cK <<std::endl;
        std::cout << "cK.inverse() = " << std::endl << cK.inverse() <<std::endl;
        std::cout << "x = " << x.transpose() <<std::endl;
        std::cout << "kXx = " << std::endl << kXx.transpose() <<std::endl;
        std::cout << "kbx = " << std::endl << kbx.transpose() <<std::endl;
        std::cout << "[DBG]: Paused for debugging" <<std::endl;
        std::chrono::seconds dura(1);
        std::this_thread::sleep_for(dura);
    }

    if(std::isnan(sloc)){
        std::cout << "[DBG]: @GPInnovationLocal::efficient_innovation(): " << std::endl;
        std::cout << "sloc has nans " << std::endl;
        std::cout << "sloc_opt1_inside = " << std::endl << sloc_opt1_inside <<std::endl;
        std::cout << "sloc_opt2_inside = " << std::endl << sloc_opt2_inside <<std::endl;
        std::cout << "gp->var(x.data()) = " << std::endl << gp->var(x.data()) <<std::endl;
        std::cout << "kxx = " << std::endl << kxx <<std::endl;
        std::cout << "cKkXx = " << std::endl << cKkXx <<std::endl;
        std::cout << "cKkXx.dot(cKkXx) = " << std::endl << cKkXx.dot(cKkXx) <<std::endl;
        std::cout << "kxx(0) = " << std::endl << kxx(0) <<std::endl;
        std::cout << "kxx(0) - cKkXx.dot(cKkXx) = " << std::endl << kxx(0) - cKkXx.dot(cKkXx) <<std::endl;
        std::cout << "[DBG]: Paused for debugging" <<std::endl;
        std::chrono::seconds dura(1);
        std::this_thread::sleep_for(dura);
    }

    if(MathTools::isNaN_vec(Lx)){
        std::cout << "[DBG]: Paused for debugging, Lx" <<std::endl;
        std::chrono::seconds dura(1);
        std::this_thread::sleep_for(dura);
    }

    // std::cout << "proj = " << std::endl << proj << std::endl;
    // std::cout << "sloc = " << std::endl << sloc << std::endl;


    // TODO: remove
    // std::cout << "kbx: " << kbx << std::endl;
    // std::cout << "cKkXb: " << cKkXb << std::endl;
    // std::cout << "cKkXx: " << cKkXx << std::endl;
    // std::cout << "proj: " << proj << std::endl;
    // std::cout << "sloc: " << sloc << std::endl;
    // std::cout << "dLxdx is never used, and thus, not computed! " << std::endl; 
    dLxdx.setZero();
}



