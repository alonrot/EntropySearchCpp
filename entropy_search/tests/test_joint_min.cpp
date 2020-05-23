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
#include "JointMin.hpp"
#include "ReadYamlParameters.hpp"

int main (int argc, char const *argv[])
{

  size_t Nb = 50;
  size_t Ntop = 5;

  Eigen::VectorXd logP = Eigen::VectorXd::Zero(Nb);
  Eigen::MatrixXd dlogPdMu = Eigen::MatrixXd::Zero(Nb,Nb);
  Eigen::MatrixXd dlogPdSigma = Eigen::MatrixXd::Zero(Nb,Nb);
  std::vector<Eigen::MatrixXd> dlogPdMudMu(Nb);
  for(size_t i=0;i<Nb;++i)
    dlogPdMudMu[i] = Eigen::MatrixXd::Zero(Nb,Nb);

  size_t which_file = 7;
  std::string path2read("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/tests/matlab/test_sample_belief_locations_new/test_sbl_" + std::to_string(which_file) + ".yaml");
  YAML::Node node = YAML::LoadFile(path2read);
  Eigen::MatrixXd Mb_aux = node["Mb"].as<Eigen::MatrixXd>();
  Eigen::MatrixXd Vb = node["Vb"].as<Eigen::MatrixXd>();
  Eigen::VectorXd Mb = Mb_aux.col(0);
  // std::cout << "Mb = " << Mb.head(8).transpose() << std::endl;
  // std::cout << "Vb = " << Vb.block(0,0,8,8).transpose() << std::endl;

  JointMin bel_min;
  bel_min = JointMin();

  bel_min.joint_min(Mb, Vb, logP, dlogPdMu, dlogPdSigma, dlogPdMudMu);

  std::cout << "logP = " << std::endl << logP.head(8).transpose() << std::endl;
  std::cout << "dlogPdMu = " << std::endl << dlogPdMu.block(0,0,Ntop,Ntop) << std::endl;
  std::cout << "dlogPdSigma = " << std::endl << dlogPdSigma.block(0,0,Ntop,Ntop) << std::endl;
  std::cout << "dlogPdMudMu[0] = " << std::endl << dlogPdMudMu[8].block(0,0,Ntop,Ntop) << std::endl;
  std::cout << "dlogPdMudMu[1] = " << std::endl << dlogPdMudMu[19].block(0,0,Ntop,Ntop) << std::endl;

  // logP.setZero();
  // dlogPdMu.setZero();
  // dlogPdSigma.setZero();
  // for(size_t i=0;i<Nb;++i)
  //   dlogPdMudMu[i].setZero();

  // bel_min.joint_min(Mb, Vb, logP, dlogPdMu, dlogPdSigma, dlogPdMudMu);
  // std::cout << "logP.head(8).transpose() = " << std::endl << logP.head(8).transpose() << std::endl;
  // std::cout << "dlogPdMu.block(0,0,8,8) = " << std::endl << dlogPdMu.block(0,0,8,8) << std::endl;
  // std::cout << "dlogPdSigma.block(0,0,8,8) = " << std::endl << dlogPdSigma.block(0,0,8,8) << std::endl;
  // std::cout << "dlogPdMudMu[0].block(0,0,ind_cut,ind_cut) = " << std::endl << dlogPdMudMu[0].block(0,0,ind_cut,ind_cut) << std::endl;
  // std::cout << "dlogPdMudMu[1].block(0,0,ind_cut,ind_cut) = " << std::endl << dlogPdMudMu[1].block(0,0,ind_cut,ind_cut) << std::endl;
  // logP.setZero();
  // dlogPdMu.setZero();
  // dlogPdSigma.setZero();
  // for(size_t i=0;i<Nb;++i)
  //   dlogPdMudMu[i].setZero();

}
