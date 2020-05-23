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
#include "LogLoss.hpp"
#include "ReadYamlParameters.hpp"

int main (int argc, char const *argv[])
{

  YAML::Node node = YAML::LoadFile("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/test_LogLoss_input.yaml");
  Eigen::VectorXd logP = node["logP"].as<Eigen::VectorXd>();
  Eigen::VectorXd lmb = node["lmb"].as<Eigen::VectorXd>();
  Eigen::MatrixXd lPred = node["lPred"].as<Eigen::MatrixXd>();
  Eigen::VectorXd dHp(logP.size());
  

  LogLoss logloss = LogLoss();

  dHp = logloss.LogLoss_f(logP,lmb,lPred);

  std::cout << "logP = " << logP.head(8).transpose() << std::endl;
  std::cout << "lmb = " << lmb.head(8).transpose() << std::endl;
  std::cout << "lPred = " << std::endl << lPred.block(0,0,8,8) << std::endl;
  std::cout << "dHp = " << dHp.head(8).transpose() << std::endl;

}