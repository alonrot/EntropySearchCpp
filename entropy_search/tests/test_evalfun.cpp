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
#include "ReadYamlParameters.hpp"

// For Debugging:
    #include <chrono>
    #include <thread>

int main (int argc, char const *argv[])
{

  std::cout << "Loading input parameters" << std::endl;
  std::cout << "========================" << std::endl;
  std::cout << "Path: " << YAML_CONFIG_PATH << std::endl;
  std::cout << "Note: This is defined in the CMakeLists.txt" << std::endl;

  // Load parameters:
  ReadYamlParameters params = ReadYamlParameters(std::string(YAML_CONFIG_PATH) + "input_parameters.yaml"); // this is defined in the CMakeLists.txt
  
  // Standard options:
  size_t  Dim             = params.get<size_t>("Dim");
  size_t  Ndiv_plot       = params.get<size_t>("Ndiv_plot");
  std::string name_evalfun = params.get<std::string>("name_evalfun");
  double  xmin_s          = params.get<double>("xmin_s");
  double  xmax_s          = params.get<double>("xmax_s");

  // ObjectiveFunction:
  std::string path2evalfun(std::string(YAML_CONFIG_PATH) + name_evalfun);
  std::shared_ptr<bfgs_optimizer::ObjectiveFunction> real_system = std::make_shared<RealSystem>(Dim,Ndiv_plot,path2evalfun);

  std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
  std::srand(nanoseconds.count());

  // Add data:
  size_t Ndata_init = 3;
  if(Ndata_init > 0){
    Eigen::MatrixXd X(Ndata_init,Dim);
    Eigen::VectorXd foo(Dim);
    Eigen::VectorXd Y(Ndata_init);
    MathTools::sample_unif_mat(xmin_s,xmax_s,X);

    for(size_t i = 0; i < Ndata_init; ++i)
      real_system->evaluate(X.row(i),&Y(i),&foo);

    std::cout << "X = " << std::endl;
    std::cout << X << std::endl;
    std::cout << "Y = " << std::endl;
    std::cout << Y << std::endl;
  }

}
