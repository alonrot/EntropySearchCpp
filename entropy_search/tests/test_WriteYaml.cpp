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
#include <cmath>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <ctime>
#include "ReadYamlParameters.hpp"

void
print_std_vector(std::vector<double> vec, std::string vec_name){

  size_t D = vec.size();

  if(D >= 1){
    std::cout << vec_name << " = [";
    for(size_t i=0;i<D;++i){
      if(i < D-1){
        std::cout << vec[i] << ",";
      }
      else{
        std::cout << vec[i];
      }

    }
    std::cout << "]" << std::endl;
  }
  else
    std::cout << vec_name << " = " << vec_name[0] << std::endl;
}

int main (int argc, char const *argv[])
{

  std::srand(std::time(0));
  // sranddev();

  size_t D = 3;
  size_t Nk = 2;

  // Create variables:
  Eigen::VectorXd vec_eig = Eigen::VectorXd(D);
  Eigen::MatrixXd mat_eig = Eigen::MatrixXd(D,D+1);
  std::vector<Eigen::MatrixXd> array3D = std::vector<Eigen::MatrixXd>(Nk);
  for(size_t i = 0;i<Nk;++i){
    array3D[i] = Eigen::MatrixXd(D,D+1);
    array3D[i].setRandom();
  }

  vec_eig.setRandom();
  mat_eig.setRandom();

  // Path:
  std::string path_to_file("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/tests/test_WriteYaml.yaml");

  std::cout << "Loading input parameters" << std::endl;
  std::cout << "========================" << std::endl;
  std::cout << "Path: " << path_to_file << std::endl;


  // YAML::Emitter node_to_emit; // This corresponds to the lod API. Better not to use it
  YAML::Node node_to_write;

  node_to_write.SetStyle(YAML::EmitterStyle::Block);
  node_to_write["vec"] = vec_eig;
  node_to_write["mat"] = mat_eig;
  node_to_write["array3D"] = array3D;

  // Write to file:
  std::ofstream fout(path_to_file);
  fout << node_to_write;
  fout.close();

  // Read now the file:
  YAML::Node node_to_read = YAML::LoadFile(path_to_file); 
  // Note:  LoadFile is different from Load or from Node. 
  //        The expression vec_read = node_to_read["vec"].as<std::vector<double>>(); won't work
  //        More info in https://github.com/jbeder/yaml-cpp/wiki/Tutorial

  // Read the fields:
  Eigen::VectorXd vec_read_eigen;
  vec_read_eigen = node_to_read["vec"].as<Eigen::VectorXd>();

  Eigen::MatrixXd mat_read_eigen;
  mat_read_eigen = node_to_read["mat"].as<Eigen::MatrixXd>();

  std::vector<Eigen::MatrixXd> array3D_read_eigen;
  array3D_read_eigen = node_to_read["array3D"].as<std::vector<Eigen::MatrixXd>>();
  

  std::cout << "Comparison writing/reading Eigen Vector and Matrix" << std::endl;
  std::cout << "==========" << std::endl;
  std::cout << "vec_eig = " << vec_eig.transpose() << std::endl;
  std::cout << "vec_read_eigen = " << vec_read_eigen.transpose() << std::endl;
  std::cout << "mat_eig = " << mat_eig << std::endl;
  std::cout << "mat_read_eigen = " << mat_read_eigen << std::endl;
  std::cout << "array3D[0] = " << array3D[0] << std::endl;
  std::cout << "array3D_read_eigen[0] = " << array3D_read_eigen[0] << std::endl;

  // Test with std::vector
  std::vector<double> vec_std(vec_eig.data(), vec_eig.data() + vec_eig.rows() * vec_eig.cols());
  std::vector<double> vec_read = std::vector<double>(D);
  vec_read = node_to_read["vec"].as<std::vector<double>>();
  print_std_vector(vec_read,"vec_read");
  print_std_vector(vec_std,"vec_std");

  // Simple test for 3D arrays written from matlab:
  YAML::Node node_to_read_from_mat = YAML::LoadFile("/Users/alonrot/MPI/projects_WIP/probabilistic_numerics/EntropySearch/tests/test_WriteYaml_from_matlab.yaml"); 
  std::vector<Eigen::MatrixXd> my_3Darray_mat;
  my_3Darray_mat = node_to_read_from_mat["my_3Darray"].as<std::vector<Eigen::MatrixXd>>();
  std::cout << "\nmy_3Darray_mat[0]" << std::endl << my_3Darray_mat[0] << std::endl;
  std::cout << "my_3Darray_mat[1]" << std::endl << my_3Darray_mat[1] << std::endl;
  std::cout << "my_3Darray_mat[2]" << std::endl << my_3Darray_mat[2] << std::endl;

}
