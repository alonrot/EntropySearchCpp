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
#ifndef __READ_YAML_PARAM_H__
#define __READ_YAML_PARAM_H__

#include <string>
#include "yaml-cpp/yaml.h" 
#include <iostream> 
#include <fstream>
#include <Eigen/Dense>

class ReadYamlParameters {
public:
  ReadYamlParameters(std::string yaml_file_path);

  template<typename T> T get(std::string key) const {

    if (this->node[key]) 
      return this->node[key].as<T>();

    std::string error = key + " parameter not found in " + this->file_path;
    throw std::runtime_error(error);
  }

  template<typename T> T get(const char *key) const {

    return this->get<T>(std::string(key));
  }
  
  bool failed(std::string &error_message);
  
private:
  YAML::Node node;
  std::string error_message;
  bool read_file_failed;
  std::string file_path;
};

// Template to be able to read Eigen::VectorXd and Eigen::MatrixXd
// Based on the Tutorial https://github.com/jbeder/yaml-cpp/wiki/Tutorial
namespace YAML {
template<> 
struct convert<Eigen::VectorXd> {

  static Node encode(const Eigen::VectorXd & vec) {
    Node node;

    for(int i=0;i<vec.size();++i)
      node.push_back(vec(i));

    return node;
  }

  static bool decode(const Node& node, Eigen::VectorXd & vec) 
  {
    if(!node.IsSequence()) 
      return false;

    // Initialize:
    vec = Eigen::VectorXd::Zero(node.size());

    // Fill:
    for(size_t i=0;i<node.size();++i)
      vec(i) = node[i].as<double>();

    return true;
  }

};

template<> 
struct convert<Eigen::VectorXi> {

  static Node encode(const Eigen::VectorXi & vec) {
    Node node;

    for(int i=0;i<vec.size();++i)
      node.push_back(vec(i));

    return node;
  }

  static bool decode(const Node& node, Eigen::VectorXi & vec) 
  {
    if(!node.IsSequence()) 
      return false;

    // Initialize:
    vec = Eigen::VectorXi::Zero(node.size());

    // Fill:
    for(size_t i=0;i<node.size();++i)
      vec(i) = node[i].as<double>();

    return true;
  }

};

template<> 
struct convert<Eigen::MatrixXi> {

  static Node encode(const Eigen::MatrixXi & mat) {
    Node node;

    for(int i=0;i<mat.rows();++i){
      for(int j=0;j<mat.cols();++j){
        node[i].push_back(mat(i,j));
      }
    }

    return node;
  }

  static bool decode(const Node& node, Eigen::MatrixXi & mat) 
  {
    if(!node.IsSequence()) 
      return false;

    size_t Nrows = node.size();
    size_t Ncols = node[0].size();

    // Initialize:
    mat = Eigen::MatrixXi::Zero(Nrows,Ncols);

    // Fill:
    for(size_t i=0;i<Nrows;++i){
      for(size_t j=0;j<Ncols;++j){
        mat(i,j) = node[i][j].as<double>();
      }
    }

    return true;
  }

};

template<> 
struct convert<Eigen::MatrixXd> {

  static Node encode(const Eigen::MatrixXd & mat) {
    Node node;

    for(int i=0;i<mat.rows();++i){
      for(int j=0;j<mat.cols();++j){
        node[i].push_back(mat(i,j));
      }
    }

    return node;
  }

  static bool decode(const Node& node, Eigen::MatrixXd & mat) 
  {
    if(!node.IsSequence()) 
      return false;

    size_t Nrows = node.size();
    size_t Ncols = node[0].size();

    // Initialize:
    mat = Eigen::MatrixXd::Zero(Nrows,Ncols);

    // Fill:
    for(size_t i=0;i<Nrows;++i){
      for(size_t j=0;j<Ncols;++j){
        mat(i,j) = node[i][j].as<double>();
      }
    }

    return true;
  }

};

template<> 
struct convert<std::vector<Eigen::VectorXd>> {

  static Node encode(const std::vector<Eigen::VectorXd> & array2D) {
    Node node;

    int Nrows = array2D.size();

    for(int i=0;i<Nrows;++i){
      for(int j=0;j<array2D[i].size();++j){
        node[i][j] = array2D[i](j);
      }
    }

    return node;
  }

  static bool decode(const Node& node, std::vector<Eigen::VectorXd> & array2D)
  {
    if(!node.IsSequence()) 
      return false;

    int Nrows = node.size();
    int Ncols;

    // Initialize:
    array2D.resize(Nrows);

    // Fill:
    for(int i=0;i<Nrows;++i){
      Ncols = node[i].size();
      array2D[i] = Eigen::VectorXd(Ncols);
      for(int j=0;j<Ncols;++j){
        array2D[i](j) = node[i][j].as<double>();
      }
    }

    return true;
  }

};

template<> 
struct convert<std::vector<Eigen::MatrixXd>> {

  static Node encode(const std::vector<Eigen::MatrixXd> & array3D) {
    Node node;

    size_t Nk = array3D.size();
    size_t Nrows = array3D[0].rows();
    size_t Ncols = array3D[0].cols();

    for(size_t k=0;k<Nk;++k){
      for(size_t i=0;i<Nrows;++i){
        for(size_t j=0;j<Ncols;++j){
          node[k][i][j] = array3D[k](i,j);
        }
      }
    }

    return node;
  }

  static bool decode(const Node& node, std::vector<Eigen::MatrixXd> & array3D) 
  {
    if(!node.IsSequence()) 
      return false;

    size_t Nk    = node.size();
    size_t Nrows = node[0].size();
    size_t Ncols = node[0][0].size();

    // Initialize:
    array3D = std::vector<Eigen::MatrixXd>(Nk);
    for(size_t k=0;k<Nk;++k){
      array3D[k] = Eigen::MatrixXd::Zero(Nrows,Ncols);
    }

    for(size_t k=0;k<Nk;++k){
      for(size_t i=0;i<Nrows;++i){
        for(size_t j=0;j<Ncols;++j){
          array3D[k](i,j) = node[k][i][j].as<double>();
        }
      }
    }

    return true;
  }

};

} // namespace YAML

#endif // __READ_YAML_PARAM_H__

