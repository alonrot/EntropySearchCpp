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
#include "ReadYamlParameters.hpp"


ReadYamlParameters::ReadYamlParameters(std::string yaml_file_path) {

  try {
    this->node = YAML::LoadFile(yaml_file_path);
    this->read_file_failed = false;
  } 
  catch(const std::exception& e) {
    this->read_file_failed=true;
    this->error_message = e.what();
  }

  this->file_path = yaml_file_path;
}


bool 
ReadYamlParameters::failed(std::string &error_message) {

  if (!this->read_file_failed) 
    return false;

  error_message = this->error_message;
  return true;
}