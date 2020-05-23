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
#ifndef __SOME_TEST_H__
#define __SOME_TEST_H__

#include <cmath>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include "DummyFunction.hpp"

class TestPolymorphism {
  public:
    TestPolymorphism(){};
    virtual ~TestPolymorphism(){}
    void enter_function(bfgs_optimizer::ObjectiveFunction * fun);
};

#endif /* __SOME_TEST_H__ */