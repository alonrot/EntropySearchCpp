# libgp - Gaussian process library for Machine Learning
# Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
# All rights reserved.

ADD_EXECUTABLE(gpdense gp_example_dense.cc)
TARGET_LINK_LIBRARIES(gpdense gp)

ADD_EXECUTABLE(simplefunction simple_function_example.cc)
TARGET_LINK_LIBRARIES(simplefunction gp)

ADD_EXECUTABLE(max_simplefunction max_simple_function_example.cc)
TARGET_LINK_LIBRARIES(max_simplefunction gp)