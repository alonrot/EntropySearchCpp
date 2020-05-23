CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(entropy_search)

# option : default = release
IF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Release)
ENDIF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
# set(CMAKE_CXX_FLAGS_RELEASE "-DCLSQUARE -Wall -O2 -fPIC")
# set(CMAKE_CXX_FLAGS_DEBUG "-DCLSQUARE -Wall -g -fPIC")

# FIND_PACKAGE(Eigen3 3.2.8 REQUIRED)
FIND_PACKAGE(Eigen3)
FIND_PACKAGE(catkin REQUIRED COMPONENTS libgp YAML_CPP)

catkin_package( 
  LIBRARIES entropy_search 
  INCLUDE_DIRS include 
  CATKIN_DEPENDS YAML_CPP libgp
)

INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include ${catkin_INCLUDE_DIRS})

# ----- Add library target -----
ADD_LIBRARY(${PROJECT_NAME}  
  src/EntropySearch.cpp
  src/dH_MC_local.cpp
  src/CostFunction.cpp
  src/DummyFunction.cpp
  src/erfinv.cpp
  src/ExpectedImprovement.cpp
  src/GP_innovation_local.cpp
  src/JointMin.cpp
  src/logdet.cpp
  src/LoggingTools.cpp
  src/LogLoss.cpp
  src/logsumexp.cpp
  src/lt_factor.cpp
  src/MathTools.cpp
  src/ReadYamlParameters.cpp
  src/SampleBeliefLocations.cpp
  src/Slice_ShrinkRank_nolog.cpp
  src/TestPolymorphism.cpp
)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${catkin_LIBRARIES})

# ----- Examples -----
ADD_EXECUTABLE(runES examples/runES.cpp)
TARGET_LINK_LIBRARIES(runES ${PROJECT_NAME})
get_filename_component(CONFIG_PATH config/ ABSOLUTE)
set_target_properties(runES PROPERTIES COMPILE_DEFINITIONS YAML_CONFIG_PATH="${CONFIG_PATH}/")

# ----- Tests -----
# INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)

# # ----- Install ES -----
# install(TARGETS runES
#   RUNTIME DESTINATION bin
#   LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
#   ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}
# )