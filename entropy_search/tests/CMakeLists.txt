# ADD_EXECUTABLE(test_WriteYaml test_WriteYaml.cpp)
# TARGET_LINK_LIBRARIES(test_WriteYaml read_yaml_parameters yaml-cpp)

# ADD_EXECUTABLE(test_evalfun test_evalfun.cpp)
# TARGET_LINK_LIBRARIES(test_evalfun dummy_function read_yaml_parameters yaml-cpp erfinv)
# get_filename_component(CONFIG_PATH ../config/ ABSOLUTE)
# set_target_properties(test_evalfun PROPERTIES COMPILE_DEFINITIONS YAML_CONFIG_PATH="${CONFIG_PATH}/")

# ADD_EXECUTABLE(test_EntropySearch test_EntropySearch.cpp)
# TARGET_LINK_LIBRARIES(test_EntropySearch GP_innovation_local dH_MC_local dummy_function gp joint_min entropy_search slice_sampling sample_belief_locations mathtools expected_improvement)

# ADD_EXECUTABLE(test_GP_innovation_local test_GP_innovation_local.cpp)
# TARGET_LINK_LIBRARIES(test_GP_innovation_local gp GP_innovation_local mathtools expected_improvement)

# ADD_EXECUTABLE(test_joint_min test_joint_min.cpp)
# TARGET_LINK_LIBRARIES(test_joint_min joint_min yaml-cpp read_yaml_parameters)

# ADD_EXECUTABLE(test_dummyfunction test_dummyfunction.cpp)
# TARGET_LINK_LIBRARIES(test_dummyfunction dummy_function gp mathtools test_polymorphism)

ADD_EXECUTABLE(test_ker_derivative tests/test_ker_derivative.cpp)
TARGET_LINK_LIBRARIES(test_ker_derivative ${PROJECT_NAME})

# ADD_EXECUTABLE(test_GPmean_optimization test_GPmean_optimization.cpp)
# TARGET_LINK_LIBRARIES(test_GPmean_optimization dummy_function gp GP_innovation_local mathtools)

# ADD_EXECUTABLE(test_dH test_dH_MC_local.cpp)
# TARGET_LINK_LIBRARIES(test_dH dH_MC_local gp GP_innovation_local dummy_function mathtools read_yaml_parameters yaml-cpp erfinv)
# get_filename_component(THIS_PATH . ABSOLUTE)
# set_target_properties(test_dH PROPERTIES COMPILE_DEFINITIONS YAML_CONFIG_FILE="${THIS_PATH}/../config/input_parameters.yaml")

# ADD_EXECUTABLE(test_dH_new_loop test_dH_new_loop.cpp)
# TARGET_LINK_LIBRARIES(test_dH_new_loop dH_MC_local gp GP_innovation_local dummy_function mathtools read_yaml_parameters yaml-cpp erfinv)


# ADD_EXECUTABLE(test_sample_belief_locations test_sample_belief_locations.cpp)
# TARGET_LINK_LIBRARIES(test_sample_belief_locations sample_belief_locations dummy_function slice_sampling mathtools expected_improvement)

# ADD_EXECUTABLE(test_slice_sampler test_Slice_ShrinkRank_nolog.cpp)
# TARGET_LINK_LIBRARIES(test_slice_sampler dummy_function slice_sampling mathtools)

# ADD_EXECUTABLE(test_EI test_EI.cpp)
# TARGET_LINK_LIBRARIES(test_EI dummy_function mathtools expected_improvement)

# ADD_EXECUTABLE(test_LogLoss test_LogLoss.cpp)
# TARGET_LINK_LIBRARIES(test_LogLoss yaml-cpp read_yaml_parameters LogLoss)