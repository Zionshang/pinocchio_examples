cmake_minimum_required(VERSION 3.5.0)
project(inverse_dynamics_external_force_test VERSION 0.1.0 LANGUAGES C CXX)

find_package(pinocchio)

add_executable(inverse_dynamics_external_force_test inverse_dynamics_external_force.cpp)
target_link_libraries(inverse_dynamics_external_force_test pinocchio::pinocchio)

