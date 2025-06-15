cmake_minimum_required(VERSION 3.5.0)
project(pinocchio_examples VERSION 0.1.0 LANGUAGES C CXX)

find_package(pinocchio)
find_package(example-robot-data)

add_executable(inverse_dynamics_with_external_force inverse_dynamics_with_external_force/inverse_dynamics_with_external_force.cpp)
target_link_libraries(inverse_dynamics_with_external_force pinocchio::pinocchio)

add_executable(compulate_collision_distance compulate_collision_distance/compulate_collision_distance.cpp)
target_link_libraries(compulate_collision_distance pinocchio::pinocchio)

add_executable(velocity_derivatives velocity_derivatives/velocity_derivatives.cpp)
target_link_libraries(velocity_derivatives pinocchio::pinocchio
    example-robot-data::example-robot-data
    cppad_lib)
