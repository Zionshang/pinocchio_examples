cmake_minimum_required(VERSION 3.5.0)
project(pinocchio_examples VERSION 0.1.0 LANGUAGES C CXX)

find_package(pinocchio REQUIRED)
find_package(example-robot-data REQUIRED)

add_executable(inverse_dynamics_with_external_force inverse_dynamics_with_external_force/inverse_dynamics_with_external_force.cpp)
target_link_libraries(inverse_dynamics_with_external_force pinocchio::pinocchio)

add_executable(compulate_collision_distance compulate_collision_distance/compulate_collision_distance.cpp)
target_link_libraries(compulate_collision_distance pinocchio::pinocchio)

add_executable(velocity_derivatives velocity_derivatives/velocity_derivatives.cpp)
target_link_libraries(velocity_derivatives pinocchio::pinocchio
    example-robot-data::example-robot-data
    cppad_lib)
add_executable(manifold_cppad manifold_cppad/manifold_cppad.cpp)
target_link_libraries(manifold_cppad
    pinocchio::pinocchio
    example-robot-data::example-robot-data)

add_executable(print_frame_information print_frame_information/print_frame_information.cpp)
target_link_libraries(print_frame_information
    pinocchio::pinocchio
    example-robot-data::example-robot-data)

add_executable(semi_impl_euler semi_impl_euler/semi_impl_euler.cpp)
target_link_libraries(semi_impl_euler
    pinocchio::pinocchio
    example-robot-data::example-robot-data)
