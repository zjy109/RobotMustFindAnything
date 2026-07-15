cmake_minimum_required(VERSION 3.8)
project(sensor_fusion_ndt)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(message_filters REQUIRED)
find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(ndt_fusion_node src/ndt_fusion_node.cpp)
ament_target_dependencies(ndt_fusion_node
  rclcpp sensor_msgs tf2 tf2_ros tf2_sensor_msgs pcl_conversions message_filters
)
target_link_libraries(ndt_fusion_node ${PCL_LIBRARIES})

install(TARGETS ndt_fusion_node DESTINATION lib/${PROJECT_NAME})
ament_package()