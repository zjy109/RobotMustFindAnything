cmake_minimum_required(VERSION 3.8)
project(sensor_fusion_icp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ==========================================
# 1. 寻找 ROS 2 基础依赖包
# ==========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(message_filters REQUIRED)

# ==========================================
# 2. 寻找并配置 PCL (点云库) 依赖
# ==========================================
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# ==========================================
# 3. 编译规则与链接
# ==========================================
# 生成可执行文件
add_executable(sensor_fusion_node src/sensor_fusion_node.cpp)

# 链接 ROS 2 的依赖
ament_target_dependencies(sensor_fusion_node
  rclcpp 
  sensor_msgs 
  tf2 
  tf2_ros 
  tf2_sensor_msgs
  pcl_conversions 
  pcl_ros 
  message_filters
)

# 链接 PCL 的依赖
target_link_libraries(sensor_fusion_node ${PCL_LIBRARIES})

# ==========================================
# 4. 安装规则 (确保 ros2 run 能找到节点)
# ==========================================
install(TARGETS
  sensor_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

# ==========================================
# 测试与打包选项 (保持原有逻辑)
# ==========================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()