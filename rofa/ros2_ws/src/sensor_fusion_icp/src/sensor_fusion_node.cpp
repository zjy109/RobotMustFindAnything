#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

class SensorFusionNode : public rclcpp::Node {
public:
    SensorFusionNode() : Node("sensor_fusion_node") {
        // ==========================================================
        // 改进 1：万能方法 - 使用 ROS 2 参数动态获取话题名称
        // ==========================================================
        this->declare_parameter<std::string>("lidar_topic", "/livox/lidar");
        this->declare_parameter<std::string>("camera_topic", "/camera/depth/color/points");
        
        std::string lidar_topic = this->get_parameter("lidar_topic").as_string();
        std::string camera_topic = this->get_parameter("camera_topic").as_string();

        RCLCPP_INFO(this->get_logger(), "监听雷达话题: %s", lidar_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "监听相机话题: %s", camera_topic.c_str());

        // ==========================================================
        // 改进 2：初始化 TF 监听器，用于自动坐标系转换
        // ==========================================================
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // 初始化订阅器 (使用获取到的参数)
        lidar_sub_.subscribe(this, lidar_topic); 
        camera_sub_.subscribe(this, camera_topic); 

        // 设置时间同步
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), lidar_sub_, camera_sub_);
        sync_->registerCallback(std::bind(&SensorFusionNode::syncCallback, this, std::placeholders::_1, std::placeholders::_2));

        // 初始化发布器
        fused_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/fused_point_cloud", 10);
        RCLCPP_INFO(this->get_logger(), "🔥 终极版配准节点已启动！正在等待数据和 TF 树...");
    }

private:
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2> SyncPolicy;

    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> lidar_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> camera_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr fused_pub_;
    
    // TF 相关指针
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    void syncCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg,
                      const sensor_msgs::msg::PointCloud2::ConstSharedPtr& camera_msg) {

        // ==========================================================
        // 核心改动：在运行 ICP 前，先把相机点云用 TF 转换到雷达坐标系下
        // ==========================================================
        sensor_msgs::msg::PointCloud2 transformed_camera_msg;
        try {
            // 查找从 "相机坐标系" 到 "雷达坐标系" 的物理转换关系
            geometry_msgs::msg::TransformStamped transform_stamped = 
                tf_buffer_->lookupTransform(
                    lidar_msg->header.frame_id,  // 目标坐标系 (雷达)
                    camera_msg->header.frame_id, // 源坐标系 (相机)
                    tf2::TimePointZero           // 获取最新的可用 TF
                );

            // 执行转换：把相机的原始点云，扭转成雷达视角下的点云
            tf2::doTransform(*camera_msg, transformed_camera_msg, transform_stamped);
        } catch (const tf2::TransformException & ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                "等待 TF 坐标转换关系: %s", ex.what());
            return; // 如果还没拿到物理距离数据，就先跳过这一帧
        }

        // --- 步骤 A: 将 ROS 消息转换为 PCL 点云格式 ---
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_lidar(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_camera(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*lidar_msg, *pcl_lidar);
        
        // 注意这里！使用的是转换后(transformed)的相机点云
        pcl::fromROSMsg(transformed_camera_msg, *pcl_camera);

        // --- 步骤 B: 降采样与视野裁剪 ---
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_lidar(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_camera(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setLeafSize(0.05f, 0.05f, 0.05f); 
        
        voxel_filter.setInputCloud(pcl_lidar);
        voxel_filter.filter(*downsampled_lidar);
        voxel_filter.setInputCloud(pcl_camera);
        voxel_filter.filter(*downsampled_camera);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_lidar(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(downsampled_lidar);
        pass.setFilterFieldName("x");  
        pass.setFilterLimits(0.3, 5.0); 
        pass.filter(*cropped_lidar);

        if (cropped_lidar->empty() || downsampled_camera->empty()) return;

        // --- 步骤 C: 执行 ICP 毫米级精细配准 ---
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(downsampled_camera);  
        icp.setInputTarget(cropped_lidar);
        
        icp.setMaxCorrespondenceDistance(0.5); 
        icp.setMaximumIterations(50);          
        icp.setTransformationEpsilon(1e-8);    
        icp.setEuclideanFitnessEpsilon(1e-4);  

        pcl::PointCloud<pcl::PointXYZ> final_cloud;
        icp.align(final_cloud);

        // --- 步骤 D: 输出结果 ---
        if (icp.hasConverged()) {
            // RCLCPP_INFO(this->get_logger(), "✅ ICP 成功! 误差得分: %f", icp.getFitnessScore());
            sensor_msgs::msg::PointCloud2 output_msg;
            pcl::toROSMsg(final_cloud, output_msg);
            
            // 因为前面已经用 TF 转过了，所以这里名正言顺地挂载到雷达坐标系下
            output_msg.header.frame_id = lidar_msg->header.frame_id; 
            output_msg.header.stamp = lidar_msg->header.stamp;
            fused_pub_->publish(output_msg);
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SensorFusionNode>());
    rclcpp::shutdown();
    return 0;
}