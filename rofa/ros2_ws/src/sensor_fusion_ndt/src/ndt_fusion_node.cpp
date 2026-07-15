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
#include <pcl/registration/ndt.h> // 核心：NDT头文件
#include <pcl/filters/approximate_voxel_grid.h>

class NDTFusionNode : public rclcpp::Node {
public:
    NDTFusionNode() : Node("sensor_fusion_ndt_node") {
        // 参数声明
        this->declare_parameter<std::string>("lidar_frame", "livox_frame");
        this->declare_parameter<double>("ndt_res", 1.0); // NDT网格分辨率，越大越快越粗糙

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // 订阅话题
        lidar_sub_.subscribe(this, "/livox/lidar");
        camera_sub_.subscribe(this, "/camera/depth/color/points");

        // 时间同步 (50ms 窗口)
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), lidar_sub_, camera_sub_);
        sync_->registerCallback(std::bind(&NDTFusionNode::syncCallback, this, std::placeholders::_1, std::placeholders::_2));

        fused_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/fused_point_cloud", 10);
        RCLCPP_INFO(this->get_logger(), "🚀 NDT 配准节点已启动！");
    }

private:
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> lidar_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> camera_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr fused_pub_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    void syncCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg,
                      const sensor_msgs::msg::PointCloud2::ConstSharedPtr& camera_msg) {
        
        // 1. TF 预变换：将相机点云转到雷达坐标系（基于你现有的外参）
        sensor_msgs::msg::PointCloud2 camera_transformed;
        try {
            auto tf = tf_buffer_->lookupTransform(lidar_msg->header.frame_id, camera_msg->header.frame_id, tf2::TimePointZero);
            tf2::doTransform(*camera_msg, camera_transformed, tf);
        } catch (const tf2::TransformException & ex) { return; }

        // 2. 格式转换
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_lidar(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cam(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*lidar_msg, *pcl_lidar);
        pcl::fromROSMsg(camera_transformed, *pcl_cam);

        // 3. 降采样 (这是NDT实时的关键)
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_lidar(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setLeafSize(0.1, 0.1, 0.1);
        voxel_filter.setInputCloud(pcl_lidar);
        voxel_filter.filter(*filtered_lidar);

        // 4. NDT 配置与执行
        pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
        ndt.setTransformationEpsilon(0.01);
        ndt.setStepSize(0.1);
        ndt.setResolution(this->get_parameter("ndt_res").as_double());
        ndt.setMaximumIterations(30);

        ndt.setInputSource(pcl_cam);        // 待对齐的相机点云
        ndt.setInputTarget(filtered_lidar); // 静态参考的雷达点云

        pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        ndt.align(*output_cloud);

        if (ndt.hasConverged()) {
            sensor_msgs::msg::PointCloud2 output_msg;
            pcl::toROSMsg(*output_cloud, output_msg);
            output_msg.header = lidar_msg->header;
            fused_pub_->publish(output_msg);
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<NDTFusionNode>());
    rclcpp::shutdown();
    return 0;
}