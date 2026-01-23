#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include "percept_interfaces/msg/pointcloud1_m.hpp"

namespace perception
{
  class ShmMonitorNode : public rclcpp::Node
  {
  public:
    ShmMonitorNode() : Node("shm_monitor_node"), msg_count_(0)
    {
      this->declare_parameter("frame_id", "panda_link0");
      this->declare_parameter("shm_topic", "/pointcloud_shm");
      this->declare_parameter("output_topic", "/pointcloud");
      this->declare_parameter("publish_pc2", false);

      frame_id_ = this->get_parameter("frame_id").as_string();
      bool publish_pc2 = this->get_parameter("publish_pc2").as_bool();
      std::string shm_topic = this->get_parameter("shm_topic").as_string();
      std::string output_topic = this->get_parameter("output_topic").as_string();

      sub_ = this->create_subscription<percept_interfaces::msg::Pointcloud1M>(
        shm_topic, 
        rclcpp::SensorDataQoS(), 
        std::bind(&ShmMonitorNode::callback, this, std::placeholders::_1));

      if (publish_pc2) {
        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic, 10);
      }

      timer_ = this->create_wall_timer(
        std::chrono::seconds(1), 
        std::bind(&ShmMonitorNode::monitorFrequency, this));
    }

  private:
    void callback(const percept_interfaces::msg::Pointcloud1M::SharedPtr msg)
    {
      msg_count_++;

      if (!pub_) return;

      auto pc2_msg = sensor_msgs::msg::PointCloud2();
      pc2_msg.header.stamp = this->now();
      pc2_msg.header.frame_id = frame_id_;

      sensor_msgs::PointCloud2Modifier modifier(pc2_msg);
      modifier.setPointCloud2FieldsByString(1, "xyz");
      modifier.resize(msg->num_points);

      sensor_msgs::PointCloud2Iterator<float> iter_x(pc2_msg, "x");
      sensor_msgs::PointCloud2Iterator<float> iter_y(pc2_msg, "y");
      sensor_msgs::PointCloud2Iterator<float> iter_z(pc2_msg, "z");

      for (size_t i = 0; i < msg->num_points; ++i) {
        *iter_x = msg->x[i];
        *iter_y = msg->y[i];
        *iter_z = msg->z[i];
        ++iter_x; ++iter_y; ++iter_z;
      }

      pub_->publish(pc2_msg);
    }

    void monitorFrequency()
    {
      RCLCPP_INFO(this->get_logger(), "Input SHM Frequency: %u Hz", msg_count_);
      msg_count_ = 0;
    }

    rclcpp::Subscription<percept_interfaces::msg::Pointcloud1M>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::string frame_id_;
    uint32_t msg_count_;
  };
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<perception::ShmMonitorNode>());
  rclcpp::shutdown();
  return 0;
}