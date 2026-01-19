#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "percept_interfaces/srv/set_goal.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

class TaskSequencer : public rclcpp::Node {
public:
    TaskSequencer() : Node("task_sequencer") {
        // Parameters
        this->declare_parameter("file_path", "/tmp/planner_goals.txt");
        this->declare_parameter("rad", 0.01);
        this->declare_parameter("loop", true);

        std::string file_path = this->get_parameter("file_path").as_string();
        loadGoals(file_path);

        // Service Client
        client_ = this->create_client<percept_interfaces::srv::SetGoal>("/manipulator/set_goal_callback");

        // Subscriptions
        sub_pose_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/manipulator/pose", 10, std::bind(&TaskSequencer::poseCallback, this, std::placeholders::_1));
        sub_target_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/manipulator/target", 10, std::bind(&TaskSequencer::targetCallback, this, std::placeholders::_1));

        // Control Timer (10Hz)
        timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&TaskSequencer::controlLoop, this));

        wait_for_service();
    }

private:
    void loadGoals(const std::string& path) {
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::vector<double> goal;
            double val;
            while (ss >> val) goal.push_back(val);
            if (!goal.empty()) goals_.push_back(goal);
        }
    }

    void wait_for_service() {
        while (!client_->wait_for_service(std::chrono::seconds(1))) {
            if (!rclcpp::ok()) return;
            RCLCPP_INFO(this->get_logger(), "Waiting for service...");
        }
    }

    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) { current_pose_ = msg; }
    void targetCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) { target_pose_ = msg; }

    double calculateDistance() {
        if (!current_pose_ || !target_pose_) return std::numeric_limits<double>::max();
        double dx = current_pose_->pose.position.x - target_pose_->pose.position.x;
        double dy = current_pose_->pose.position.y - target_pose_->pose.position.y;
        double dz = current_pose_->pose.position.z - target_pose_->pose.position.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    void sendGoal(const std::vector<double>& joint_positions) {
        auto request = std::make_shared<percept_interfaces::srv::SetGoal::Request>();
        request->joint_positions = joint_positions;
        
        is_waiting_for_result_ = true;
        client_->async_send_request(request, [this](rclcpp::Client<percept_interfaces::srv::SetGoal>::SharedFuture future) {
            auto response = future.get();
            if (response->success) {
                RCLCPP_INFO(this->get_logger(), "Goal accepted.");
                goal_in_progress_ = true;
            }
            is_waiting_for_result_ = false;
        });
    }

    void controlLoop() {
        if (is_waiting_for_result_ || goals_.empty()) return;

        if (!goal_in_progress_) {
            sendGoal(goals_[current_goal_idx_]);
            return;
        }

        double dist = calculateDistance();
        if (dist < this->get_parameter("rad").as_double()) {
            RCLCPP_INFO(this->get_logger(), "Goal %zu reached. Distance: %f", current_goal_idx_, dist);
            goal_in_progress_ = false;
            current_goal_idx_++;

            if (current_goal_idx_ >= goals_.size()) {
                if (this->get_parameter("loop").as_bool()) {
                    current_goal_idx_ = 0;
                } else {
                    RCLCPP_INFO(this->get_logger(), "All goals completed. Shutting down.");
                    rclcpp::shutdown();
                }
            }
        }
    }

    rclcpp::Client<percept_interfaces::srv::SetGoal>::SharedPtr client_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_pose_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_target_;
    rclcpp::TimerBase::SharedPtr timer_;

    geometry_msgs::msg::PoseStamped::SharedPtr current_pose_;
    geometry_msgs::msg::PoseStamped::SharedPtr target_pose_;

    std::vector<std::vector<double>> goals_;
    size_t current_goal_idx_ = 0;
    bool goal_in_progress_ = false;
    bool is_waiting_for_result_ = false;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TaskSequencer>());
    rclcpp::shutdown();
    return 0;
}