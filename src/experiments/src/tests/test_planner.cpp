#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <rclcpp/rclcpp.hpp>
#include "percept_interfaces/srv/agent_state_to_circ_force.hpp"

class ServiceCadenceProfiler : public rclcpp::Node
{
public:
    ServiceCadenceProfiler() : Node("service_cadence_profiler")
    {
        this->declare_parameter<std::string>("service_topic", "agent_state_to_circ_force");
        this->declare_parameter<int>("iterations", 100);

        service_topic_ = this->get_parameter("service_topic").as_string();
        iterations_ = this->get_parameter("iterations").as_int();

        service_ = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
            service_topic_,
            std::bind(&ServiceCadenceProfiler::handle_service, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Profiling service: %s for %ld iterations", service_topic_.c_str(), iterations_);
        timestamps_.reserve(iterations_);
    }

private:
    void handle_service(
        const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
        std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
    {
        (void)request;
        auto now = std::chrono::steady_clock::now();
        
        response->circ_force.x = 1.0;
        response->circ_force.y = 1.0;
        response->circ_force.z = 1.0;
        response->not_null = true;

        if (completed_) return;

        timestamps_.push_back(now);

        if (timestamps_.size() >= static_cast<size_t>(iterations_)) {
            completed_ = true;
            compute_and_report();
        }
    }

    void compute_and_report()
    {
        if (timestamps_.size() < 2) return;

        std::vector<double> intervals;
        intervals.reserve(timestamps_.size() - 1);

        for (size_t i = 1; i < timestamps_.size(); ++i) {
            std::chrono::duration<double, std::milli> diff = timestamps_[i] - timestamps_[i - 1];
            intervals.push_back(diff.count());
        }

        double sum = std::accumulate(intervals.begin(), intervals.end(), 0.0);
        double mean = sum / intervals.size();
        double sq_sum = std::inner_product(intervals.begin(), intervals.end(), intervals.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / intervals.size() - mean * mean);
        double min_val = *std::min_element(intervals.begin(), intervals.end());
        double max_val = *std::max_element(intervals.begin(), intervals.end());
        double avg_freq = 1000.0 / mean;

        std::cout << "\n--- Cadence Profile Report: " << service_topic_ << " ---" << std::endl;
        std::cout << "Total Requests Captured: " << timestamps_.size() << std::endl;
        std::cout << "Average Frequency:       " << std::fixed << std::setprecision(2) << avg_freq << " Hz" << std::endl;
        std::cout << "Mean Interval:           " << mean << " ms" << std::endl;
        std::cout << "Min Interval:            " << min_val << " ms" << std::endl;
        std::cout << "Max Interval:            " << max_val << " ms" << std::endl;
        std::cout << "Jitter (StdDev):         " << stdev << " ms" << std::endl;
        std::cout << "--------------------------------------------------\n" << std::endl;
    }

    std::string service_topic_;
    long int iterations_;
    bool completed_ = false;
    std::vector<std::chrono::steady_clock::time_point> timestamps_;
    rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ServiceCadenceProfiler>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}