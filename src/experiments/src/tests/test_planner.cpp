#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <memory>
#include <string>
#include <map>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include "percept_interfaces/srv/agent_state_to_circ_force.hpp"

struct ProfilerData {
    std::string topic_name;
    size_t target_iterations;
    bool completed = false;
    std::vector<std::chrono::steady_clock::time_point> timestamps;
    rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service;
    std::mutex data_mutex;
};

class MultiServiceCadenceProfiler : public rclcpp::Node
{
public:
    MultiServiceCadenceProfiler(const rclcpp::NodeOptions & options) : Node("multi_service_cadence_profiler", options)
    {
        this->declare_parameter<std::vector<std::string>>("service_topics", {"agent_state_to_circ_force"});
        this->declare_parameter<int>("iterations", 100);

        auto topics = this->get_parameter("service_topics").as_string_array();
        int iterations = this->get_parameter("iterations").as_int();

        // Reentrant callback group allows the executor to run multiple service calls in parallel
        callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

        for (const auto & topic : topics) {
            auto data = std::make_shared<ProfilerData>();
            data->topic_name = topic;
            data->target_iterations = static_cast<size_t>(iterations);
            data->timestamps.reserve(iterations);

            // Pass rclcpp::ServicesQoS() and callback_group_ to enable multi-threading
            data->service = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
                topic,
                [this, data](const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
                             std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response) {
                    this->handle_service(data, request, response);
                },
                rclcpp::ServicesQoS(),
                callback_group_);

            profilers_.push_back(data);
            RCLCPP_INFO(this->get_logger(), "Initialized parallel profiler for: %s", topic.c_str());
        }
    }

private:
    void handle_service(
        std::shared_ptr<ProfilerData> data,
        const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
        std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
    {
        (void)request;
        auto now = std::chrono::steady_clock::now();

        response->circ_force.x = 1.0;
        response->circ_force.y = 1.0;
        response->circ_force.z = 1.0;
        response->not_null = true;

        {
            // Protect data during concurrent updates from the thread pool
            std::lock_guard<std::mutex> lock(data->data_mutex);
            if (data->completed) return;

            data->timestamps.push_back(now);

            if (data->timestamps.size() >= data->target_iterations) {
                data->completed = true;
                compute_and_report(data);
            }
        }
    }

    void compute_and_report(const std::shared_ptr<ProfilerData> data)
    {
        if (data->timestamps.size() < 2) return;

        std::vector<double> intervals;
        intervals.reserve(data->timestamps.size() - 1);

        for (size_t i = 1; i < data->timestamps.size(); ++i) {
            std::chrono::duration<double, std::milli> diff = data->timestamps[i] - data->timestamps[i - 1];
            intervals.push_back(diff.count());
        }

        double sum = std::accumulate(intervals.begin(), intervals.end(), 0.0);
        double mean = sum / intervals.size();
        double sq_sum = std::inner_product(intervals.begin(), intervals.end(), intervals.begin(), 0.0);
        double stdev = std::sqrt(std::abs(sq_sum / intervals.size() - mean * mean));
        double min_val = *std::min_element(intervals.begin(), intervals.end());
        double max_val = *std::max_element(intervals.begin(), intervals.end());
        double avg_freq = 1000.0 / mean;

        std::lock_guard<std::mutex> lock(output_mutex_);
        std::cout << "\n--- Cadence Profile Report: " << data->topic_name << " ---" << std::endl;
        std::cout << "Total Requests Captured: " << data->timestamps.size() << std::endl;
        std::cout << "Average Frequency:       " << std::fixed << std::setprecision(2) << avg_freq << " Hz" << std::endl;
        std::cout << "Mean Interval:           " << mean << " ms" << std::endl;
        std::cout << "Min Interval:            " << min_val << " ms" << std::endl;
        std::cout << "Max Interval:            " << max_val << " ms" << std::endl;
        std::cout << "Jitter (StdDev):         " << stdev << " ms" << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
    }

    std::vector<std::shared_ptr<ProfilerData>> profilers_;
    rclcpp::CallbackGroup::SharedPtr callback_group_;
    std::mutex output_mutex_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    
    rclcpp::NodeOptions options;

    options.use_intra_process_comms(true);
    
    auto node = std::make_shared<MultiServiceCadenceProfiler>(options);
    
    rclcpp::ExecutorOptions executor_options;

    size_t thread_count = std::thread::hardware_concurrency();

    // rclcpp::executors::MultiThreadedExecutor executor;
    rclcpp::executors::MultiThreadedExecutor executor(executor_options, thread_count);
    
    executor.add_node(node);
    
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}