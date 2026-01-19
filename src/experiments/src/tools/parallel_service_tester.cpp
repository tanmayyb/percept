#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <map>
#include <mutex>
#include <numeric>
#include <iomanip>
#include <algorithm>
#include <condition_variable>

#include "rclcpp/rclcpp.hpp"
#include "percept_interfaces/srv/agent_state_to_circ_force.hpp"

using namespace std::chrono_literals;

struct ClientStats {
    std::vector<double> latencies; 
    std::mutex mutex;
};

class ParallelServiceStressTester : public rclcpp::Node
{
public:
    ParallelServiceStressTester() : Node("parallel_service_stress_tester")
    {
        this->declare_parameter<std::vector<std::string>>("service_topics", {"agent_state_to_circ_force"});
        this->declare_parameter<int>("burst_size", 10);
        this->declare_parameter<int>("total_bursts", 5);

        auto topics = this->get_parameter("service_topics").as_string_array();
        burst_size_ = this->get_parameter("burst_size").as_int();
        total_bursts_ = this->get_parameter("total_bursts").as_int();

        callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

        for (const auto & topic : topics) {
            auto client = this->create_client<percept_interfaces::srv::AgentStateToCircForce>(
                topic, rclcpp::ServicesQoS(), callback_group_);
            clients_.push_back({topic, client});
            accumulated_stats_[topic] = std::make_shared<ClientStats>();
        }

        // Start worker threads immediately
        start_workers();
    }

    ~ParallelServiceStressTester() {
        finished_ = true;
        cv_.notify_all();
        for (auto & t : worker_threads_) {
            if (t.joinable()) t.join();
        }
    }

private:
    void start_workers() {
        for (auto & client_pair : clients_) {
            worker_threads_.emplace_back([this, client_pair]() {
                this->worker_loop(client_pair.first, client_pair.second);
            });
        }

        // Dedicated reporting thread to coordinate bursts
        std::thread([this]() {
            for (int b = 0; b < total_bursts_; ++b) {
                std::this_thread::sleep_for(1s); // Initial settle
                
                {
                    std::lock_guard<std::mutex> lock(sync_mutex_);
                    current_burst_++;
                    RCLCPP_INFO(this->get_logger(), "--- Starting Burst %d/%d ---", current_burst_, total_bursts_);
                    active_threads_ = clients_.size();
                }
                cv_.notify_all();

                // Wait for all client threads to finish this burst
                std::unique_lock<std::mutex> lock(sync_mutex_);
                cv_.wait(lock, [this] { return active_threads_ == 0; });

                report_accumulated_stats(current_burst_);
                std::this_thread::sleep_for(500ms);
            }
            RCLCPP_INFO(this->get_logger(), "Stress test complete.");
            finished_ = true;
            cv_.notify_all();
        }).detach();
    }

    void worker_loop(std::string name, rclcpp::Client<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr client) {
        int last_processed_burst = 0;

        while (rclcpp::ok() && !finished_) {
            std::unique_lock<std::mutex> lock(sync_mutex_);
            cv_.wait(lock, [this, last_processed_burst] { 
                return (current_burst_ > last_processed_burst) || finished_; 
            });

            if (finished_) break;
            last_processed_burst = current_burst_;
            lock.unlock();

            // Perform iterations for this burst
            std::vector<double> local_latencies;
            for (int i = 0; i < burst_size_; ++i) {
                double latency = call_service(name, client);
                if (latency > 0) {
                    local_latencies.push_back(latency);
                }
            }

            // Transfer to global stats
            {
                std::lock_guard<std::mutex> stat_lock(accumulated_stats_[name]->mutex);
                auto & global_latencies = accumulated_stats_[name]->latencies;
                global_latencies.insert(global_latencies.end(), local_latencies.begin(), local_latencies.end());
            }

            // Signal completion of burst for this thread
            {
                std::lock_guard<std::mutex> sync_lock(sync_mutex_);
                active_threads_--;
                if (active_threads_ == 0) cv_.notify_all();
            }
        }
    }

    double call_service(std::string name, rclcpp::Client<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr client) {
        if (!client->wait_for_service(1s)) return -1.0;

        auto request = std::make_shared<percept_interfaces::srv::AgentStateToCircForce::Request>();
        auto start = std::chrono::steady_clock::now();
        auto result_future = client->async_send_request(request);

        if (result_future.wait_for(3s) == std::future_status::ready) {
            auto end = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(end - start).count();
        }
        RCLCPP_ERROR(this->get_logger(), "Service %s: Timeout", name.c_str());
        return -1.0;
    }

    void report_accumulated_stats(int burst_id) {
        const int topic_w = 60;
        const int col_w = 15;
        const int total_cols = 5;
        const int total_width = topic_w + (col_w * total_cols);

        std::cout << "\n" << std::string(total_width, '=') << std::endl;
        std::cout << " ACCUMULATED REPORT AFTER BURST " << burst_id << std::endl;
        std::cout << std::string(total_width, '-') << std::endl;
        
        std::cout << std::left << std::setw(topic_w) << "Topic" 
                  << std::right << std::setw(col_w) << "Total Requests" 
                  << std::setw(col_w) << "Total Time(ms)"
                  << std::setw(col_w) << "Mean Lat(ms)" 
                  << std::setw(col_w) << "Min Lat(ms)" 
                  << std::setw(col_w) << "Max Lat(ms)" << std::endl;
        
        std::cout << std::string(total_width, '-') << std::endl;

        for (auto const& [topic, stats_ptr] : accumulated_stats_) {
            std::lock_guard<std::mutex> lock(stats_ptr->mutex);
            auto & latencies = stats_ptr->latencies;

            std::cout << std::left << std::setw(topic_w) << topic;

            if (latencies.empty()) {
                std::cout << std::right << std::setw(col_w * total_cols) << "NO DATA" << std::endl;
                continue;
            }

            double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
            double mean = sum / latencies.size();
            double min_v = *std::min_element(latencies.begin(), latencies.end());
            double max_v = *std::max_element(latencies.begin(), latencies.end());

            std::cout << std::right << std::fixed << std::setprecision(3)
                      << std::setw(col_w) << latencies.size()
                      << std::setw(col_w) << sum
                      << std::setw(col_w) << mean 
                      << std::setw(col_w) << min_v 
                      << std::setw(col_w) << max_v << std::endl;
        }
        std::cout << std::string(total_width, '=') << "\n" << std::endl;
    }

    struct ClientEntry {
        std::string first;
        rclcpp::Client<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr second;
    };

    std::vector<ClientEntry> clients_;
    std::vector<std::thread> worker_threads_;
    std::map<std::string, std::shared_ptr<ClientStats>> accumulated_stats_;
    
    rclcpp::CallbackGroup::SharedPtr callback_group_;
    int burst_size_;
    int total_bursts_;
    
    // Synchronization
    std::condition_variable cv_;
    std::mutex sync_mutex_;
    int current_burst_ = 0;
    int active_threads_ = 0;
    bool finished_ = false;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ParallelServiceStressTester>();
    
    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(), 
        std::thread::hardware_concurrency()
    );
    
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}