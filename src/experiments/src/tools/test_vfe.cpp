#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "percept_interfaces/srv/agent_state_to_circ_force.hpp"

using namespace std::chrono_literals;

class ServiceLatencyTester : public rclcpp::Node {
public:
    ServiceLatencyTester() : Node("service_latency_tester") {
        this->declare_parameter<std::string>("service_topic", "/get_apf_heuristic_circforce");
        this->declare_parameter<int>("iterations", 1000);

        service_topic = this->get_parameter("service_topic").as_string();
        iterations_ = this->get_parameter("iterations").as_int();

        client_ = this->create_client<percept_interfaces::srv::AgentStateToCircForce>(service_topic);



        while (!client_->wait_for_service(1s)) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for service.");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Waiting for service %s...", service_topic.c_str());
        }

        run_test();
    }

private:
		void run_test() {
				rclcpp::sleep_for(std::chrono::seconds(2));

				auto request = std::make_shared<percept_interfaces::srv::AgentStateToCircForce::Request>();
				
				// Dummy data initialization
				request->agent_pose.position.x = 0.0;
				request->agent_pose.position.y = 0.0;
				request->agent_pose.position.z = 0.0;
				request->agent_pose.orientation.w = 1.0;
				request->agent_velocity.x = 0.1;
				request->agent_velocity.y = -0.1;
				request->agent_velocity.z = 0.1;

				request->target_pose.position.x = 1.0;
				request->target_pose.position.y = 1.0;
				request->target_pose.position.z = 1.0;
				request->target_pose.orientation.w = 1.0;
				request->detect_shell_rad = 10.0;
				request->k_force = 1.0;
				request->max_allowable_force = 20.0;

				std::vector<double> valid_latencies;
				valid_latencies.reserve(iterations_);
				int total_attempts = 0;
				int valid_responses = 0;

				RCLCPP_INFO(this->get_logger(), "Starting test: %d iterations", iterations_);

				for (int i = 0; i < iterations_; ++i) {
						total_attempts++;
						auto start = std::chrono::high_resolution_clock::now();

						auto result_future = client_->async_send_request(request);

						if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), result_future) ==
								rclcpp::FutureReturnCode::SUCCESS) 
						{
								auto response = result_future.get();
								if (response->not_null) {
										auto end = std::chrono::high_resolution_clock::now();
										std::chrono::duration<double, std::milli> elapsed = end - start;
										valid_latencies.push_back(elapsed.count());
										valid_responses++;
								}
						}
				}

				compute_statistics(valid_latencies, total_attempts, valid_responses);
		}

		void compute_statistics(const std::vector<double>& latencies, int total, int valid) {
				double success_rate = (total > 0) ? (static_cast<double>(valid) / total) * 100.0 : 0.0;

				printf("\n--- Performance Results ---\n");
				printf("Topic:		%s", service_topic.c_str());

				if (latencies.empty()) {
						printf("Total Attempts:  %d\n", total);
						printf("Valid Responses: %d\n", valid);
						printf("Success Rate:    %.2f%%\n", success_rate);
						printf("No valid data for latency metrics.\n");
						return;
				}

				double sum = 0.0;
				double min_val = latencies[0];
				double max_val = latencies[0];

				for (double val : latencies) {
						sum += val;
						if (val < min_val) min_val = val;
						if (val > max_val) max_val = val;
				}

				double avg = sum / latencies.size();

				printf("Total Attempts:  %d\n", total);
				printf("Valid Responses: %d\n", valid);
				printf("Success Rate:    %.2f%%\n", success_rate);
				printf("Average Latency: %.4f ms\n", avg);
				printf("Min Latency:     %.4f ms\n", min_val);
				printf("Max Latency:     %.4f ms\n", max_val);
				printf("Average Rate:    %.2f Hz (valid only)\n", 1000.0 / avg);
				printf("---------------------------\n");
		}
		std::string service_topic;
    rclcpp::Client<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr client_;
    int iterations_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ServiceLatencyTester>();
    rclcpp::shutdown();
    return 0;
}