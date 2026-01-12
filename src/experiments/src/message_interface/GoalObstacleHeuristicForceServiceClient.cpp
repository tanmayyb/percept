#include <gafro_ros2/conversion/Motor.hpp>
#include <message_interface/GoalObstacleHeuristicForceServiceClient.hpp>

namespace message_interface
{

    GoalObstacleHeuristicForceServiceClient::GoalObstacleHeuristicForceServiceClient(sackmesser_ros::Interface *interface, const std::string &ns,
                                                           const std::string &callback_request, const std::string &callback_response)
      : sackmesser_ros::CallbackClient<ga_circular_fields_planner::ObstacleForceRequest, ga_circular_fields_planner::ObstacleForceResponse,
                                       percept_interfaces::srv::AgentStateToCircForce>(interface, ns, callback_request, callback_response)
    {}

    GoalObstacleHeuristicForceServiceClient::~GoalObstacleHeuristicForceServiceClient() = default;

    GoalObstacleHeuristicForceServiceClient::Message::Request::SharedPtr GoalObstacleHeuristicForceServiceClient::encodeRequest(
      const ga_circular_fields_planner::ObstacleForceRequest &request) const
    {
        auto request_msg = std::make_shared<GoalObstacleHeuristicForceServiceClient::Message::Request>();

        request_msg->agent_pose = gafro_ros::convertToPose(request.agent_pose);
        request_msg->target_pose = gafro_ros::convertToPose(request.target_pose);

        request_msg->agent_velocity.x = request.agent_velocity.get<gafro::blades::e1i>();
        request_msg->agent_velocity.y = request.agent_velocity.get<gafro::blades::e2i>();
        request_msg->agent_velocity.z = request.agent_velocity.get<gafro::blades::e3i>();

        request_msg->detect_shell_rad = request.detect_shell_radius;
        request_msg->k_force = request.k_force;
        request_msg->max_allowable_force = request.max_allowable_force;
        
        return request_msg;
    }

    ga_circular_fields_planner::ObstacleForceResponse GoalObstacleHeuristicForceServiceClient::decodeResponse(
      const Message::Response::SharedPtr response_msg) const
    {
        ga_circular_fields_planner::ObstacleForceResponse response;

        response.wrench = gafro::Wrench<double>::Zero();

        response.wrench.set<gafro::blades::e01>(response_msg->circ_force.x);
        response.wrench.set<gafro::blades::e02>(response_msg->circ_force.y);
        response.wrench.set<gafro::blades::e03>(response_msg->circ_force.z);

        return response;
    }
}  // namespace message_interface

REGISTER_CLASS(sackmesser_ros::base::CallbackClient, message_interface::GoalObstacleHeuristicForceServiceClient, "goalobstacle_heuristic_force");