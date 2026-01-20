#include <gafro_ros2/conversion/Motor.hpp>
#include <message_interface/ObstacleDistanceServiceClient.hpp>

namespace message_interface
{

    ObstacleDistanceServiceClient::ObstacleDistanceServiceClient(sackmesser_ros::Interface *interface, const std::string &ns,
                                                           const std::string &callback_request, const std::string &callback_response)
      : sackmesser_ros::CallbackClient<ga_circular_fields_planner::ObstacleDistanceRequest, ga_circular_fields_planner::ObstacleDistanceResponse,
                                       percept_interfaces::srv::AgentPoseToMinObstacleDist>(interface, ns, callback_request, callback_response)
    {}

    ObstacleDistanceServiceClient::~ObstacleDistanceServiceClient() = default;

    ObstacleDistanceServiceClient::Message::Request::SharedPtr ObstacleDistanceServiceClient::encodeRequest(
      const ga_circular_fields_planner::ObstacleDistanceRequest &request) const
    {
        auto request_msg = std::make_shared<ObstacleDistanceServiceClient::Message::Request>();

        request_msg->agent_pose = gafro_ros::convertToPose(request.agent_pose);

        request_msg->radius = request.radius;

        return request_msg;
    }

    ga_circular_fields_planner::ObstacleDistanceResponse ObstacleDistanceServiceClient::decodeResponse(
      const Message::Response::SharedPtr response_msg) const
    {
        ga_circular_fields_planner::ObstacleDistanceResponse response;
  
        response.distance = response_msg->distance;

        return response;
    }
}  // namespace message_interface

REGISTER_CLASS(sackmesser_ros::base::CallbackClient, message_interface::ObstacleDistanceServiceClient, "obstacle_distance_cost");