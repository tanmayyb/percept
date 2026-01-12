#include <gafro_ros2/conversion/Motor.hpp>
#include <message_interface/ObstacleForceServiceServer.hpp>

namespace message_interface
{

    ObstacleForceServiceServer::ObstacleForceServiceServer(sackmesser_ros::Interface *interface, const std::string &ns, const std::string &callback)
      : sackmesser_ros::CallbackServer<ga_circular_fields_planner::ObstacleForceRequest, ga_circular_fields_planner::ObstacleForceResponse,
                                       percept_interfaces::srv::AgentStateToCircForce>(interface, ns, callback)
    {}

    ObstacleForceServiceServer::~ObstacleForceServiceServer() = default;

    ga_circular_fields_planner::ObstacleForceResponse ObstacleForceServiceServer::callback(const ga_circular_fields_planner::ObstacleForceRequest &)
    {
        return ga_circular_fields_planner::ObstacleForceResponse();
    }

    void ObstacleForceServiceServer::encodeResponse(const ga_circular_fields_planner::ObstacleForceResponse & /*response*/,
                                                    Message::Response::SharedPtr response_msg) const
    {
        // request_msg->agent_pose = gafro_ros::convertToPose(request.agent_pose);
        // request_msg->target_pose = gafro_ros::convertToPose(request.target_pose);

        // request_msg->agent_velocity.x = request.agent_velocity.get<gafro::blades::e1i>();
        // request_msg->agent_velocity.y = request.agent_velocity.get<gafro::blades::e2i>();
        // request_msg->agent_velocity.z = request.agent_velocity.get<gafro::blades::e3i>();

        // request_msg->detect_shell_rad = request.detect_shell_radius;
    }

    ga_circular_fields_planner::ObstacleForceRequest ObstacleForceServiceServer::decodeRequest(
      const Message::Request::SharedPtr /*request_msg*/) const
    {
        ga_circular_fields_planner::ObstacleForceRequest request;

        // request.wrench = gafro::Wrench<double>::Zero();

        // request.wrench.set<gafro::blades::e01>(response_msg->circ_force.x);
        // request.wrench.set<gafro::blades::e02>(response_msg->circ_force.y);
        // request.wrench.set<gafro::blades::e03>(response_msg->circ_force.z);

        return request;
    }
}  // namespace message_interface

REGISTER_CLASS(sackmesser_ros::base::CallbackServer, message_interface::ObstacleForceServiceServer, "obstacle_force");