#include <gafro_ros2/conversion/Motor.hpp>
#include <message_interface/SetGoalServiceServer.hpp>


namespace message_interface
{

    SetGoalServiceServer::SetGoalServiceServer(sackmesser_ros::Interface *interface, const std::string &ns, const std::string &callback)
      : sackmesser_ros::CallbackServer<ga_circular_fields_planner::SetGoalRequest, ga_circular_fields_planner::SetGoalResponse,
                                       percept_interfaces::srv::SetGoal>(interface, ns, callback)
    {}

    SetGoalServiceServer::~SetGoalServiceServer() = default;

    ga_circular_fields_planner::SetGoalResponse SetGoalServiceServer::callback(const ga_circular_fields_planner::SetGoalRequest &)
    {
        return ga_circular_fields_planner::SetGoalResponse();
    }

    void SetGoalServiceServer::encodeResponse(const ga_circular_fields_planner::SetGoalResponse & /*response*/,
                                                    Message::Response::SharedPtr response_msg) const
    {
        // request_msg->agent_pose = gafro_ros::convertToPose(request.agent_pose);
        // request_msg->target_pose = gafro_ros::convertToPose(request.target_pose);

        // request_msg->agent_velocity.x = request.agent_velocity.get<gafro::blades::e1i>();
        // request_msg->agent_velocity.y = request.agent_velocity.get<gafro::blades::e2i>();
        // request_msg->agent_velocity.z = request.agent_velocity.get<gafro::blades::e3i>();

        // request_msg->detect_shell_rad = request.detect_shell_radius;

        response_msg->success = true;
    }

    ga_circular_fields_planner::SetGoalRequest SetGoalServiceServer::decodeRequest(
      const Message::Request::SharedPtr /*request_msg*/) const
    {
        ga_circular_fields_planner::SetGoalRequest request;

        // request.wrench = gafro::Wrench<double>::Zero();

        // request.wrench.set<gafro::blades::e01>(response_msg->circ_force.x);
        // request.wrench.set<gafro::blades::e02>(response_msg->circ_force.y);
        // request.wrench.set<gafro::blades::e03>(response_msg->circ_force.z);



        return request;
    }

}  // namespace message_interface

REGISTER_CLASS(sackmesser_ros::base::CallbackServer, message_interface::SetGoalServiceServer, "set_goal_server");