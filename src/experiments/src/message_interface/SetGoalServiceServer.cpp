#include <gafro_ros2/conversion/Motor.hpp>
#include <message_interface/SetGoalServiceServer.hpp>


namespace message_interface
{

    SetGoalServiceServer::SetGoalServiceServer(sackmesser_ros::Interface *interface, const std::string &ns, const std::string &callback)
      : sackmesser_ros::CallbackServer<ga_circular_fields_planner::SetGoalRequest, ga_circular_fields_planner::SetGoalResponse,
                                       percept_interfaces::srv::SetGoal>(interface, ns, callback)
    {}

    SetGoalServiceServer::~SetGoalServiceServer() = default;

    ga_circular_fields_planner::SetGoalResponse SetGoalServiceServer::callback(const ga_circular_fields_planner::SetGoalRequest &request)
    {
        ga_circular_fields_planner::SetGoalResponse response;

        if(!getInterface()->getCallbacks()->execute("set_goal_callback", response, request))
        {
          response.success = false;
        }

        return response;
    }

    void SetGoalServiceServer::encodeResponse(const ga_circular_fields_planner::SetGoalResponse & response,
                                                    Message::Response::SharedPtr response_msg) const
    {
        response_msg->success = response.success;
    }

    ga_circular_fields_planner::SetGoalRequest SetGoalServiceServer::decodeRequest(
      const Message::Request::SharedPtr request_msg) const
    {
        ga_circular_fields_planner::SetGoalRequest request;

        request.joint_positions = Eigen::Vector<double, 7>(
                request_msg->joint_positions[0], 
                request_msg->joint_positions[1], 
                request_msg->joint_positions[2], 
                request_msg->joint_positions[3], 
                request_msg->joint_positions[4], 
                request_msg->joint_positions[5],
                request_msg->joint_positions[6]
        );

        return request;
    }

}  // namespace message_interface

REGISTER_CLASS(sackmesser_ros::base::CallbackServer, message_interface::SetGoalServiceServer, "set_goal_server");