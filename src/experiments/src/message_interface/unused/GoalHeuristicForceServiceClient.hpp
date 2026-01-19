#pragma once

#include <ga_circular_fields_planner/ObstacleForceInterface.hpp>
#include <gafro/gafro.hpp>
#include <percept_interfaces/srv/agent_state_to_circ_force.hpp>
#include <sackmesser_ros2/CallbackClient.hpp>
#include <sackmesser_ros2/Interface.hpp>

namespace message_interface
{

    class GoalHeuristicForceServiceClient
      : public sackmesser_ros::CallbackClient<ga_circular_fields_planner::ObstacleForceRequest, ga_circular_fields_planner::ObstacleForceResponse,
                                              percept_interfaces::srv::AgentStateToCircForce>
    {
        using Message = percept_interfaces::srv::AgentStateToCircForce;

      public:
        GoalHeuristicForceServiceClient(sackmesser_ros::Interface *interface, const std::string &ns, const std::string &callback_request,
                                   const std::string &callback_response);

        virtual ~GoalHeuristicForceServiceClient();

      protected:
        Message::Request::SharedPtr encodeRequest(const ga_circular_fields_planner::ObstacleForceRequest &request) const;

        ga_circular_fields_planner::ObstacleForceResponse decodeResponse(const Message::Response::SharedPtr response_msg) const;

      protected:
      private:
    };

}  // namespace message_interface