#pragma once

#include <ga_circular_fields_planner/ObstacleForceInterface.hpp>
#include <gafro/gafro.hpp>
#include <percept_interfaces/srv/agent_state_to_circ_force.hpp>
#include <sackmesser_ros2/CallbackServer.hpp>
#include <sackmesser_ros2/Interface.hpp>

namespace message_interface
{

    class ObstacleForceServiceServer
      : public sackmesser_ros::CallbackServer<ga_circular_fields_planner::ObstacleForceRequest, ga_circular_fields_planner::ObstacleForceResponse,
                                              percept_interfaces::srv::AgentStateToCircForce>
    {
        using Message = percept_interfaces::srv::AgentStateToCircForce;

      public:
        ObstacleForceServiceServer(sackmesser_ros::Interface *interface, const std::string &ns, const std::string &callback);

        virtual ~ObstacleForceServiceServer();

        ga_circular_fields_planner::ObstacleForceResponse callback(const ga_circular_fields_planner::ObstacleForceRequest &request);

      protected:
        void encodeResponse(const ga_circular_fields_planner::ObstacleForceResponse &response, Message::Response::SharedPtr response_msg) const;

        ga_circular_fields_planner::ObstacleForceRequest decodeRequest(const Message::Request::SharedPtr request_msg) const;

      protected:
      private:
    };

}  // namespace message_interface