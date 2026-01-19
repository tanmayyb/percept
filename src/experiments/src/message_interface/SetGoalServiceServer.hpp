#pragma once

#include <ga_circular_fields_planner/SetGoalInterface.hpp>

#include <gafro/gafro.hpp>
#include <percept_interfaces/srv/set_goal.hpp>

#include <sackmesser_ros2/CallbackServer.hpp>
#include <sackmesser_ros2/Interface.hpp>


namespace message_interface
{

    class SetGoalServiceServer
      : public sackmesser_ros::CallbackServer<ga_circular_fields_planner::SetGoalRequest, ga_circular_fields_planner::SetGoalResponse,
                                              percept_interfaces::srv::SetGoal>
    {
        using Message = percept_interfaces::srv::SetGoal;

      public:
        SetGoalServiceServer(sackmesser_ros::Interface *interface, const std::string &ns, const std::string &callback);

        virtual ~SetGoalServiceServer();

        ga_circular_fields_planner::SetGoalResponse callback(const ga_circular_fields_planner::SetGoalRequest &request);

      protected:
        void encodeResponse(const ga_circular_fields_planner::SetGoalResponse &response, Message::Response::SharedPtr response_msg) const;

        ga_circular_fields_planner::SetGoalRequest decodeRequest(const Message::Request::SharedPtr request_msg) const;

      protected:
      private:
    };

}  // namespace message_interface