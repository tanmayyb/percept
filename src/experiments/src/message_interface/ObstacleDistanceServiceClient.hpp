#pragma once

#include <ga_circular_fields_planner/ObstacleDistanceInterface.hpp>
#include <gafro/gafro.hpp>
#include <percept_interfaces/srv/agent_pose_to_min_obstacle_dist.hpp>
#include <sackmesser_ros2/CallbackClient.hpp>
#include <sackmesser_ros2/Interface.hpp>

namespace message_interface
{

    class ObstacleDistanceServiceClient
      : public sackmesser_ros::CallbackClient<ga_circular_fields_planner::ObstacleDistanceRequest, ga_circular_fields_planner::ObstacleDistanceResponse,
                                              percept_interfaces::srv::AgentPoseToMinObstacleDist>
    {
        using Message = percept_interfaces::srv::AgentPoseToMinObstacleDist;

      public:
        ObstacleDistanceServiceClient(sackmesser_ros::Interface *interface, const std::string &ns, const std::string &callback_request,
                                   const std::string &callback_response);

        virtual ~ObstacleDistanceServiceClient();

      protected:
        Message::Request::SharedPtr encodeRequest(const ga_circular_fields_planner::ObstacleDistanceRequest &request) const;

        ga_circular_fields_planner::ObstacleDistanceResponse decodeResponse(const Message::Response::SharedPtr response_msg) const;

      protected:
      private:
    };

}  // namespace message_interface