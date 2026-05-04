/*
   Class for obstacle information
*/
#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace ga_circular_fields_planner
{

    class Obstacle
    {
      public:
        // Constructors
        Obstacle(const std::string &name, const Eigen::Vector3d &pos, const Eigen::Vector3d &vel, const double &rad, bool is_dynamic,
                 const double &angular_speed);

        Obstacle(const Eigen::Vector3d &pos, const double &rad);

        Obstacle(const Eigen::Vector3d &pos, const Eigen::Vector3d &vel, const double &rad, bool is_dynamic = false,
                 const double &angular_speed = 0.0);

        Obstacle();

        //

        void setPosition(const Eigen::Vector3d &pos);

        void setVelocity(const Eigen::Vector3d &vel);

        void setDynamic(bool is_dynamic);

        void setCenter(const Eigen::Vector3d &center);

        void setAngularSpeed(const double &angular_speed);

        // Update position for dynamic obstacles
        void updatePosition(const double &delta_time);

        //

        const std::string &getName() const;

        const Eigen::Vector3d &getPosition() const;

        const Eigen::Vector3d &getVelocity() const;

        const double &getRadius() const;

        bool isDynamic() const;

        const Eigen::Vector3d &getCenter() const;

        const double &getAngularSpeed() const;

        const Obstacle *getClosestNeighbour() const;

        void setClosestNeighbour(const Obstacle *obstacle);

      private:
        std::string name_;

        Eigen::Vector3d position_;

        Eigen::Vector3d velocity_;

        double radius_;

        Eigen::Vector3d center_;  // For dynamic obstacles

        bool is_dynamic_;

        double angular_speed_;  // Angular speed

        int direction_;  // 顺时针或逆时针

        const Obstacle *closest_neighbour_;
    };
}  // namespace ga_circular_fields_planner
