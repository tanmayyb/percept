#include <ga_circular_fields_planner/Obstacle.hpp>

namespace ga_circular_fields_planner
{

    Obstacle::Obstacle(const std::string &name, const Eigen::Vector3d &position, const Eigen::Vector3d &velocity, const double &radius,
                       bool is_dynamic,              // = false
                       const double &angular_speed)  // = 0true
      : name_{ name },                               //
        position_{ position },                       //
        velocity_{ velocity },                       //
        radius_{ radius },                           //
        center_{ position },                         //
        is_dynamic_{ is_dynamic },                   //
        angular_speed_{ angular_speed }
    {
        direction_ = (std::rand() % 2 == 0) ? -1 : 1;
    }

    Obstacle::Obstacle(const Eigen::Vector3d &position, const double &radius)
      : position_{ position },  //
        radius_{ radius },      //
        velocity_{ 0, 0, 0 },   //
        name_{ "" },            //
        center_{ position },    //
        is_dynamic_{ false },   //
        angular_speed_{ 0.0 }
    {}

    Obstacle::Obstacle(const Eigen::Vector3d &position, const Eigen::Vector3d &velocity, const double &radius, bool is_dynamic,
                       const double &angular_speed)
      : position_{ position },      //
        radius_{ radius },          //
        velocity_{ velocity },      //
        name_{ "" },                //
        center_{ position },        //
        is_dynamic_{ is_dynamic },  //
        angular_speed_{ angular_speed }
    {}

    Obstacle::Obstacle()
      : position_{ 0, 0, 0 },  //
        radius_{ 0 },          //
        velocity_{ 0, 0, 0 },  //
        name_{ "" },           //
        center_{ 0, 0, 0 },    //
        is_dynamic_{ false },  //
        angular_speed_{ 0.0 }
    {}

    void Obstacle::setPosition(const Eigen::Vector3d &position)
    {
        position_ = position;
    }

    void Obstacle::setVelocity(const Eigen::Vector3d &velocity)
    {
        velocity_ = velocity;
    }

    void Obstacle::setDynamic(bool is_dynamic)
    {
        is_dynamic_ = is_dynamic;
    }

    bool Obstacle::isDynamic() const
    {
        return is_dynamic_;
    }

    void Obstacle::setCenter(const Eigen::Vector3d &center)
    {
        center_ = center;
    }

    void Obstacle::setAngularSpeed(const double &angular_speed)
    {
        angular_speed_ = angular_speed;
    }

    // Update position for dynamic obstacles
    void Obstacle::updatePosition(const double &delta_time)
    {
        if (is_dynamic_)
        {
            // 使用障碍物的中心位置（即 YAML 中的 position）作为圆心
            if (center_.isZero())
            {
                center_ = position_;  // 初始化圆心为初始位置
            }

            double angle = direction_ * angular_speed_ * delta_time;
            // double angle =  angular_speed_ * delta_time;
            Eigen::Vector3d offset = position_ - center_;
            if (offset.norm() < 1e-6)
            {
                offset = 2 * Eigen::Vector3d(radius_, 0, 0);
            }

            Eigen::Matrix3d rotation;
            rotation = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ());
            Eigen::Vector3d rotated_offset = rotation * offset;

            position_ = center_ + rotated_offset;
        }
        else
        {
            // 静态障碍物只按线性速度移动
            position_ += velocity_ * delta_time;
        }
    }

    const std::string &Obstacle::getName() const
    {
        return name_;
    }

    const Eigen::Vector3d &Obstacle::getPosition() const
    {
        return position_;
    }

    const Eigen::Vector3d &Obstacle::getVelocity() const
    {
        return velocity_;
    }

    const double &Obstacle::getRadius() const
    {
        return radius_;
    }

    const Eigen::Vector3d &Obstacle::getCenter() const
    {
        return center_;
    }

    const double &Obstacle::getAngularSpeed() const
    {
        return angular_speed_;
    }

    const Obstacle *Obstacle::getClosestNeighbour() const
    {
        return closest_neighbour_;
    }

    void Obstacle::setClosestNeighbour(const Obstacle *obstacle)
    {
        closest_neighbour_ = obstacle;
    }

}  // namespace ga_circular_fields_planner
