/*
   Class for circular field agents
*/
#pragma once

#include <ga_circular_fields_planner/Agent.hpp>
#include <gafro_robot_descriptions/FrankaEmikaRobot.hpp>

namespace ga_circular_fields_planner
{

    class ManipulatorAgent : public Agent
    {
      public:
        class State : public Agent::State
        {
          public:
            State(const Eigen::Vector<double, 7> &joint_position,  //
                  const Eigen::Vector<double, 7> &joint_velocity,  //
                  const gafro::Motor<double> &pose,                //
                  const gafro::Twist<double> &velocity);

            const Eigen::Vector<double, 7> &getJointPosition() const;

            const Eigen::Vector<double, 7> &getJointVelocity() const;

            void setJointPosition(const Eigen::Vector<double, 7> &q) { joint_position_ = q; }

            void setJointVelocity(const Eigen::Vector<double, 7> &dq) { joint_velocity_ = dq; }

            std::shared_ptr<Agent::State> copy() const override;

          private:
            Eigen::Vector<double, 7> joint_position_;
            Eigen::Vector<double, 7> joint_velocity_;
        };

      public:
        ManipulatorAgent(const sackmesser::Interface::Ptr &interface, const std::string &name);

        virtual ~ManipulatorAgent() = default;

        void updateStateInPlace(std::shared_ptr<Agent::State> &state_ptr, 
                                const gafro::Wrench<double> &force, 
                                const double delta_t) override;

        std::shared_ptr<Agent::State> updateState(const std::shared_ptr<Agent::State> &state, const gafro::Wrench<double> &force,
                                                  const double delta_t);

      private:
        gafro::FrankaEmikaRobot<double> panda_;
    };
}  // namespace ga_circular_fields_planner
