#include <ga_circular_fields_planner/ManipulatorAgent.hpp>

namespace ga_circular_fields_planner
{
    ManipulatorAgent::ManipulatorAgent(const sackmesser::Interface::Ptr &interface, const std::string &name) : Agent(interface, name) {}

    std::shared_ptr<Agent::State> ManipulatorAgent::updateState(const std::shared_ptr<Agent::State> &state_ptr, const gafro::Wrench<double> &force,
                                                                const double delta_t)
    {
        auto state = std::dynamic_pointer_cast<State>(state_ptr);

        gafro::Motor<double> ee_motor = panda_.getEEMotor(state->getJointPosition());

        auto jacobian = panda_.getGeometricJacobian(state->getJointPosition(), ee_motor).embed();

        Eigen::Matrix<double, 7, 6> inverse_jacobian =
          (jacobian.transpose() * jacobian + 1e-5 * Eigen::Matrix<double, 7, 7>::Identity()).inverse() * jacobian.transpose();

        gafro::Twist<double> acceleration = inertia_(force);

        gafro::Twist<double> velocity = state->getVelocity() + 0.5 * delta_t * acceleration;

        Eigen::Vector<double, 7> joint_velocity = inverse_jacobian * velocity.vector();
        Eigen::Vector<double, 7> joint_position = state->getJointPosition() + delta_t * joint_velocity;

        gafro::Motor<double> pose = panda_.getEEMotor(joint_position);

        return std::make_shared<State>(joint_position, joint_velocity, pose, velocity);
    }

    

    void ManipulatorAgent::updateStateInPlace(std::shared_ptr<Agent::State> &state_ptr, 
                                                  const gafro::Wrench<double> &force, 
                                                  const double delta_t)
    {
      auto state = std::static_pointer_cast<State>(state_ptr);

      gafro::Motor<double> ee_motor = panda_.getEEMotor(state->getJointPosition());

      auto jacobian = panda_.getGeometricJacobian(state->getJointPosition(), ee_motor).embed();
      
      Eigen::Matrix<double, 7, 6> inverse_jacobian =
        (jacobian.transpose() * jacobian + 1e-5 * Eigen::Matrix<double, 7, 7>::Identity()).inverse() * jacobian.transpose();

      gafro::Twist<double> acceleration = inertia_(force);

      gafro::Twist<double> velocity = state->getVelocity() + 0.5 * delta_t * acceleration;

      Eigen::Vector<double, 7> joint_velocity = inverse_jacobian * velocity.vector();

      Eigen::Vector<double, 7> joint_position = state->getJointPosition() + delta_t * joint_velocity;

      gafro::Motor<double> pose = panda_.getEEMotor(joint_position);
      
      state->setJointPosition(joint_position);

      state->setJointVelocity(joint_velocity);

      state->setPose(pose);

      state->setVelocity(velocity);
      
    }


    ManipulatorAgent::State::State(const Eigen::Vector<double, 7> &joint_position,  //
                                   const Eigen::Vector<double, 7> &joint_velocity,  //
                                   const gafro::Motor<double> &pose,                //
                                   const gafro::Twist<double> &velocity)
      : Agent::State(pose, velocity), joint_position_(joint_position), joint_velocity_(joint_velocity)
    {}

    const Eigen::Vector<double, 7> &ManipulatorAgent::State::getJointPosition() const
    {
        return joint_position_;
    }

    const Eigen::Vector<double, 7> &ManipulatorAgent::State::getJointVelocity() const
    {
        return joint_velocity_;
    }

    std::shared_ptr<Agent::State> ManipulatorAgent::State::copy() const
    {
        return std::make_shared<ManipulatorAgent::State>(joint_position_, joint_velocity_, getPose(), getVelocity());
    }

}  // namespace ga_circular_fields_planner