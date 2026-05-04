/*
   Class for circular field agents
*/
#pragma once

#include <Eigen/Dense>
#include <gafro/gafro.hpp>
#include <sackmesser/Configuration.hpp>
#include <sackmesser/Interface.hpp>
#include <vector>

namespace ga_circular_fields_planner
{

    class Force;

    class Agent
    {
      public:
        class State
        {
          public:
            State(const gafro::Motor<double> &pose, const gafro::Twist<double> &velocity);

            virtual ~State() = default;

            void setPose(const gafro::Motor<double> &pose);

            void setVelocity(const gafro::Twist<double> &velocity);

            const gafro::Motor<double> &getPose() const;

            const gafro::Twist<double> &getVelocity() const;

            virtual std::shared_ptr<State> copy() const;

          private:
            gafro::Motor<double> pose_;

            gafro::Twist<double> velocity_;
        };

      protected:
        struct Configuration : public sackmesser::Configuration
        {
            bool load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server);

            double mass;
            double radius;
            
            std::vector<std::string> forces;
        } config_;

      protected:
        std::vector<gafro::Motor<double>> trajectory_;

        gafro::Inertia<double> inertia_;

        double planning_time_;

      public:
        Agent(const sackmesser::Interface::Ptr &interface, const std::string &name);

        Agent() = default;

        virtual ~Agent() = default;

        const Configuration &getConfiguration() const;

        const std::vector<gafro::Motor<double>> &getPath() const;

        double getDistFromGoal(const gafro::Motor<double> &pose, const gafro::Motor<double> &target) const;

        gafro::Wrench<double> computeForce(const std::shared_ptr<State> &state, const gafro::Motor<double> &target);

        // void plan(const std::shared_ptr<State> &initial_state, const gafro::Motor<double> &target, const double delta_t,
        //           const unsigned max_prediction_steps);

        std::shared_ptr<State> planStep(const std::shared_ptr<State> &state, const gafro::Motor<double> &target, const double delta_t);

        void updateTrajectory(const gafro::Motor<double> &pose);

        virtual std::shared_ptr<State> updateState(const std::shared_ptr<State> &state, const gafro::Wrench<double> &force, const double delta_t) = 0;

        virtual void updateStateInPlace(std::shared_ptr<State> &state, const gafro::Wrench<double> &force, const double delta_t) = 0;

        const std::string &getName() const;

        void clearTrajectory() {
            trajectory_.clear();
        }

        void addToTrajectory(const gafro::Motor<double> &pose) {
            trajectory_.push_back(pose);
        }

        void clearPlanningTime() {
            planning_time_ = 0.0;
        }

        void accumulatePlanningTime(const double time) {
            planning_time_ += time;
        }

        double getPlanningTime() const {
            return planning_time_;
        }

        double getRadius() const {
            return config_.radius;
        }

        const double getMaxAcceleration();
        
      private:
        sackmesser::Interface::Ptr interface_;

        std::vector<std::shared_ptr<Force>> forces_;

        std::string name_;
    };
}  // namespace ga_circular_fields_planner
