#pragma once

#include <ga_circular_fields_planner/Agent.hpp>
#include <gafro/gafro.hpp>
#include <sackmesser/FactoryClass.hpp>

namespace ga_circular_fields_planner
{
    class Cost : public sackmesser::FactoryClass<Cost, const sackmesser::Interface::Ptr &, const std::string &>
    {
      public:
        Cost() = default;

        virtual ~Cost() = default;

        virtual double computeCost(const std::shared_ptr<Agent> &agent, const gafro::Motor<double> &target) const = 0;

        virtual std::string getName() const = 0;

        virtual double getWeight() const = 0;

      protected:
      private:
    };

    class ObstacleDistanceCost : public Cost
    {
      public:
        ObstacleDistanceCost(const sackmesser::Interface::Ptr &interface, const std::string &name);

        virtual ~ObstacleDistanceCost() = default;

        double computeCost(const std::shared_ptr<Agent> &agent, const gafro::Motor<double> &target) const;

        std::string getName() const override { return "obstacle_distance_cost"; }

        double getWeight() const;

      protected:
      private:
        struct Configuration : public sackmesser::Configuration
        {
            bool load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server);

            double weight;

            double radius;
        } config_;

        sackmesser::Interface::Ptr interface_;
    };


    class GoalDistanceCost : public Cost
    {
      public:
        GoalDistanceCost(const sackmesser::Interface::Ptr &interface, const std::string &name);

        virtual ~GoalDistanceCost() = default;

        double computeCost(const std::shared_ptr<Agent> &agent, const gafro::Motor<double> &target) const;

        std::string getName() const override { return "goal_distance_cost"; };

        double getWeight() const;

      protected:
      private:
        struct Configuration : public sackmesser::Configuration
        {
            bool load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server);

            double weight;

            double radius;
        } config_;
    };


    class PathLengthCost : public Cost
    {
      public:
        PathLengthCost(const sackmesser::Interface::Ptr &interface, const std::string &name);

        virtual ~PathLengthCost() = default;

        double computeCost(const std::shared_ptr<Agent> &agent, const gafro::Motor<double> &target) const;

        std::string getName() const override { return "path_length_cost"; }

        double getWeight() const;

      protected:
      private:
        struct Configuration : public sackmesser::Configuration
        {
            bool load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server);

            double weight;
        } config_;
    };


    class TrajectorySmoothnessCost : public Cost
    {
      public:
        TrajectorySmoothnessCost(const sackmesser::Interface::Ptr &interface, const std::string &name);

        virtual ~TrajectorySmoothnessCost() = default;

        double computeCost(const std::shared_ptr<Agent> &agent, const gafro::Motor<double> &target) const;

        std::string getName() const override { return "trajectory_smoothness_cost"; };

        double getWeight() const;

      protected:
      private:
        struct Configuration : public sackmesser::Configuration
        {
            bool load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server);

            double weight;
        } config_;
    };

}  // namespace ga_circular_fields_planner