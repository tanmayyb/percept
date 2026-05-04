/*
 * Copyright (C) Tobias Löw (tobi.loew@protonmail.ch)
 *
 * This file is part of sackmesser
 *
 * sackmesser is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * sackmesser is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with sackmesser.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <sackmesser/Callbacks.hpp>
#include <sackmesser/Configuration.hpp>
#include <sackmesser/Interface.hpp>
#include <sackmesser/Logger.hpp>
//
#include <rclcpp/rclcpp.hpp>

namespace sackmesser_ros
{
    namespace base
    {
        class Publisher;

        class Subscriber;

        class CallbackClient;

        class CallbackServer;
    }  // namespace base

    class Interface : public sackmesser::Interface, public rclcpp::Node
    {
      public:
        Interface(const std::string &name, const std::string &package,
                  const std::shared_ptr<sackmesser::Logger> &logger = std::make_shared<sackmesser::Logger>());

        virtual ~Interface();

        bool ok() const;

        void loop(const std::function<void()> &callback);

        rclcpp::Node::SharedPtr getNode();

        rclcpp::executors::MultiThreadedExecutor &getExecutor();

        rclcpp::CallbackGroup::SharedPtr getDefaultCallbackGroup();

        rclcpp::CallbackGroup::SharedPtr getClientCallbackGroup();

      protected:
      private:
        struct Configuration : public sackmesser::Configuration
        {
            bool load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server);

            std::vector<std::string> publishers;

            std::vector<std::string> subscribers;

            std::vector<std::string> callback_clients;

            std::vector<std::string> callback_servers;

            double loop_frequency;
        };

        Configuration config_;

        std::vector<std::shared_ptr<base::Publisher>> publishers_;

        std::vector<std::shared_ptr<base::Subscriber>> subscribers_;

        std::vector<std::shared_ptr<base::CallbackClient>> callback_clients_;

        std::vector<std::shared_ptr<base::CallbackServer>> callback_servers_;

        rclcpp::executors::MultiThreadedExecutor executor_;

        rclcpp::CallbackGroup::SharedPtr default_callback_group_;

        rclcpp::CallbackGroup::SharedPtr client_callback_group_;

      public:
        using Ptr = std::shared_ptr<Interface>;

        static sackmesser_ros::Interface::Ptr create(int argc, char **argv, const std::string &name, const std::string &package,
                                                     const std::shared_ptr<sackmesser::Logger> &logger = std::make_shared<sackmesser::Logger>());
    };

}  // namespace sackmesser_ros