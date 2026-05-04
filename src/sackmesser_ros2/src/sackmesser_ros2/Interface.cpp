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

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <sackmesser/Callbacks.hpp>
#include <sackmesser_ros2/CallbackClient.hpp>
#include <sackmesser_ros2/CallbackServer.hpp>
#include <sackmesser_ros2/Interface.hpp>
#include <sackmesser_ros2/Publisher.hpp>
#include <sackmesser_ros2/Subscriber.hpp>
#include <sackmesser_runtime/ConfigurationsYAML.hpp>

namespace sackmesser_ros
{

    Interface::Interface(const std::string &name, const std::string &package, const std::shared_ptr<sackmesser::Logger> &logger)
      : sackmesser::Interface(std::make_shared<sackmesser::runtime::ConfigurationsYAML>(
                                ament_index_cpp::get_package_share_directory(package) + "/" + name + "/" + name + "_config.yaml", logger),  //
                              std::make_shared<sackmesser::Callbacks>(logger),                                                              //
                              logger),
        rclcpp::Node(name)
    {
        config_ = getConfigurations()->load<Configuration>("");

        default_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        client_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

        for (const std::string &publisher : config_.publishers)
        {
            std::string type, topic, callback_queue;

            if (getConfigurations()->loadParameter("publisher/" + publisher + "/type", &type))
            {
                publishers_.push_back(base::Publisher::getFactory()->createShared(type, this, "publisher/" + publisher + "/"));
            }
            else
            {
                this->log()->warn() << "no publisher " << publisher << " added" << std::endl;
            }
        }

        for (const std::string &subscriber : config_.subscribers)
        {
            std::string type, topic, callback_queue;

            if (getConfigurations()->loadParameter("subscriber/" + subscriber + "/type", &type) &&
                getConfigurations()->loadParameter("subscriber/" + subscriber + "/topic", &topic) &&
                getConfigurations()->loadParameter("subscriber/" + subscriber + "/callback_queue", &callback_queue))
            {
                subscribers_.push_back(base::Subscriber::getFactory()->createShared(type, this, topic, callback_queue));
            }
            else
            {
                this->log()->warn() << "no subscriber " << subscriber << " added" << std::endl;
            }
        }

        for (const std::string &callback_client : config_.callback_clients)
        {
            std::string type, callback_request, callback_response;

            if (getConfigurations()->loadParameter("callback_client/" + callback_client + "/type", &type) &&
                getConfigurations()->loadParameter("callback_client/" + callback_client + "/callback_request", &callback_request) &&
                getConfigurations()->loadParameter("callback_client/" + callback_client + "/callback_response", &callback_response))
            {
                callback_clients_.push_back(base::CallbackClient::getFactory()->createShared(type, this, "callback_client/" + callback_client,
                                                                                             callback_request, callback_response));
            }
            else
            {
                this->log()->warn() << "no callback_client " << callback_client << " added" << std::endl;
            }
        }

        for (const std::string &callback_server : config_.callback_servers)
        {
            std::string type, callback;

            if (getConfigurations()->loadParameter("callback_server/" + callback_server + "/type", &type) &&
                getConfigurations()->loadParameter("callback_server/" + callback_server + "/callback", &callback))
            {
                callback_servers_.push_back(
                  base::CallbackServer::getFactory()->createShared(type, this, "callback_server/" + callback_server, callback));
            }
            else
            {
                this->log()->warn() << "no callback_server " << callback_server << " added" << std::endl;
            }
        }
    }

    Interface::~Interface()
    {
        rclcpp::shutdown();
    }

    bool Interface::ok() const
    {
        return rclcpp::ok();
    }

    void Interface::loop(const std::function<void()> &callback)
    {
        executor_.add_node(getNode());

        while (ok())
        {
            callback();

            executor_.spin_once(std::chrono::nanoseconds(static_cast<int>(100 * config_.loop_frequency)));

            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000.0 / config_.loop_frequency)));
        }
    }

    rclcpp::Node::SharedPtr Interface::getNode()
    {
        return rclcpp::Node::shared_from_this();
    }

    rclcpp::executors::MultiThreadedExecutor &Interface::getExecutor()
    {
        return executor_;
    }

    rclcpp::CallbackGroup::SharedPtr Interface::getDefaultCallbackGroup()
    {
        return default_callback_group_;
    }

    rclcpp::CallbackGroup::SharedPtr Interface::getClientCallbackGroup()
    {
        return client_callback_group_;
    }

    Interface::Ptr Interface::create(int argc, char **argv, const std::string &name, const std::string &package,
                                     const std::shared_ptr<sackmesser::Logger> &logger)
    {
        rclcpp::init(argc, argv);

        // if (argc != 2)
        // {
        //     throw std::runtime_error("Interface: usage <config_file>");
        // }

        return std::make_shared<Interface>(name, package, logger);
    }

    bool Interface::Configuration::load(const std::string & /*ns*/, const std::shared_ptr<sackmesser::Configurations> &server)
    {
        return server->loadParameter("publishers", &publishers) &&              //
               server->loadParameter("subscribers", &subscribers) &&            //
               server->loadParameter("callback_clients", &callback_clients) &&  //
               server->loadParameter("callback_servers", &callback_servers) &&  //
               server->loadParameter("loop_frequency", &loop_frequency, true);
    }

}  // namespace sackmesser_ros