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

#include <sackmesser/Callbacks.hxx>
#include <sackmesser_ros2/CallbackClient.hpp>

namespace sackmesser_ros
{

    template <class Request, class Response, class Message>
    CallbackClient<Request, Response, Message>::CallbackClient(Interface *interface, const std::string &ns, const std::string &callback_request,
                                                               const std::string &callback_response)
      : interface_(interface), callback_response_(callback_response)
    {
        config_ = interface->getConfigurations()->load<Configuration>(ns);

        interface->log()->info() << "providing callback client for " << callback_request << std::endl;

        std::function<bool(Response &, const Request &)> function = [this](Response &response, const Request &request) -> bool {
            return callback(response, request);
        };
        interface->getCallbacks()->addCallback<Response, Request>(callback_request, function);

        client_ = interface->create_client<Message>(callback_request, rclcpp::ServicesQoS(), interface->getClientCallbackGroup());
    }

    template <class Request, class Response, class Message>
    bool CallbackClient<Request, Response, Message>::callback(Response &response, const Request &request)
    {
        auto request_msg = this->encodeRequest(request);
        auto response_msg = client_->async_send_request(request_msg);

        std::future_status status = response_msg.wait_for(std::chrono::milliseconds(config_.timeout));

        if (status == std::future_status::ready)
        {
            response = this->decodeResponse(response_msg.get());

            return true;
        }

        RCLCPP_ERROR_ONCE(rclcpp::get_logger("rclcpp"), "Failed to call service");

        return false;
    }

    template <class Request, class Response, class Message>
    Interface *CallbackClient<Request, Response, Message>::getInterface()
    {
        return interface_;
    }

    template <class Request, class Response, class Message>
    bool CallbackClient<Request, Response, Message>::Configuration::load(const std::string &ns, const sackmesser::Configurations::Ptr &configurations)
    {
        return configurations->loadParameter(ns + "/timeout", &timeout, false);
    }

}  // namespace sackmesser_ros