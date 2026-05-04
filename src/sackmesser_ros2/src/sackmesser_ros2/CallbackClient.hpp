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

#include <sackmesser_ros2/Interface.hpp>
//
#include <sackmesser/FactoryClass.hpp>

namespace sackmesser_ros
{

    namespace base
    {
        class CallbackClient
          : public sackmesser::FactoryClass<CallbackClient, Interface *, const std::string &, const std::string &, const std::string &>
        {
          public:
            CallbackClient() = default;

            virtual ~CallbackClient() = default;
        };
    }  // namespace base

    template <class Request, class Response, class Message>
    class CallbackClient : public base::CallbackClient
    {
      public:
        CallbackClient(Interface *interface, const std::string &ns, const std::string &callback_request, const std::string &callback_response);

        virtual ~CallbackClient() = default;

        struct Configuration : public sackmesser::Configuration
        {
            bool load(const std::string &ns, const sackmesser::Configurations::Ptr &configurations);

            int timeout;
        };

      protected:
        virtual typename Message::Request::SharedPtr encodeRequest(const Request &request) const = 0;

        virtual Response decodeResponse(const typename Message::Response::SharedPtr response_msg) const = 0;

        Interface *getInterface();

      private:
        bool callback(Response &response, const Request &request);

        Interface *interface_;

        std::string callback_response_;

        typename rclcpp::Client<Message>::SharedPtr client_;

        Configuration config_;
    };

}  // namespace sackmesser_ros

#include <sackmesser_ros2/CallbackClient.hxx>
