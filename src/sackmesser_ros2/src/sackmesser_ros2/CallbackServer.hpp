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
        class CallbackServer : public sackmesser::FactoryClass<CallbackServer, Interface *, const std::string &, const std::string &>
        {
          public:
            CallbackServer() = default;

            virtual ~CallbackServer() = default;
        };
    }  // namespace base

    template <class Request, class Response, class Message>
    class CallbackServer : public base::CallbackServer
    {
      public:
        CallbackServer(Interface *interface, const std::string &ns, const std::string &callback);

        virtual ~CallbackServer() = default;

        struct Configuration : public sackmesser::Configuration
        {
            bool load(const std::string &ns, const sackmesser::Configurations::Ptr &configurations);
        };

      protected:
        Interface *getInterface();

      private:
        void invokeCallback(const typename Message::Request::SharedPtr request_msg, typename Message::Response::SharedPtr response_msg);

        virtual Response callback(const Request &request) = 0;

        virtual void encodeResponse(const Response &response, typename Message::Response::SharedPtr response_msg) const = 0;

        virtual Request decodeRequest(const typename Message::Request::SharedPtr request_msg) const = 0;

      private:
        Interface *interface_;

        typename rclcpp::Service<Message>::SharedPtr server_;

        Configuration config_;
    };

}  // namespace sackmesser_ros

#include <sackmesser_ros2/CallbackServer.hxx>
