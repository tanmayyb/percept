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
#include <sackmesser_ros2/CallbackServer.hpp>

namespace sackmesser_ros
{

    template <class Request, class Response, class Message>
    CallbackServer<Request, Response, Message>::CallbackServer(Interface *interface, const std::string &ns, const std::string &callback)
      : interface_(interface)
    {
        config_ = interface->getConfigurations()->load<Configuration>(ns);

        interface->log()->info() << "providing callback server for " << callback << std::endl;

        server_ = interface->create_service<Message>(callback,
                                                     std::bind(&CallbackServer::invokeCallback, this, std::placeholders::_1, std::placeholders::_2));
    }

    template <class Request, class Response, class Message>
    void CallbackServer<Request, Response, Message>::invokeCallback(const typename Message::Request::SharedPtr request_msg,
                                                                    typename Message::Response::SharedPtr response_msg)
    {
        Request request = decodeRequest(request_msg);

        Response response = callback(request);

        encodeResponse(response, response_msg);
    }

    template <class Request, class Response, class Message>
    Interface *CallbackServer<Request, Response, Message>::getInterface()
    {
        return interface_;
    }

    template <class Request, class Response, class Message>
    bool CallbackServer<Request, Response, Message>::Configuration::load(const std::string & /*ns*/,
                                                                         const sackmesser::Configurations::Ptr & /*configurations*/)
    {
        return true;
    }

}  // namespace sackmesser_ros