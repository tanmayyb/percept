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
#include <sackmesser_ros2/Subscriber.hpp>

namespace sackmesser_ros
{

    template <class Message, class Type>
    Subscriber<Message, Type>::Subscriber(Interface *interface, const std::string &topic, const std::string &callback_queue)
      : interface_(interface), callback_queue_(callback_queue)
    {
        interface->log()->info() << "subscribing to topic " << topic << std::endl;

        interface->getCallbacks()->addQueue<Type>(callback_queue_);

        rclcpp::SubscriptionOptions options;
        options.callback_group = interface->getDefaultCallbackGroup();

        subscription_ = interface->create_subscription<Message>(topic, 10, [this](const typename Message::SharedPtr msg) { callback(msg); }, options);
    }

    template <class Message, class Type>
    void Subscriber<Message, Type>::callback(const typename Message::SharedPtr message)
    {
        interface_->getCallbacks()->invoke(callback_queue_, this->convert(message));
    }

    template <class Message, class Type>
    Interface *Subscriber<Message, Type>::getInterface()
    {
        return interface_;
    }

}  // namespace sackmesser_ros