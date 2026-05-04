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
#include <sackmesser_ros2/Publisher.hpp>

namespace sackmesser_ros
{

    template <class MsgType, class... Arguments>
    Publisher<MsgType, Arguments...>::Publisher(Interface *interface, const std::string &ns) : interface_(interface)
    {
        Configuration config = interface->getConfigurations()->load<Configuration>(ns);

        interface->getCallbacks()->addQueue<Arguments...>(config.callback_queue);
        interface->getCallbacks()->addCallbackToQueue<Arguments...>(config.callback_queue,
                                                                    [this](const Arguments &...arguments) { publish(arguments...); });

        rclcpp::PublisherOptions options;
        options.callback_group = interface->getDefaultCallbackGroup();

        publisher_ = interface->create_publisher<MsgType>(config.topic, 10, options);

        interface->log()->info() << "Node: publishing to topic " << config.topic << std::endl;
    }

    template <class MsgType, class... Arguments>
    Publisher<MsgType, Arguments...>::~Publisher() = default;

    template <class MsgType, class... Arguments>
    void Publisher<MsgType, Arguments...>::publish(const Arguments &...arguments) const
    {
        MsgType message = createMessage(arguments...);

        publisher_->publish(message);
    }

    template <class MsgType, class... Arguments>
    const Interface *Publisher<MsgType, Arguments...>::getInterface() const
    {
        return interface_;
    }

    template <class MsgType, class... Arguments>
    bool Publisher<MsgType, Arguments...>::Configuration::load(const std::string &ns, const sackmesser::Configurations::Ptr &configurations)
    {
        return configurations->loadParameter(ns + "topic", &topic)  //
               && configurations->loadParameter(ns + "callback_queue", &callback_queue);
    }

}  // namespace sackmesser_ros