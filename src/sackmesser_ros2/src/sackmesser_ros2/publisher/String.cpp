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

#include <sackmesser_ros2/publisher/String.hpp>

namespace sackmesser_ros
{

    StringPublisher::StringPublisher(Interface *interface, const std::string &ns) : Publisher<std_msgs::msg::String, std::string>(interface, ns) {}

    StringPublisher::~StringPublisher() = default;

    std_msgs::msg::String StringPublisher::createMessage(const std::string &argument) const
    {
        std_msgs::msg::String string_message;

        string_message.data = argument;

        return string_message;
    }

}  // namespace sackmesser_ros

REGISTER_CLASS(sackmesser_ros::base::Publisher, sackmesser_ros::StringPublisher, "ros_string");