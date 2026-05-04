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

#include <std_msgs/msg/bool.hpp>
//
#include <sackmesser_ros2/Publisher.hpp>

namespace sackmesser_ros
{

    class BoolPublisher : public Publisher<std_msgs::msg::Bool, bool>
    {
      public:
        BoolPublisher(Interface *interface, const std::string &ns);

        virtual ~BoolPublisher();

        std_msgs::msg::Bool createMessage(const bool &argument) const;

      protected:
      private:
    };

}  // namespace sackmesser_ros