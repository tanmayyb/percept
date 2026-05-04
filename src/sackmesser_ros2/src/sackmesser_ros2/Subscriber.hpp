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
        class Subscriber : public sackmesser::FactoryClass<Subscriber, Interface *, const std::string &, const std::string &>
        {
          public:
            Subscriber();

            virtual ~Subscriber();
        };
    }  // namespace base

    template <class Message, class Type>
    class Subscriber : public base::Subscriber
    {
      public:
        Subscriber(Interface *interface, const std::string &topic, const std::string &callback_queue);

      protected:
        virtual Type convert(const typename Message::SharedPtr message) const = 0;

        Interface *getInterface();

      private:
        void callback(const typename Message::SharedPtr message);

        Interface *interface_;

        std::string callback_queue_;

        typename rclcpp::Subscription<Message>::SharedPtr subscription_;
    };
}  // namespace sackmesser_ros

#include <sackmesser_ros2/Subscriber.hxx>