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
    class Interface;

    namespace base
    {
        class Publisher : public sackmesser::FactoryClass<Publisher, Interface *, const std::string &>
        {
          public:
            Publisher();

            virtual ~Publisher();
        };
    }  // namespace base

    template <class MsgType, class... Arguments>
    class Publisher : public base::Publisher
    {
      public:
        Publisher(Interface *interface, const std::string &ns);

        virtual ~Publisher();

        void publish(const Arguments &...arguments) const;

        virtual MsgType createMessage(const Arguments &...arguments) const = 0;

        const Interface *getInterface() const;

      protected:
      private:
        struct Configuration : public sackmesser::Configuration
        {
            bool load(const std::string &ns, const sackmesser::Configurations::Ptr &configurations);

            std::string topic;

            std::string callback_queue;
        };

      private:
        typename rclcpp::Publisher<MsgType>::SharedPtr publisher_;

        Interface *interface_;
    };

}  // namespace sackmesser_ros

#include <sackmesser_ros2/Publisher.hxx>