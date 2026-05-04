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

#include <sackmesser/Configurations.hpp>
//
#include <rclcpp/rclcpp.hpp>

namespace sackmesser_ros
{
    class Configurations : public sackmesser::Configurations
    {
      public:
        Configurations(rclcpp::Node *node, const std::shared_ptr<sackmesser::Logger> &logger);

        virtual ~Configurations();

      protected:
        bool load(const std::string &, bool *);

        bool load(const std::string &, double *);

        bool load(const std::string &, int *);

        bool load(const std::string &, unsigned *);

        bool load(const std::string &, std::string *);

        bool load(const std::string &, std::map<std::string, double> *);

        bool load(const std::string &, std::vector<std::string> *);

        bool load(const std::string &, std::vector<double> *);

        bool load(const std::string &, std::vector<int> *);

      private:
        rclcpp::Node *node_;
    };

}  // namespace sackmesser_ros