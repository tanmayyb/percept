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

#include <sackmesser_ros2/Configurations.hpp>

namespace sackmesser_ros
{
    Configurations::Configurations(rclcpp::Node *node, const std::shared_ptr<sackmesser::Logger> &logger)
      : sackmesser::Configurations(logger), node_(node)
    {}

    Configurations::~Configurations() {}

    bool Configurations::load(const std::string &name, bool *param)
    {
        return node_->get_parameter(name, *param);
    }

    bool Configurations::load(const std::string &name, double *param)
    {
        return node_->get_parameter(name, *param);
    }

    bool Configurations::load(const std::string &name, int *param)
    {
        return node_->get_parameter(name, *param);
    }

    bool Configurations::load(const std::string &name, unsigned *param)
    {
        return node_->get_parameter(name, *param);
    }

    bool Configurations::load(const std::string &name, std::string *param)
    {
        return node_->get_parameter(name, *param);
    }

    bool Configurations::load(const std::string & /*name*/, std::map<std::string, double> * /*param*/)
    {
        return false;
    }

    bool Configurations::load(const std::string &name, std::vector<std::string> *param)
    {
        if (!node_->has_parameter(name))
        {
            return false;
        }

        *param = node_->get_parameter(name).as_string_array();

        return true;
    }

    bool Configurations::load(const std::string &name, std::vector<double> *param)
    {
        if (!node_->has_parameter(name))
        {
            return false;
        }

        *param = node_->get_parameter(name).as_double_array();

        return true;
    }

    bool Configurations::load(const std::string &name, std::vector<int> *param)
    {
        if (!node_->has_parameter(name))
        {
            return false;
        }

        std::vector<long int> values = node_->get_parameter(name).as_integer_array();

        for (const long int &value : values)
        {
            param->push_back(static_cast<int>(value));
        }

        return true;
    }

}  // namespace sackmesser_ros