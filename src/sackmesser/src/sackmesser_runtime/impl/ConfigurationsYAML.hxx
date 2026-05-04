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

#include <sackmesser/UtilityFunctions.hpp>
//
#include <sackmesser_runtime/impl/ConfigurationsYAML.hpp>

namespace sackmesser::runtime
{

    template <class Type>
    bool ConfigurationsYAML::Impl::find(const std::string &name, Type *param) const
    {
        std::vector<std::string> node_names = splitString(name, '/');

        YAML::Node node = YAML::Clone(file_);

        for (const std::string &s : node_names)
        {
            if (!node[s])
            {
                logger_->fatal() << "ConfigurationServer: no parameter named " << name << std::endl;

                return false;
            }

            node = node[s];
        }

        *param = node.as<Type>();

        logger_->info() << "ConfigurationServer: loaded " << name << std::endl;

        return true;
    }

}  // namespace sackmesser::runtime