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

namespace sackmesser
{

    template <class C>
    C Configurations::load(const std::string &ns)
    {
        C config;

        if (!config.load(ns, shared_from_this()))
        {
            logger_->fatal() << "ConfigurationServer: failed to load " << ns << std::endl;
        }

        return config;
    }

    template <class C>
    std::unique_ptr<C> Configurations::createAndLoad(const std::string &ns)
    {
        std::unique_ptr<C> config = std::make_unique<C>();

        if (!config->load(ns, shared_from_this()))
        {
            logger_->fatal() << "ConfigurationServer: failed to load " << ns << std::endl;
        }

        return std::move(config);
    }

    template <int N>
    bool Configurations::loadParameter(const std::string &name, std::array<double, N> *param)
    {
        std::vector<double> data;

        if (!loadParameter(name, &data))
        {
            return false;
        }

        if (data.size() != N)
        {
            logger_->info() << "ConfigurationServer: number of values does not match size " << N << " for parameter " << name << std::endl;

            return false;
        }

        std::copy(data.begin(), data.end(), param->begin());

        return true;
    }

    template <typename Type>
    bool Configurations::reconfigure(const std::string &name, const Type &new_value)
    {
        if (parameters_dynamic_.find(name) == parameters_dynamic_.end())
        {
            logger_->error() << "ConfigurationServer: unknown parameter " << name << std::endl;

            return false;
        }

        if (!checkParameter(name, new_value))
        {
            return false;
        }

        *std::any_cast<Type *>(parameters_dynamic_.at(name)) = new_value;

        logger_->info() << "ConfigurationServer: reconfigured " << name << " -> " << new_value << std::endl;

        return true;
    }

    template <typename Type>
    bool Configurations::checkParameter(const std::string &name, const Type &param)
    {
        if (param < std::any_cast<Type>(parameters_min_val_.at(name)))
        {
            logger_->error() << "ConfigurationServer: " << name << " value (" << param << ") smaller than minimum ("
                             << std::any_cast<Type>(parameters_min_val_.at(name)) << ")" << std::endl;

            return false;
        }

        if (param > std::any_cast<Type>(parameters_max_val_.at(name)))
        {
            logger_->error() << "ConfigurationServer: " << name << " value (" << param << ") larger than maximum ("
                             << std::any_cast<Type>(parameters_max_val_.at(name)) << ")" << std::endl;

            return false;
        }

        return true;
    }
    template <typename Type>
    void Configurations::addParameter(const std::string &name, Type *param, bool is_dynamic, const Type &min_val, const Type &max_val)
    {
        parameters_default_.emplace(std::make_pair(name, *param));
        parameters_min_val_.emplace(std::make_pair(name, min_val));
        parameters_max_val_.emplace(std::make_pair(name, max_val));

        if (is_dynamic)
        {
            parameters_dynamic_.emplace(std::make_pair(name, param));
        }
    }

}  // namespace sackmesser