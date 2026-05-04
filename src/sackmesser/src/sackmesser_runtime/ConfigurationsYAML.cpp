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

#include <sackmesser_runtime/ConfigurationsYAML.hpp>
#include <sackmesser_runtime/impl/ConfigurationsYAML.hpp>

namespace sackmesser::runtime
{
    ConfigurationsYAML::ConfigurationsYAML(const std::string &file, const Logger::Ptr &logger) : Configurations(logger)
    {
        impl_ = std::make_unique<Impl>(file, logger);
    }

    ConfigurationsYAML::~ConfigurationsYAML() {}

    bool ConfigurationsYAML::load(const std::string &n, bool *p)
    {
        return impl_->load(n, p);
    }

    bool ConfigurationsYAML::load(const std::string &n, double *p)
    {
        return impl_->load(n, p);
    }

    bool ConfigurationsYAML::load(const std::string &n, int *p)
    {
        return impl_->load(n, p);
    }

    bool ConfigurationsYAML::load(const std::string &n, unsigned *p)
    {
        return impl_->load(n, p);
    }

    bool ConfigurationsYAML::load(const std::string &n, std::string *p)
    {
        return impl_->load(n, p);
    }

    bool ConfigurationsYAML::load(const std::string &n, std::map<std::string, double> *p)
    {
        return impl_->load(n, p);
    }

    bool ConfigurationsYAML::load(const std::string &n, std::vector<std::string> *p)
    {
        return impl_->load(n, p);
    }

    bool ConfigurationsYAML::load(const std::string &n, std::vector<double> *p)
    {
        return impl_->load(n, p);
    }

    bool ConfigurationsYAML::load(const std::string &n, std::vector<int> *p)
    {
        return impl_->load(n, p);
    }

}  // namespace sackmesser::runtime