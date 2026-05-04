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

#include <filesystem>
#include <sackmesser_runtime/impl/ConfigurationsYAML.hpp>

namespace sackmesser::runtime
{
    ConfigurationsYAML::Impl::Impl(const std::string &file, const Logger::Ptr &logger) : logger_(logger)
    {
        if (!std::filesystem::exists(std::filesystem::path(file)))
        {
            logger->fatal() << "file '" << file << "' does not exist" << std::endl;
        }

        file_ = YAML::LoadFile(file);
    }

    ConfigurationsYAML::Impl::~Impl() {}

    bool ConfigurationsYAML::Impl::load(const std::string &n, bool *p)
    {
        return find<bool>(n, p);
    }

    bool ConfigurationsYAML::Impl::load(const std::string &n, double *p)
    {
        return find<double>(n, p);
    }

    bool ConfigurationsYAML::Impl::load(const std::string &n, int *p)
    {
        return find<int>(n, p);
    }

    bool ConfigurationsYAML::Impl::load(const std::string &n, unsigned *p)
    {
        return find<unsigned>(n, p);
    }

    bool ConfigurationsYAML::Impl::load(const std::string &n, std::string *p)
    {
        return find<std::string>(n, p);
    }

    bool ConfigurationsYAML::Impl::load(const std::string &n, std::map<std::string, double> *p)
    {
        return find<std::map<std::string, double>>(n, p);
    }

    bool ConfigurationsYAML::Impl::load(const std::string &n, std::vector<std::string> *p)
    {
        return find<std::vector<std::string>>(n, p);
    }

    bool ConfigurationsYAML::Impl::load(const std::string &n, std::vector<double> *p)
    {
        return find<std::vector<double>>(n, p);
    }

    bool ConfigurationsYAML::Impl::load(const std::string &n, std::vector<int> *p)
    {
        return find<std::vector<int>>(n, p);
    }
}  // namespace sackmesser::runtime