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
#include <sackmesser/Callbacks.hpp>
#include <sackmesser/Logger.hpp>
#include <sackmesser_runtime/ConfigurationsYAML.hpp>
#include <sackmesser_runtime/Interface.hpp>
#include <thread>

namespace sackmesser::runtime
{
    Interface::Interface(const std::string &config_file, const Logger::Ptr &logger)
      : sackmesser::Interface(std::make_shared<ConfigurationsYAML>(std::filesystem::path(config_file), logger), std::make_shared<Callbacks>(logger),
                              logger)
    {}

    Interface::~Interface() {}

    Interface::Ptr Interface::create(int argc, char **argv, const Logger::Ptr &logger)
    {
        if (argc != 2)
        {
            throw std::runtime_error("Interface: usage <config_file>");
        }

        return std::make_shared<Interface>(argv[1], logger);
    }

    void Interface::loop(const std::function<void()> &callback)
    {
        while (ok())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000.0 / 100.0)));
        }
    }

}  // namespace sackmesser::runtime