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
#include <sackmesser/Configurations.hpp>
#include <sackmesser/Interface.hpp>
#include <sackmesser/Logger.hpp>

namespace sackmesser
{
    Interface::Interface(const std::shared_ptr<Configurations> &config_server,  //
                         const std::shared_ptr<Callbacks> &callbacks,           //
                         const Logger::Ptr &logger)
      : config_server_(config_server), callbacks_(callbacks), logger_(logger)
    {
        is_running_ = true;
    }

    Interface::~Interface() {}

    std::shared_ptr<Configurations> Interface::getConfigurations()
    {
        return config_server_;
    }

    std::shared_ptr<Callbacks> Interface::getCallbacks()
    {
        return callbacks_;
    }

    Logger::Ptr Interface::log()
    {
        return logger_;
    }

    bool Interface::ok() const
    {
        return is_running_;
    }

    int Interface::shutdown()
    {
        is_running_ = false;

        return 0;
    }

}  // namespace sackmesser