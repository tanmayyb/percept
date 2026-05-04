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

#include <memory>
#include <string>

namespace sackmesser
{
    class Configurations;

    /**
     * @brief base configuration class, all local configuration should inherit from this class
     */
    class Configuration
    {
      public:
        /**
         * @brief default constructor
         * @details
         */
        Configuration() = default;

        /**
         * @brief destructor
         */
        virtual ~Configuration() = default;

        /**
         * @brief load configuration parameters
         * @return true if all parameters were found
         */
        virtual bool load(const std::string &ns, const std::shared_ptr<Configurations> &server) = 0;
    };
}  // namespace sackmesser