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

#include <sackmesser/Interface.hpp>

namespace sackmesser::runtime
{
    /**
     * @brief runtime interface
     * @details this class can be used if your binary should not depend on any external frameworks
     *
     */
    class Interface : public sackmesser::Interface
    {
      public:
        /**
         * @brief constructor
         * @details
         *
         * @param config_file configuration yaml file that will be loaded by ConfigurationsYAML
         * @param logger Logger
         */
        Interface(const std::string &config_file, const Logger::Ptr &logger = std::make_shared<Logger>());

        /**
         * @brief default destructor
         * @details
         */
        virtual ~Interface();

        void loop(const std::function<void()> &callback);

      protected:
      private:
      public:
        /**
         * @brief create an instance of the runtime Interface
         * @details
         *
         * @param argc argc from main, expected to be 2
         * @param argv argv from main, expected to be the path to the configuration yaml file
         * @param logger custom looger
         * @return Interface
         */
        static Interface::Ptr create(int argc, char **argv, const Logger::Ptr &logger = std::make_shared<Logger>());
    };

}  // namespace sackmesser::runtime