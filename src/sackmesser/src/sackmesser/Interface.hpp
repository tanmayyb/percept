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
//
#include <sackmesser/Logger.hpp>

namespace sackmesser
{
    class Configurations;
    class Callbacks;

    /**
     * @brief interface class that should be injected into classes
     * @details This class is meant as an abstraction layer to different frameworks. The idea is that libraries can be built by only having this
     * lightweight library as a dependency. And then in applications the different frameworks can injected interfaces that inherited from this class.
     */
    class Interface
    {
      public:
        /**
         * @brief constructor
         * @param config_server
         * @param callbacks
         * @param logger
         */
        Interface(const std::shared_ptr<Configurations> &config_server,  //
                  const std::shared_ptr<Callbacks> &callbacks,           //
                  const Logger::Ptr &logger);

        /**
         * @brief default constructor
         */
        virtual ~Interface();

        /**
         * @brief access the configurations
         * @return ConfigurationServer
         */
        std::shared_ptr<Configurations> getConfigurations();

        /**
         * @brief access callbacks
         * @return CallbackHandler
         */
        std::shared_ptr<Callbacks> getCallbacks();

        /**
         * @brief access logging
         * @return Logger
         */
        Logger::Ptr log();

        /**
         * @brief status of the application
         * @return false if something is wrong
         */
        virtual bool ok() const;

        virtual void loop(const std::function<void()> &callback) = 0;

        int shutdown();

      private:
        ///
        std::shared_ptr<Configurations> config_server_;

        ///
        std::shared_ptr<Callbacks> callbacks_;

        ///
        Logger::Ptr logger_;

        bool is_running_;

      public:
        /// typedef
        using Ptr = std::shared_ptr<Interface>;
    };

}  // namespace sackmesser