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

namespace sackmesser::runtime
{
    /**
     * @brief yaml configurations loader
     * @details
     */
    class ConfigurationsYAML : public Configurations
    {
      public:
        /**
         * @brief constructor
         * @details [long description]
         *
         * @param file path of the configuration yaml file
         * @param logger Logger
         */
        ConfigurationsYAML(const std::string &file, const Logger::Ptr &logger);

        /**
         * @brief destructor
         * @details
         */
        virtual ~ConfigurationsYAML();

      protected:
      private:
        /**
         * @brief load a bool parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &n, bool *p);

        /**
         * @brief load a double parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &n, double *p);

        /**
         * @brief load a int parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &n, int *p);

        /**
         * @brief load a unsigned parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &n, unsigned *p);

        /**
         * @brief load a std::string parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &n, std::string *p);

        /**
         * @brief load a std::map<std::string, double> parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &n, std::map<std::string, double> *p);

        /**
         * @brief load a std::vector<std::string> parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &n, std::vector<std::string> *p);

        /**
         * @brief load a std::vector<double> parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &n, std::vector<double> *p);

        /**
         * @brief load a std::vector<int> parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &n, std::vector<int> *p);

      public:
        /// typedef
        using Ptr = std::shared_ptr<ConfigurationsYAML>;

      private:
        /// actual implementation of the yaml configuration loader
        class Impl;

        /// instance of the actual implementation of the yaml configuration loader
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sackmesser::runtime