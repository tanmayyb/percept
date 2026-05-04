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

#include <yaml-cpp/yaml.h>

#include <sackmesser_runtime/ConfigurationsYAML.hpp>

namespace sackmesser::runtime
{
    /**
     * @brief implementation of the yaml configurations loader
     * @details
     */
    class ConfigurationsYAML::Impl
    {
      public:
        /**
         * @brief constructor
         * @details [long description]
         *
         * @param file path of the configuration yaml file
         * @param logger Logger
         */
        Impl(const std::string &file, const Logger::Ptr &logger);

        /**
         * @brief destructor
         * @details
         */
        ~Impl();

        /**
         * @brief load a bool parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &name, bool *parameter);

        /**
         * @brief load a double parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &name, double *parameter);

        /**
         * @brief load a int parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &name, int *parameter);

        /**
         * @brief load a unsigned parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &name, unsigned *parameter);

        /**
         * @brief load a std::string parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &name, std::string *parameter);

        /**
         * @brief load a std::map<std::string, double> parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &name, std::map<std::string, double> *parameter);

        /**
         * @brief load a std::vector<std::string> parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &name, std::vector<std::string> *parameter);

        /**
         * @brief load a std::vector<double> parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &name, std::vector<double> *parameter);

        /**
         * @brief load a std::vector<int> parameter
         * @details
         *
         * @param name name of the parameter
         * @param parameter pointer to the parameter
         *
         * @return true if the parameter was found
         */
        bool load(const std::string &name, std::vector<int> *parameter);

      private:
        /**
         * @brief find a parameter in the configuration file
         * @details
         *
         * @param name name of the parameter
         * @param param pointer to the parameter
         *
         * @return true if the parameter was found
         */
        template <class Type>
        bool find(const std::string &name, Type *param) const;

      private:
        /// configuration yaml file
        YAML::Node file_;

        /// Logger
        Logger::Ptr logger_;
    };

}  // namespace sackmesser::runtime

#include <sackmesser_runtime/impl/ConfigurationsYAML.hxx>