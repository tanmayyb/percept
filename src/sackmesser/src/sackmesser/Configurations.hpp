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

#include <algorithm>
#include <any>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>
//
#include <sackmesser/Configuration.hpp>
#include <sackmesser/Logger.hpp>

namespace sackmesser
{

    /**
     * contains all configuration parameters related and can be accessed via Interface::getConfigurations
     * */
    class Configurations : public std::enable_shared_from_this<Configurations>
    {
      public:
        /**
         * @brief constructor
         * @details
         *
         * @param logger Logger
         */
        Configurations(const Logger::Ptr &logger);

        /**
         * @brief destructor
         */
        virtual ~Configurations();

        /**
         * @brief load a Configuration
         * @details this method will throw if not all parameters are found!
         *
         * @param ns namespace of all contained parameters
         * @tparam C class derived from Configuration
         * @return Configuration with loaded parameters
         */
        template <class C>
        C load(const std::string &ns);

        /**
         * @brief creates and load a Configuration
         * @details this method will throw if not all parameters are found!
         *
         * @param ns namespace of all contained parameters
         * @tparam C class derived from Configuration
         * @return Configuration with loaded parameters
         */
        template <class C>
        std::unique_ptr<C> createAndLoad(const std::string &ns);

        /**
         * @brief loads a bool parameter
         * @param name name of the parameter
         * @param param bool parameter
         * @param is_dynamic true if the parameter is reconfigurable
         * @return true if the paramter was found
         */
        bool loadParameter(const std::string &name, bool *param, bool is_dynamic);

        /**
         * @brief loads a int parameter
         * @param name name of the parameter
         * @param param int parameter
         * @param is_dynamic true if the parameter is reconfigurable
         * @param min_val lower limit of the parameter
         * @param max_val upper limit of the parameter
         * @return true if the paramter was found
         */
        bool loadParameter(const std::string &name, int *param, bool is_dynamic, const int &min_val = std::numeric_limits<int>::min(),
                           const int &max_val = std::numeric_limits<int>::max());

        /**
         * @brief loads an unsigned int parameter
         * @param name name of the parameter
         * @param param int parameter
         * @param is_dynamic true if the parameter is reconfigurable
         * @param max_val upper limit of the parameter
         * @return true if the paramter was found
         */
        bool loadParameter(const std::string &name, unsigned *param, bool is_dynamic, const unsigned &max_val = std::numeric_limits<unsigned>::max());

        /**
         * @brief loads a double parameter
         * @param name name of the parameter
         * @param param double parameter
         * @param is_dynamic true if the parameter is reconfigurable
         * @param min_val lower limit of the parameter
         * @param max_val upper limit of the parameter
         * @return true if the paramter was found
         */
        bool loadParameter(const std::string &name, double *param, bool is_dynamic, const double &min_val = -std::numeric_limits<double>::max(),
                           const double &max_val = std::numeric_limits<double>::max());

        /**
         * @brief loads a string parameter
         * @param name name of the parameter
         * @param param string parameter
         * @return true if the paramter was found
         */
        bool loadParameter(const std::string &name, std::string *param);

        /**
         * @brief loads a string-double map parameter
         * @param name name of the parameter
         * @param param string parameter
         * @return true if the paramter was found
         */
        bool loadParameter(const std::string &name, std::map<std::string, double> *param);

        /**
         * @brief loads a double vector parameter
         * @param name name of the parameter
         * @param param string parameter
         * @return true if the paramter was found
         */
        bool loadParameter(const std::string &name, std::vector<double> *param);

        /**
         * @brief loads a int vector parameter
         * @param name name of the parameter
         * @param param string parameter
         * @return true if the paramter was found
         */
        bool loadParameter(const std::string &name, std::vector<int> *param);

        /**
         * @brief loads a string vector parameter
         * @param name name of the parameter
         * @param param string parameter
         * @return true if the paramter was found
         */
        bool loadParameter(const std::string &name, std::vector<std::string> *param);

        /**
         * @brief loads a double array parameter
         * @param name name of the parameter
         * @param param string parameter
         * @return true if the paramter was found
         */
        template <int N>
        bool loadParameter(const std::string &name, std::array<double, N> *param);

        /**
         * @brief reconfigure the parameter with the given name
         * @param name name of the parameter
         * @param new_value new value of the paramter
         * @return true if reconfigured successfully
         */
        template <typename Type>
        bool reconfigure(const std::string &name, const Type &new_value);

      protected:
        /**
         * @brief get pointer to all dynamic parameters
         * @return pointer to all dynamic parameters
         */
        std::map<std::string, std::any> *getDynamicParameters();

        /**
         * @brief get reference to all minimum parameters
         * @return reference to all minimum parameters
         */
        const std::map<std::string, std::any> &getParametersMinimum();

        /**
         * @brief get reference to all maximum parameters
         * @return reference to all maximum parameters
         */
        const std::map<std::string, std::any> &getParametersMaximum();

        /**
         * @brief get reference to all default parameters
         * @return reference to all default parameters
         */
        const std::map<std::string, std::any> &getParametersDefault();

        /**
         * @brief access the Logger
         * @details
         * @return Logger
         */
        Logger::Ptr log();

      private:
        /**
         * @brief placeholder function for loading bool variables
         * @param n name of the parameter (including namespace)
         * @param p pointer to the parameter
         * @return true if loaded successfully
         */
        virtual bool load(const std::string &n, bool *p) = 0;

        /**
         * @brief placeholder function for loading double variables
         * @param n name of the parameter (including namespace)
         * @param p pointer to the parameter
         * @return true if loaded successfully
         */
        virtual bool load(const std::string &n, double *p) = 0;

        /**
         * @brief placeholder function for loading int variables
         * @param n name of the parameter (including namespace)
         * @param p pointer to the parameter
         * @return true if loaded successfully
         */
        virtual bool load(const std::string &n, int *p) = 0;

        /**
         * @brief placeholder function for loading string variables
         * @param n name of the parameter (including namespace)
         * @param p pointer to the parameter
         * @return true if loaded successfully
         */
        virtual bool load(const std::string &n, unsigned *p) = 0;

        /**
         * @brief placeholder function for loading std::string variables
         * @param n name of the parameter (including namespace)
         * @param p pointer to the parameter
         * @return true if loaded successfully
         */
        virtual bool load(const std::string &n, std::string *p) = 0;

        /**
         * @brief placeholder function for loading std::map<std::string, double> variables
         * @param n name of the parameter (including namespace)
         * @param p pointer to the parameter
         * @return true if loaded successfully
         */
        virtual bool load(const std::string &n, std::map<std::string, double> *p) = 0;

        /**
         * @brief placeholder function for loading std::vector<std::string> variables
         * @param n name of the parameter (including namespace)
         * @param p pointer to the parameter
         * @return true if loaded successfully
         */
        virtual bool load(const std::string &n, std::vector<std::string> *p) = 0;

        /**
         * @brief placeholder function for loading std::vector<std::string> variables
         * @param n name of the parameter (including namespace)
         * @param p pointer to the parameter
         * @return true if loaded successfully
         */
        virtual bool load(const std::string &n, std::vector<double> *p) = 0;

        /**
         * @brief placeholder function for loading int vector variables
         * @param n name of the parameter (including namespace)
         * @param p pointer to the parameter
         * @return true if loaded successfully
         */
        virtual bool load(const std::string &n, std::vector<int> *p) = 0;

        /**
         * @brief check if the given parameter lies within the specified limits
         * @details limits are generally hardcoded
         * @param name name of the parameter (including namespace)
         * @param param oarameter reference
         * @return true if the parameter lies within the limits
         */
        template <typename Type>
        bool checkParameter(const std::string &name, const Type &param);

        /**
         * @brief add parameter to the list including its limits
         * @details the parameter is stored as a pointer to the correct configuration instance
         *
         * @param name name of the parameter (including namespace)
         * @param param pointer to the parameter
         * @param is_dynamic set to true if the parameter can be modified after loading
         * @param min_val minimum value of the parameter
         * @param max_val maximum value of the parameter
         */
        template <typename Type>
        void addParameter(const std::string &name, Type *param, bool is_dynamic, const Type &min_val, const Type &max_val);

        /// stores all default parameters
        std::map<std::string, std::any> parameters_default_;
        /// stores all minimum values of the parameters
        std::map<std::string, std::any> parameters_min_val_;
        /// stores all maximum values of the parameters
        std::map<std::string, std::any> parameters_max_val_;
        /// stores all dynamic parameters
        std::map<std::string, std::any> parameters_dynamic_;

        /// Logger
        Logger::Ptr logger_;

      public:
        /// typedef
        using Ptr = std::shared_ptr<Configurations>;
    };

}  // namespace sackmesser

#include <sackmesser/Configurations.hxx>