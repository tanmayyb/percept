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

#include <sackmesser/Configurations.hpp>
#include <sackmesser/Logger.hpp>

namespace sackmesser
{
    Configurations::Configurations(const Logger::Ptr &logger) : logger_(logger) {}

    Configurations::~Configurations() {}

    Logger::Ptr Configurations::log()
    {
        return logger_;
    }

    bool Configurations::loadParameter(const std::string &name, bool *param, bool is_dynamic)
    {
        if (!load(name, param))
        {
            return false;
        }

        addParameter(name, param, is_dynamic, false, true);

        if (!checkParameter(name, *param))
        {
            return false;
        }

        return true;
    }

    bool Configurations::loadParameter(const std::string &name, int *param, bool is_dynamic, const int &min_val, const int &max_val)
    {
        if (!load(name, param))
        {
            return false;
        }

        addParameter(name, param, is_dynamic, min_val, max_val);

        if (!checkParameter(name, *param))
        {
            return false;
        }

        return true;
    }

    bool Configurations::loadParameter(const std::string &name, unsigned *param, bool /*is_dynamic*/, const unsigned & /*max_val*/)
    {
        // addParameter(name, param, is_dynamic, static_cast<unsigned>(0), max_val);

        if (!load(name, param))
        {
            return false;
        }

        // if (!checkParameter(name, *param))
        // {
        //     return false;
        // }

        return true;
    }

    bool Configurations::loadParameter(const std::string &name, double *param, bool is_dynamic, const double &min_val, const double &max_val)
    {
        if (!load(name, param))
        {
            return false;
        }

        addParameter(name, param, is_dynamic, min_val, max_val);

        if (!checkParameter(name, *param))
        {
            return false;
        }

        return true;
    }

    bool Configurations::loadParameter(const std::string &name, std::string *param)
    {
        return load(name, param);
    }

    bool Configurations::loadParameter(const std::string &name, std::map<std::string, double> *param)
    {
        return load(name, param);
    }

    bool Configurations::loadParameter(const std::string &name, std::vector<std::string> *param)
    {
        return load(name, param);
    }

    bool Configurations::loadParameter(const std::string &name, std::vector<double> *param)
    {
        return load(name, param);
    }

    bool Configurations::loadParameter(const std::string &name, std::vector<int> *param)
    {
        return load(name, param);
    }

    std::map<std::string, std::any> *Configurations::getDynamicParameters()
    {
        return &parameters_dynamic_;
    }

    const std::map<std::string, std::any> &Configurations::getParametersMinimum()
    {
        return parameters_min_val_;
    }

    const std::map<std::string, std::any> &Configurations::getParametersMaximum()
    {
        return parameters_max_val_;
    }

    const std::map<std::string, std::any> &Configurations::getParametersDefault()
    {
        return parameters_default_;
    }

}  // namespace sackmesser