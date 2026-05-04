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

#include <sackmesser/UtilityFunctions.hpp>

namespace sackmesser
{

    std::vector<std::string> splitString(const std::string &string, const char &delimiter)
    {
        std::vector<std::string> split;

        std::size_t p1 = 0, p2 = 0;
        while (p2 != std::string::npos)
        {
            p2 = string.find(delimiter, p1);
            split.push_back(string.substr(static_cast<unsigned>(p1), ((p2 != std::string::npos) ? p2 : string.size()) - static_cast<unsigned>(p1)));
            p1 = p2 + 1;
        }

        return split;
    }

}  // namespace sackmesser