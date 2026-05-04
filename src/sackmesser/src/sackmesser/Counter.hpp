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

namespace sackmesser
{

    /**
     * @brief utility class for counting
     * @details
     *
     * @tparam T type of the counter
     */
    template <typename T>
    class Counter
    {
      public:
        /**
         * @brief constructor
         * @details
         *
         * @param max maximum of the counter variable
         */
        Counter(const T &max);

        /**
         * @brief constructor
         * @details
         *
         * @param min minimum of the counter variable
         * @param max maximum of the counter variable
         */
        Counter(const T &min, const T &max);

        /**
         * @brief destructor
         * @details
         */
        virtual ~Counter();

        /**
         * @brief increase the counter
         * @details
         */
        T &operator++();

      protected:
      private:
        /// current counter value
        T value_;

        /// minimum counter value, value_ will be reset to this value
        T min_;

        /// maximum counter value, value_ will be reset at this value
        T max_;
    };
}  // namespace sackmesser

#include <sackmesser/Counter.hxx>