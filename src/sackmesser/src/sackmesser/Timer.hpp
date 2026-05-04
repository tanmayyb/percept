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

#include <chrono>
#include <functional>

namespace sackmesser
{

    /**
     * @brief utility class for measuring execution time
     */
    class Timer
    {
      public:
        /**
         * @brief default constructor
         */
        Timer();

        /**
         * @brief default destructor
         */
        virtual ~Timer();

        /// typedef
        using TimePoint = std::chrono::high_resolution_clock::time_point;

        /**
         * @brief starts the timer
         * @details only one timer can be active, calling this method twice has no effect
         */
        void start();

        /**
         * @brief stops the timer
         * @details resets internal flag, such that a new timer can be started
         * @return time that has passed between calling Timer::start and Timer::stop in nanoseconds
         */
        double stop();

        /**
         * @brief get the time that has passed since calling Timer::start
         * @details internal flag is not reset, this method can be called multiple times
         * @return time that has passed since calling Timer::start in nanoseconds
         */
        double getMeasurement();

        /**
         * @brief measures the execution time of the given function
         * @details this function can be used indepently from Timer::start
         *
         * @param function function for which the execution time should be determined
         * @return execution time of the given function in nanoseconds
         */
        static double measureTime(const std::function<void()> &function);

        /**
         * @brief measure the average execution time of the given function
         * @details as opposed to Timer::measureTime this computerAverageTime calls the function multiple time in order to compute an average
         *
         * @param function function for which the execution time should be determined
         * @param executions how many times the function should be called in order to compute the average execution time
         * @return execution time of the given function in nanoseconds
         */
        static double computeAverageTime(const std::function<void()> &function, const int &executions = 10000);

        /**
         * @brief get the current TimePoint from the high_resolution_clock
         * @return current TimePoint
         */
        static TimePoint now();

        /**
         * @brief compute the difference between two TimePoints
         *
         * @param start first TimePoint
         * @param end second TimePoint
         *
         * @return difference in nanoseconds
         */
        static double difference(const TimePoint &start, const TimePoint &end);

      protected:
      private:
        /// flag to keep track if the timer is running
        bool is_running_;

        /// TimePoint when Timer::start has been called
        TimePoint start_;
    };

}  // namespace sackmesser