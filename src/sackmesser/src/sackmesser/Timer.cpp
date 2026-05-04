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

#include <numeric>
#include <sackmesser/Timer.hpp>

namespace sackmesser
{

    Timer::Timer() : is_running_(false) {}

    Timer::~Timer() = default;

    void Timer::start()
    {
        if (!is_running_)
        {
            start_ = now();
            is_running_ = true;
        }
    }

    double Timer::stop()
    {
        if (!is_running_)
        {
            return 0.0;
        }

        is_running_ = false;

        return difference(start_, now());
    }

    double Timer::getMeasurement()
    {
        if (!is_running_)
        {
            return 0.0;
        }

        return difference(start_, now());
    }

    double Timer::measureTime(const std::function<void()> &function)
    {
        TimePoint start = now();
        function();
        TimePoint end = now();

        return difference(start, end);
    }

    double Timer::computeAverageTime(const std::function<void()> &function, const int &executions)
    {
        TimePoint start = now();

        for (int i = 0; i < executions; ++i)
        {
            function();
        }

        TimePoint end = now();

        return difference(start, end) / static_cast<double>(executions);
    }

    Timer::TimePoint Timer::now()
    {
        return std::chrono::high_resolution_clock::now();
    }

    double Timer::difference(const TimePoint &start, const TimePoint &end)
    {
        return 1e-9 * static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

}  // namespace sackmesser