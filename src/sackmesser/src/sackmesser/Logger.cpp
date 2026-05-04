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

#include <iostream>
#include <sackmesser/Logger.hpp>

namespace sackmesser
{

    Logger::Logger()
      : ostream_(std::cout.rdbuf()), prefix_("\033[37mINFO "), level_(Level::INFO), is_enabled_(true), is_verbose_(false), debug_(false)

    {
        start_time_ = Timer::now();
    }

    Logger::~Logger() {}

    void Logger::setLevel(Level level)
    {
        switch (level)
        {
        case Level::INFO: {
            prefix_ = "\033[37mINFO ";
            level_ = Level::INFO;
            break;
        }
        case Level::WARN: {
            prefix_ = "\033[33mWARN ";
            level_ = Level::WARN;
            break;
        }
        case Level::ERROR: {
            prefix_ = "\033[31mERROR ";
            level_ = Level::ERROR;
            break;
        }
        case Level::VERBOSE: {
            prefix_ = "\033[37mINFO ";
            level_ = Level::VERBOSE;
            break;
        }
        }
    }

    void Logger::enable()
    {
        is_enabled_ = true;
    }

    void Logger::disable()
    {
        is_enabled_ = false;
    }

    void Logger::enableDebug()
    {
        debug_ = true;
    }

    void Logger::enableVerbose()
    {
        is_verbose_ = true;
    }

    void Logger::disableDebug()
    {
        debug_ = true;
    }

    void Logger::disableVerbose()
    {
        is_verbose_ = false;
    }

    double Logger::getTime() const
    {
        return Timer::difference(start_time_, Timer::now());
    }

    Logger &Logger::debug()
    {
        setLevel(Level::INFO);

        return *this;
    }

    Logger &Logger::verbose()
    {
        setLevel(Level::VERBOSE);

        return *this;
    }

    Logger &Logger::info()
    {
        setLevel(Level::INFO);

        return *this;
    }

    Logger &Logger::warn()
    {
        setLevel(Level::WARN);

        return *this;
    }

    Logger &Logger::error()
    {
        setLevel(Level::ERROR);

        return *this;
    }

    Logger::Fatal Logger::fatal()
    {
        setLevel(Level::ERROR);

        return Logger::Fatal(this);
    }

    Logger::Fatal::Fatal(Logger *log) : log_(log) {}

    Logger::Fatal::~Fatal() noexcept(false)
    {
        throw std::runtime_error("fatal error");
    }

    Logger &operator<<(Logger &log, ostream_manipulator pf)
    {
        if (!log.is_enabled_ || (!log.is_verbose_ && log.level_ == Logger::Level::VERBOSE))
        {
            return log;
        }

        log.ostream_ << pf;

        return log;
    }

}  // namespace sackmesser