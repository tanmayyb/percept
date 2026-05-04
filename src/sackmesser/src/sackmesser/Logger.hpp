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
#include <iostream>
#include <map>
#include <memory>
#include <string>
//
#include <sackmesser/Timer.hpp>

namespace sackmesser
{

    ///
    typedef std::ostream &(*ostream_manipulator)(std::ostream &);

    /**
     * @brief logger class to handle status messages
     */
    class Logger
    {
      public:
        /**
         * @brief all levels of logging
         */
        enum class Level
        {
            INFO,
            WARN,
            ERROR,
            VERBOSE

        };

      public:
        /**
         * @brief default constructor
         */
        Logger();

        /**
         * @brief default destructor
         */
        virtual ~Logger();

        /**
         * @brief enable logging
         */
        void enable();

        /**
         * @brief enable debug messages
         */
        void enableDebug();

        /**
         * @brief enable verbose messages
         */
        void enableVerbose();

        /**
         * @brief disable logging
         */
        void disable();

        /**
         * @brief disable debug messages
         */
        void disableDebug();

        /**
         * @brief disable verbose messages
         */
        void disableVerbose();

        /**
         * @brief get the time that has passed since starting the application
         * @return passed time in nanoseconds
         */
        double getTime() const;

        /**
         * @brief print debug message
         * @details sets the logger level to Level::INFO
         * @return Logger reference to enable operator<<
         */
        Logger &debug();

        /**
         * @brief print verbose message
         * @details sets the logger level to Level::VERBOSE
         * @return Logger reference to enable operator<<
         */
        Logger &verbose();

        /**
         * @brief print message
         * @details sets the logger level to Level::INFO
         * @return Logger reference to enable operator<<
         */
        Logger &info();

        /**
         * @brief print warning message
         * @details sets the logger level to Level::WARN
         * @return Logger reference to enable operator<<
         */
        Logger &warn();

        /**
         * @brief print error message
         * @details sets the logger level to Level::ERROR
         * @return Logger reference to enable operator<<
         */
        Logger &error();

        /**
         * @brief class to handle fatal messages
         * @details this is a class that only lives temporarily, it will throw a runtime error on destruction!
         *
         * @param log Logger
         */
        class Fatal
        {
          public:
            /**
             * @brief constructor
             *
             * @param log pointer to the current Logger
             */
            Fatal(Logger *log);

            /**
             * @brief destructor
             * @details will throw an exception!
             */
            virtual ~Fatal() noexcept(false);

            /**
             * @brief print message that caused the application to terminate
             * @return current Logger
             */
            template <class T>
            Logger &operator<<(const T &msg)
            {
                return log_->error() << msg;
            }

          private:
            /// current Logger
            Logger *log_;
        };

        /**
         * @brief print error message and throw a fatal exception
         * @details return a temporary Fatal
         * @return Fatal
         */
        Fatal fatal();

      protected:
        /**
         * @brief set logging level
         * @details changes the prefix for printing status messages
         *
         * @param level Level
         */
        void setLevel(Level level);

      private:
        /// output stream
        std::ostream ostream_;

        /// current message prefix that is printed before each message
        std::string prefix_;

        /// current logging Level
        Level level_;

        /// flag to keep track if logging is enabled
        bool is_enabled_;

        /// flag to keep track if verbose messages are enabled
        bool is_verbose_;

        /// flag to keep track if debug messages are enabled
        bool debug_;

        /// keep track of the time logging was started, to give each message a time stamp
        Timer::TimePoint start_time_;

      public:
        /// typedef
        using Ptr = std::shared_ptr<Logger>;

        /**
         * @brief operator to handle any kind of message
         *
         * @param log Logger
         * @param msg message to be printed
         */
        template <class T>
        friend Logger &operator<<(Logger &log, const T &msg);

        /**
         * @brief operator to handle std::endl
         *
         * @param log Logger
         * @param pf usually std::endl
         */
        friend Logger &operator<<(Logger &log, ostream_manipulator pf);
    };

    template <class T>
    Logger &operator<<(Logger &log, const T &msg)
    {
        if (!log.is_enabled_ || (!log.is_verbose_ && log.level_ == Logger::Level::VERBOSE))
        {
            return log;
        }

        log.ostream_ << msg;

        return log;
    }

}  // namespace sackmesser