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

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
//
#include <sackmesser/Logger.hpp>

namespace sackmesser
{

    /**
     * @brief enables calling functions between classes through Interface
     * @details This class is a member of Interface and can be accessed via Interface::getCallbacks
     */
    class Callbacks
    {
      public:
        /**
         * @brief constructor
         *
         * @param logger Logger
         */
        Callbacks(const Logger::Ptr &logger);

        /**
         * @brief default destructor
         */
        virtual ~Callbacks();

        /**
         * @brief invoke the callback with the given name for the given arguments
         * @details
         *
         * @param name the callback queue
         * @param argument arguments to pass to the callback
         */
        template <class... Arguments>
        void invoke(const std::string &name, const Arguments &...argument);

        template <class Result, class... Arguments>
        bool execute(const std::string &name, Result &result, const Arguments &...argument);

        /**
         * @brief add a callback queue
         * @details
         *
         * @param name name of the callback queue
         * @tparam Arguments types of the function arguments that will be passed to the callback
         */
        template <class... Arguments>
        void addQueue(const std::string &name);

        /**
         * @brief add the given function to the specified callback queue
         * @details
         *
         * @param name the callback queue
         * @param callback the callback function
         * @tparam Arguments types of the function arguments that will be passed to the callback
         */
        template <class... Arguments, class Function>
        void addCallbackToQueue(const std::string &name, const Function &callback);

        /**
         * @brief add a callback queue
         * @details
         *
         * @param name name of the callback queue
         * @tparam Result
         * @tparam Arguments types of the function arguments that will be passed to the callback
         */
        template <class Result, class... Arguments>
        void addCallback(const std::string &name, const std::function<bool(Result &, const Arguments &...)> &callback);

      protected:
        /// Logger
        Logger::Ptr logger_;

      private:
        class Queue;

        /**
         * @brief base class for callback queues
         */
        class Queue
        {
          public:
            /**
             * @brief default constructor
             * @details
             */
            Queue();

            /**
             * @brief default destructor
             * @details
             */
            virtual ~Queue();

            /**
             * @brief cast this queue to a derived class
             * @details
             *
             * @return pointer of the derived class
             */
            template <class Derived>
            Derived *cast();
        };

        /**
         * @brief implementation of the callback queue
         * @details
         *
         * @tparam Arguments types of the function arguments that will be passed to the callback
         */
        template <class... Arguments>
        class CallbackQueue : public Queue
        {
          public:
            ///
            using Callback = std::function<void(const Arguments &...)>;

            /**
             * @brief default constructor
             * @details
             */
            CallbackQueue() = default;

            /**
             * @brief default destructor
             * @details
             */
            virtual ~CallbackQueue() = default;

            /**
             * @brief add a callback to this queue
             * @details
             *
             * @param callback Callback that should be added
             */
            void addCallback(const Callback &callback);

            /**
             * @brief invoke all Callbacks in the queue
             * @details
             *
             * @param arguments function arguments
             */
            void invoke(const Arguments &...arguments);

          private:
            ///
            std::vector<Callback> callbacks_;
        };

        /**
         * @brief implementation of the callback queue
         * @details
         *
         * @tparam Arguments types of the function arguments that will be passed to the callback
         */
        template <class Result, class... Arguments>
        class ReturnCallback : public Queue
        {
          public:
            ///
            using Callback = std::function<bool(Result &, const Arguments &...)>;

            /**
             * @brief default constructor
             * @details
             */
            ReturnCallback(const Callback &callback);

            /**
             * @brief default destructor
             * @details
             */
            virtual ~ReturnCallback() = default;

            /**
             * @brief invoke all Callbacks in the queue
             * @details
             *
             * @param arguments function arguments
             */
            bool invoke(Result &result, const Arguments &...arguments);

          private:
            ///
            Callback callback_;
        };

        ///
        std::map<std::string, std::unique_ptr<Queue>> queues_;
    };

}  // namespace sackmesser

#include <sackmesser/Callbacks.hxx>