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

#include <sackmesser/Callbacks.hpp>

namespace sackmesser
{

    template <class... Arguments>
    void Callbacks::invoke(const std::string &name, const Arguments &...argument)
    {
        if (queues_.find(name) != queues_.end())
        {
            CallbackQueue<Arguments...> *queue = queues_.at(name)->cast<CallbackQueue<Arguments...>>();

            if (queue)
            {
                queue->invoke(argument...);
            }
            else
            {
                logger_->warn() << "CallbackHandler: unable to invoke callback " << name << std::endl;
            }
        }
    }

    template <class Result, class... Arguments>
    bool Callbacks::execute(const std::string &name, Result &result, const Arguments &...argument)
    {
        if (queues_.find(name) != queues_.end())
        {
            ReturnCallback<Result, Arguments...> *callback = queues_.at(name)->cast<ReturnCallback<Result, Arguments...>>();

            if (callback)
            {
                return callback->invoke(result, argument...);
            }
        }

        logger_->warn() << "CallbackHandler: unable to execute callback " << name << std::endl;

        return false;
    }

    template <class... Arguments>
    void Callbacks::addQueue(const std::string &name)
    {
        queues_.emplace(name, std::make_unique<CallbackQueue<Arguments...>>());

        logger_->info() << "CallbackHandler: added callback queue " << name << std::endl;
    }

    template <class... Arguments, class Function>
    void Callbacks::addCallbackToQueue(const std::string &name, const Function &callback)
    {
        if (queues_.find(name) != queues_.end())
        {
            CallbackQueue<Arguments...> *queue = queues_.at(name)->cast<CallbackQueue<Arguments...>>();

            if (queue)
            {
                queue->addCallback(callback);

                logger_->info() << "CallbackHandler: added callback to queue " << name << std::endl;
            }
            else
            {
                logger_->warn() << "CallbackHandler: unable to add callback to queue " << name << std::endl;
            }
        }
    }

    template <class Result, class... Arguments>
    void Callbacks::addCallback(const std::string &name, const std::function<bool(Result &, const Arguments &...)> &callback)
    {
        if (queues_.find(name) != queues_.end())
        {
            logger_->warn() << "CallbackHandler: unable to add callback " << name << ": it already exists!" << std::endl;
        }
        else
        {
            queues_.emplace(name, std::make_unique<ReturnCallback<Result, Arguments...>>(callback));

            logger_->info() << "CallbackHandler: added callback queue " << name << std::endl;
        }
    }

    template <class Type>
    Type *Callbacks::Queue::cast()
    {
        return dynamic_cast<Type *>(this);
    }

    template <class... Arguments>
    void Callbacks::CallbackQueue<Arguments...>::addCallback(const Callback &callback)
    {
        callbacks_.push_back(callback);
    }

    template <class... Arguments>
    void Callbacks::CallbackQueue<Arguments...>::invoke(const Arguments &...arguments)
    {
        for (const Callback &callback : callbacks_)
        {
            callback(arguments...);
        }
    }

    template <class Result, class... Arguments>
    Callbacks::ReturnCallback<Result, Arguments...>::ReturnCallback(const Callback &callback) : callback_(callback)
    {}

    template <class Result, class... Arguments>
    bool Callbacks::ReturnCallback<Result, Arguments...>::invoke(Result &result, const Arguments &...arguments)
    {
        return callback_(result, arguments...);
    }

}  // namespace sackmesser