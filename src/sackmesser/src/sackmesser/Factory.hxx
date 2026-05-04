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

#include <sackmesser/Factory.hpp>
//
#include <sackmesser/Configurations.hpp>
#include <sackmesser/Interface.hpp>

namespace sackmesser
{
    template <class Base, class... Args>
    Factory<Base, Args...>::Factory() = default;

    template <class Base, class... Args>
    Factory<Base, Args...>::~Factory()
    {
        loaders_.clear();
    }

    template <class Base, class... Args>
    Factory<Base, Args...>::ClassLoaderBase::ClassLoaderBase() = default;

    template <class Base, class... Args>
    Factory<Base, Args...>::ClassLoaderBase::~ClassLoaderBase() = default;

    template <class Base, class... Args>
    template <class Derived>
    Factory<Base, Args...>::ClassLoader<Derived>::ClassLoader() = default;

    template <class Base, class... Args>
    template <class Derived>
    Factory<Base, Args...>::ClassLoader<Derived>::~ClassLoader() = default;

    template <class Base, class... Args>
    template <class Derived>
    std::unique_ptr<Base> Factory<Base, Args...>::ClassLoader<Derived>::createUnique(Args &&...args)
    {
        return std::make_unique<Derived>(std::forward<Args>(args)...);
    }

    template <class Base, class... Args>
    template <class Derived>
    std::shared_ptr<Base> Factory<Base, Args...>::ClassLoader<Derived>::createShared(Args &&...args)
    {
        return std::make_shared<Derived>(std::forward<Args>(args)...);
    }

    template <class Base, class... Args>
    template <class Derived>
    void Factory<Base, Args...>::add(const std::string &name)
    {
        loaders_.emplace(std::make_pair(name, std::make_unique<ClassLoader<Derived>>()));
    }

    template <class Base, class... Args>
    std::unique_ptr<Base> Factory<Base, Args...>::createUnique(const std::string &name, Args &&...args) const
    {
        if (name.empty())
        {
            return nullptr;
        }

        if (loaders_.find(name) == loaders_.end())
        {
            throw std::runtime_error("ClassFactory: no constructor for " + name);
        }

        return loaders_.at(name)->createUnique(std::forward<Args>(args)...);
    }

    template <class Base, class... Args>
    std::shared_ptr<Base> Factory<Base, Args...>::createShared(const std::string &name, Args &&...args) const
    {
        if (name.empty())
        {
            return nullptr;
        }

        if (loaders_.find(name) == loaders_.end())
        {
            throw std::runtime_error("ClassFactory: no constructor for " + name);
        }

        return loaders_.at(name)->createShared(std::forward<Args>(args)...);
    }

    template <class Base, class... Args>
    Factory<Base, Args...>::SharedGroup::SharedGroup(const std::shared_ptr<Interface> &interface, const std::string &base_name)
    {
        std::vector<std::string> names;
        interface->getConfigurations()->loadParameter(base_name + "s", &names);

        for (const std::string &name : names)
        {
            std::string type;
            interface->getConfigurations()->loadParameter(base_name + "/" + name + "/type", &type);

            list_.emplace(std::make_pair(name, Base::getFactory()->createShared(type, interface, base_name + "/" + name)));
        }
    }

    template <class Base, class... Args>
    void Factory<Base, Args...>::SharedGroup::forEach(const std::function<void(std::shared_ptr<Base>)> &function) const
    {
        for (auto &it : list_)
        {
            function(it.second);
        }
    }

    template <class Base, class... Args>
    std::shared_ptr<Base> Factory<Base, Args...>::SharedGroup::get(const std::string &name) const
    {
        return list_.at(name);
    }

}  // namespace sackmesser
