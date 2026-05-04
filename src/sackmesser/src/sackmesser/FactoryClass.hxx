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

#include <sackmesser/FactoryClass.hpp>

namespace sackmesser
{

    template <class Derived, class... Args>
    FactoryClass<Derived, Args...>::FactoryClass() = default;

    template <class Derived, class... Args>
    FactoryClass<Derived, Args...>::~FactoryClass() = default;

    template <class Derived, class... Args>
    typename FactoryClass<Derived, Args...>::Factory *FactoryClass<Derived, Args...>::getFactory()
    {
        static Factory factory;

        return &factory;
    }

    template <class Derived, class... Args>
    std::unique_ptr<Derived> FactoryClass<Derived, Args...>::createUnique(const std::string &name, Args &&...args)
    {
        getFactory()->createUnique(name, std::forward<Args>(args)...);
    }

    template <class Derived, class... Args>
    std::shared_ptr<Derived> FactoryClass<Derived, Args...>::createShared(const std::string &name, Args &&...args)
    {
        getFactory()->createUnique(name, std::forward<Args>(args)...);
    }

}  // namespace sackmesser