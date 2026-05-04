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

#include <sackmesser/Counter.hpp>

namespace sackmesser
{

    template <typename T>
    Counter<T>::Counter(const T &max) : Counter(T(0), max)
    {}

    template <typename T>
    Counter<T>::Counter(const T &min, const T &max) : value_(min), min_(min), max_(max)
    {}

    template <typename T>
    Counter<T>::~Counter() = default;

    template <typename T>
    T &Counter<T>::operator++()
    {
        value_ += T(1);

        if (value_ >= max_)
        {
            value_ = min_;
        }

        return value_;
    }

}  // namespace sackmesser