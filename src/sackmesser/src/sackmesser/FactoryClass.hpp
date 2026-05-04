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

namespace sackmesser
{

    /**
     * @brief base class for creating
     * @details [long description]
     *
     * @tparam Derived [description]
     * @tparam Args [description]
     */
    template <class Derived, class... Args>
    class FactoryClass
    {
      public:
        /**
         * @brief default constructor
         */
        FactoryClass();

        /**
         * @brief default destructor
         */
        virtual ~FactoryClass();

      private:
        ///
        using Factory = sackmesser::Factory<Derived, Args...>;

      public:
        ///
        using SharedGroup = typename Factory::SharedGroup;

      public:
        /**
         * @brief [brief description]
         * @details [long description]
         * @return [description]
         */
        static Factory *getFactory();

        /**
         * @brief [brief description]
         * @details [long description]
         *
         * @param name [description]
         * @param args [description]
         *
         * @return [description]
         */
        static std::unique_ptr<Derived> createUnique(const std::string &name, Args &&...args);

        /**
         * @brief [brief description]
         * @details [long description]
         *
         * @param name [description]
         * @param args [description]
         *
         * @return [description]
         */
        static std::shared_ptr<Derived> createShared(const std::string &name, Args &&...args);
    };

}  // namespace sackmesser

#include <sackmesser/FactoryClass.hxx>