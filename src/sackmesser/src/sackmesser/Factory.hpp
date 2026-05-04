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

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>

namespace sackmesser
{
    class Interface;

    /**
     * @brief class factory for runtime selection of derived classes
     * @details
     *
     * @tparam Base base class that other will inherit from
     * @tparam Args type argument list of the constructor
     */
    template <class Base, class... Args>
    class Factory
    {
      public:
        /**
         * @brief default constructor
         * @details
         */
        Factory();

        /**
         * @brief default destructor
         * @details
         */
        virtual ~Factory();

        /**
         * @brief base class for loading classes
         * @details
         */
        class ClassLoaderBase
        {
          public:
            /**
             * @brief default constructor
             * @details
             */
            ClassLoaderBase();

            /**
             * @brief default destructor
             * @details
             */
            virtual ~ClassLoaderBase();

            /**
             * @brief create a unique pointer for the Derived class
             * @details
             *
             * @param args arguments
             * @return unique pointer of Derived class cast to Base class
             */

            virtual std::unique_ptr<Base> createUnique(Args &&...args) = 0;

            /**
             * @brief create a shared pointer for the Derived class
             * @details
             *
             * @param args arguments
             * @return shared pointer of Derived class cast to Base class
             */
            virtual std::shared_ptr<Base> createShared(Args &&...args) = 0;
        };

        /**
         * @brief class loader for Derived class
         * @details
         *
         * @tparam Derived derived class that will be loaded
         */
        template <class Derived>
        class ClassLoader : public ClassLoaderBase
        {
          public:
            /**
             * @brief default constructor
             * @details
             */
            ClassLoader();

            /**
             * @brief default destructor
             * @details
             */
            virtual ~ClassLoader();

            /**
             * @brief create a unique pointer for the Derived class
             * @details
             *
             * @param args arguments
             * @return unique pointer of Derived class cast to Base class
             */
            std::unique_ptr<Base> createUnique(Args &&...args);

            /**
             * @brief create a shared pointer for the Derived class
             * @details
             *
             * @param args arguments
             * @return shared pointer of Derived class cast to Base class
             */
            std::shared_ptr<Base> createShared(Args &&...args);
        };

      public:
        class SharedGroup
        {
          public:
            SharedGroup(const std::shared_ptr<Interface> &interface, const std::string &name);

            void forEach(const std::function<void(std::shared_ptr<Base>)> &function) const;

            std::shared_ptr<Base> get(const std::string &name) const;

          private:
            std::map<std::string, std::shared_ptr<Base>> list_;

          public:
            friend class Factory;
        };

      public:
        /**
         * @brief add a ClassLoader
         * @details
         *
         * @param name name of the derived class
         * @tparam Derived type of the derived class
         */
        template <class Derived>
        void add(const std::string &name);

        /**
         * @brief create a unique pointer of the derived class with the given name
         * @details
         *
         * @param name name of the derived class
         * @param args arguments for the constructor
         *
         * @return unique pointer of derived class
         */
        std::unique_ptr<Base> createUnique(const std::string &name, Args &&...args) const;

        /**
         * @brief create a shared pointer of the derived class with the given name
         * @details
         *
         * @param name name of the derived class
         * @param args arguments for the constructor
         *
         * @return shared pointer of derived class
         */
        std::shared_ptr<Base> createShared(const std::string &name, Args &&...args) const;

      private:
        /// stores all class loaders
        std::map<std::string, std::unique_ptr<ClassLoaderBase>> loaders_;
    };

}  // namespace sackmesser

#include <sackmesser/Factory.hxx>

#define REGISTER_CLASS_WITH_ID(BASE, DERIVED, ID, NAME) \
    namespace                                           \
    {                                                   \
        struct ClassLoader##ID                          \
        {                                               \
            ClassLoader##ID()                           \
            {                                           \
                BASE::getFactory()->add<DERIVED>(NAME); \
            }                                           \
        };                                              \
        static ClassLoader##ID class_creator##ID;       \
    }

#define REGISTER_CLASS_T(BASE, DERIVED, ID, NAME) REGISTER_CLASS_WITH_ID(BASE, DERIVED, ID, NAME)

#define REGISTER_CLASS(BASE, DERIVED, NAME) REGISTER_CLASS_T(BASE, DERIVED, __COUNTER__, NAME)