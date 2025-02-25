#
#     This file is part of rockit.
#
#     rockit -- Rapid Optimal Control Kit
#     Copyright (C) 2019 MECO, KU Leuven. All rights reserved.
#
#     Rockit is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     Rockit is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
cmake_flags = []
compiler = "gcc"
cmake_build_type = "Release"

class GlobalOptions:
    @staticmethod
    def set_cmake_flags(flags):
        global cmake_flags
        cmake_flags = flags
    @staticmethod
    def get_cmake_flags():
        return cmake_flags
    @staticmethod
    def set_compiler(arg):
        global compiler
        compiler = arg
    @staticmethod
    def get_compiler():
        return compiler
    @staticmethod
    def set_cmake_build_type(arg):
        global cmake_build_type
        cmake_build_type = arg
    @staticmethod
    def get_cmake_build_type():
        return cmake_build_type