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
import os
import importlib

def external_method(name, **kwargs):
    try:
        external = importlib.import_module('.external.' + name, package='rockit')
    except ModuleNotFoundError:
        externals = []
        with os.scandir(os.path.dirname(os.path.realpath(__file__))) as it:
            for entry in it:
                if not entry.name.startswith('.') and not entry.name.startswith('_') and not entry.is_file():
                    externals.append(entry.name)
        externals = ",".join(["'%s'" % e for e in externals])
        raise Exception("external '%s' not found. Available: %s." % (name, externals))
    return external.method(**kwargs)

    