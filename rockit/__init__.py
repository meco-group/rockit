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
import sys

in_matlab = "libmwbuffer" in sys.modules.keys()

if in_matlab:
  try:
    dlopen_flags = sys.getdlopenflags()
    # Fixes crash of numpy in Matlab
    sys.setdlopenflags(10) # RTLD_DEEPBIND(8) | RTLD_NOW(2)
  except:
    pass

from .multiple_shooting import MultipleShooting
from .ocp import Ocp
from .stage import Stage
from .direct_method import DirectMethod
from .direct_collocation import DirectCollocation
from .spline_method import SplineMethod

from .single_shooting import SingleShooting
from .freetime import FreeTime
from .sampling_method import FreeGrid, UniformGrid, GeometricGrid, DensityGrid, DenseEdgesGrid
from .grouping_techniques import LseGroup
from .solution import OcpSolution
from .external.manager import external_method
from .casadi_helpers import rockit_pickle_context, rockit_unpickle_context


try:
  matlab_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
except:
  matlab_path = "not_found"

if in_matlab:
  try:
    sys.setdlopenflags(dlopen_flags)
    del dlopen_flags
  except:
    pass

del in_matlab
