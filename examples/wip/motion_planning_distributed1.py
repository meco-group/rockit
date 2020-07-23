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

"""
Motion planning
===============

Simple motion planning with circular obstacle
"""

from rockit import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan
from casadi import vertcat, sumsqr


# A centralised problem specification,
# fixed agent topology
# solved in a distributed way on different agents

agent1 = Agent()
agent2 = Agent()

ocp = Ocp(T=FreeTime(10.0))

env_state = ocp.state()
u    = ocp.control()

ocp.set_der(env_state, u)

agent1 = ocp.agent()

# Bicycle model
x1     = agent1.state()
V1     = agent1.control()
agent1.set_der(x, V)
agent1.subject_to(agent1.at_t0(x)==0)
agent1.subject_to(0 <= (V+env_state<=1))
agent1.add_objective(agent1.integral(x**2))
agent1.solver('ipopt')
agent1.method(MultipleShooting(N=20,M=4,intg='rk'))

agent2 = ocp.agent()
# = agent1.clone()

# Bicycle model
x2     = agent2.state()
V2     = agent2.control()
agent2.set_der(x, V)
agent2.subject_to(agent2.at_t0(x)==0)
agent2.subject_to(0 <= (V<=1))
agent2.add_objective(agent2.integral(x**2))
agent2.solver('ipopt')
agent2.method(MultipleShooting(N=20,M=4,intg='rk'))

ocp.subject_to(abs(x1-x2)>=1) # -> slack variables fort self+neighbours
# admm quadratic penalties on slacks




ocp.solver()
