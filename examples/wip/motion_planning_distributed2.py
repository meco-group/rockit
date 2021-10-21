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

# agent peer node group

from rockit import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan
from casadi import vertcat, sumsqr


# A separate problem specification for each agent
# flexible agent topology
# solved in a distributed way on different agents


# This code is present only on agent 1

ocp = Ocp(T=FreeTime(10.0))

# Bicycle model
ocp.variable

x     = ocp.state()
V     = ocp.control()
ocp.set_der(x, V)
ocp.subject_to(ocp.at_t0(x)==0)
ocp.subject_to(0 <= (V+env_state<=1))
ocp.add_objective(ocp.integral(x**2))
ocp.method(MultipleShooting(N=20,M=4,intg='rk'))

x_neighbour = ocp.peer_output('x')
V_neighbour = ocp.peer_output('V')
x_neighbour = ocp.peer_output(agent=1)['x']

x_neighbour = ocp.receive('x')

#ocp.broadcast(x,'x')
ocp.

ocp.subject_to(abs(x-x_neighbour)>=1,type='distributed') # here, the constraints is assuled to hold for each possible neighbour
                                                         # transcribed to nlp with maximum number of neighbours  (say 10)
                                                         # results in 10 opti constraints 

# distance between ships depends on size,
# maybe you want small ships to go behind bigs ships (formation)


# Topology: the exact amount of agents, and the connectivity graph
#   Fixed or flexible (connectivity graph) -- let's keep amount of agents fixed
#
# Properties: immutable quantities that describe an agent
#             (for mutable -> use state?)
#   Known up-front or not known upfront
#
# Outputs: things to cummicate with neighbours, function of states and controls, or variables
#   Specification is known upfront (e.g. position [gps coordinates])
#   Some outputs may become unavailable
#   Maybe a property is just a scalar output??

# Problem structure is baked, parametrically
# perhaps you allocate constraints that may get used, and disable (set bounds to infinity) them until needed

# Communication aspects
#   - querying versus broadcasting


# Fixed topology, known properties
peers = {0: BigShip(), 1: SmallShip(), 2: SmallShip()}

x_neighbour = ocp.peer_output(agent=0,'x')
ocp.subject_to(abs(x-x_neighbour)>=peer[0].safety_distance,type='distributed')
x_neighbour = ocp.peer_output(agent=1,'x')
ocp.subject_to(abs(x-x_neighbour)>=peer[1].safety_distance,type='distributed')
x_neighbour = ocp.peer_output(agent=2,'x')
ocp.subject_to(abs(x-x_neighbour)>=peer[2].safety_distance,type='distributed')



# Flexible connectivity graph, known properties
for peer in ocp.peers: # always the max amount
    x_neighbour = peer.output('x')
    ocp.subject_to(abs(x-x_neighbour)>=peer.property('safety_distance'),type='distributed')  #internally, you iterate over list of peers


ocp.solver(ADMM(local='ipopt',coupling='sqpmethod',local_options={},coupling_options={}))


class Communicator:
    def send(self,key,output):
        pass
    def receive(self)
        result = dict()
        return result

class UDPCommunicator(Communicator):
    def send(self,key,output):
        # send pickled dictionary over udp
    def receive(self)
        # unpickle dictionary received over udp
        result = dict()
        return result

class ROSCommunicator(Communicator):
    pass

ocp.communicator(Communicator)
# polutes the normal rockit API


ocp.solver()


# MPC loop
while True:
    ocp.solve()


for i in range(10):
    ocp.add_peer(Communicator)
# ocp.peers defined only after




# should we not bring it in another class?

peers = Peers()

from ros_stuff import RosCommunicator

ocp = DistributedOcp(Communicator(),max_peers=10)

ocp.scheme('admm')
ocp.solver('ipopt')
ocp.coupling_solver('sqpmethod')

for peer in ocp.peers: # always the max amount
    x_neighbour = peer.output('x')
    ocp.subject_to(abs(x-x_neighbour)>=peer.property('safety_distance'))


class DistributedOcp(Ocp)
    pass

# make a bit more concrete already
# 