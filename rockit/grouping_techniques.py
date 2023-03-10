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

import casadi as ca

class GroupingTechnique:
    def __hash__(self):
        return hash(self.tuple)
    def __eq__(self, other):
        return self.tuple==other.tuple
    @property
    def tuple(self):
        return ()

    def __bool__(self):
        return False

class LseGroup(GroupingTechnique):
    class_id = 0
    def __init__(self, margin_abs=None):
        self.margin_abs = margin_abs

    @property
    def tuple(self):
        return (self.class_id, "margin_abs", self.margin_abs)

    def __call__(self, M, axis=0):
        if axis==1 and M.is_row():
            try:
                logsumexp = ca.logsumexp
            except:
                def logsumexp(x, margin):
                    alpha = ca.log(x.size1()) / margin
                    return ca.log(ca.sum1(ca.exp(alpha*x)))/alpha
            return logsumexp(M.T, self.margin_abs)
        else:
            raise Exception("Not implemented")

    def __bool__(self):
        return True