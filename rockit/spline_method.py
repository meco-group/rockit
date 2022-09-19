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

from .sampling_method import SamplingMethod
from .splines.micro_spline import bspline_derivative, eval_on_knots
from casadi import sumsqr, vertcat, linspace, substitute, MX, evalf, vcat, horzsplit, veccat, DM, repmat, vvcat, vec
import numpy as np
import casadi as ca
import networkx as nx
from collections import defaultdict

from .casadi_helpers import vcat, ConstraintInspector, linear_coeffs

class SplineMethod(SamplingMethod):
    def __init__(self, **kwargs):
        SamplingMethod.__init__(self, **kwargs)

    def transcribe_start(self, stage, opti):
        # Inspect system
        ode = stage._ode()
        assert ode.numel_out("alg")==0, "DAE not supported in SplineMethod"
        assert ode.sparsity_in("t").nnz()==0, "Time dependant variables not supported in SplineMethod"
        args = ode.convert_in(ode.mx_in())
        res = ode(**args)

        # Obtained linearised dynamics
        A = ca.jacobian(res['ode'],args["x"])
        B = ca.jacobian(res['ode'],args["u"])

        # Actually, only allow truly linear systems
        try:
            A = evalf(A)
            B = evalf(B)
        except:
            raise Exception("Only linear systems supported in SplineMethod")


        def node_x(i):
            return i
        def node_u(i):
            return i+stage.nx
        # Use combined index: [x;u]

        # Obtain chains of differentiations

        # Graph
        # Point from higher-order state to lower-order or control dependencies
        G = nx.DiGraph()
        for i in range(B.size1()):
            G.add_node(node_x(i))
        for i in range(B.size2()):
            G.add_node(node_u(i)) 
        for r,c in zip(*A.sparsity().get_triplet()):
            G.add_edge(node_x(r), node_x(c), weight=float(A[r,c]))
        for r,c in zip(*B.sparsity().get_triplet()):
            G.add_edge(node_x(r), node_u(c), weight=float(B[r,c]))
        assert nx.is_forest(G)
        assert len(list(nx.isolates(G)))==0


        chains = []
        for nodes in nx.weakly_connected_components(G):
            chain = []
            sub = G.subgraph(nodes)
            assert nx.is_arborescence(sub)           # Check that each lower-order state or control is only used once as dependency for a higher-state
            assert nx.is_arborescence(sub.reverse()) # Check that each higher-order state has a single dependency
            root = [n for n,d in sub.in_degree() if d==0][0]
            edges = sub.out_edges
            e = root
            while True:
                chain.append(e)
                edge = list(edges(e,data=True))
                if len(edge)==0: break
                edge = edge[0]
                e = edge[1]
                chain.append(edge[2]["weight"])
                assert edge[2]["weight"]==1.0
            chains.append(chain)


        print(chains)
        """
        Example
            p = ocp.state()
            v = ocp.state()
            a = ocp.control()

            ocp.set_der(p, v)

            c = ocp.control(order=5)
            c = ocp.control(order=3)

            ocp.set_der(v, 6*a)

            [('x', 0), 1.0, ('x', 1), 6.0, ('u', 0)]
            [('x', 2), 1.0, ('x', 3), 1.0, ('x', 4), 1.0, ('x', 5), 1.0, ('x', 6), 1.0, ('u', 1)]
            [('x', 7), 1.0, ('x', 8), 1.0, ('x', 9), 1.0, ('u', 2)]

        """

        groups = defaultdict(list)

        for chain in chains:
            L = len(chain)//2+1
            groups[L].append(chain)


        self.groups = groups

        self.v = vvcat(stage.variables[''])
        self.free_time = False

        self.constraint_inspector = ConstraintInspector(self, stage)
        self.constraint_inspector.finalize()


    def add_variables(self, stage, opti):

        self.add_variables_V(stage, opti)

        assert not self.time_grid.localize_t0 and not self.time_grid.localize_T

        self.control_grid = self.time_grid(self.t0, self.T, self.N)

        xi = DM(self.time_grid(0, 1, self.N)).T

        self.B = {}
        Lmax = max(self.groups.keys())
        dmax = Lmax-1
        for i in range(dmax+1):
            d = dmax-i
            [tau,B] = eval_on_knots(xi,dmax-i)
            self.B[self.N+d] = B
        self.tau = tau
        print(self.B, self.tau)

        # prepare a substitute(stage.x,coeff)
        self.widths = ca.DM.zeros(stage.nx+stage.nu)

        self.coeffs_epxr = [None]*(stage.nx+stage.nu)
        self.coeffs_and_der = defaultdict(list)

        def consume(chains):
            heads = [c[0] for c in chains]
            return heads, [c[2:] for c in chains]

        for L,chains in self.groups.items():
            print(L,chains)
            d = L-1
            s = self.N+d
            coeffs = opti.variable(len(chains), s)
            e = coeffs
            for i in range(L):
                self.coeffs_and_der[d].append(e)
                c_indices, chains = consume(chains)
                self.widths[c_indices] = s-i
                esplit = ca.vertsplit(e)
                for k,j in enumerate(c_indices):
                    self.coeffs_epxr[j] = esplit[k]
                if d-i>0:
                    e = bspline_derivative(e,xi,d-i)/self.T


        self.unique_widths = set(int(i) for i in self.widths.nonzeros())
        
        self.XU_expr = [None]*(stage.nx+stage.nu)
        self.XU0_expr = [None]*(stage.nx+stage.nu)
        self.XUF_expr = [None]*(stage.nx+stage.nu)

        self.X = [None] * (self.N+1)
        self.U = [None] * self.N

        # Evaluate spline on knots
        for L,chains in self.groups.items():
            d = L-1
            s = self.N+d
            for i in range(L):
                c_indices, chains = consume(chains)
                xu_sampled = self.coeffs_and_der[d][i] @ self.B[s-i]
                xu0 = self.coeffs_and_der[d][i] @ self.B[s-i][:,0]
                xuf = self.coeffs_and_der[d][i] @ self.B[s-i][:,-1]
                xu_sampled_split = ca.vertsplit(xu_sampled)
                xu0_split = ca.vertsplit(xu0)
                xuf_split = ca.vertsplit(xuf)
                for k,j in enumerate(c_indices):
                    self.XU_expr[j] = xu_sampled_split[k]
                    self.XU0_expr[j] = xu0_split[k]
                    self.XUF_expr[j] = xuf_split[k]
        self.X = ca.horzsplit(ca.vcat(self.XU_expr[:stage.nx]))
        self.U = ca.horzsplit(ca.vcat(self.XU_expr[stage.nx:]))[:-1]

        #self.X[0] = ca.vcat(self.XU0_expr[:stage.nx])
        #self.U[0] = ca.vcat(self.XU0_expr[stage.nx:])
        #self.X[-1] = ca.vcat(self.XUF_expr[:stage.nx])
        #self.U[-1] = ca.vcat(self.XUF_expr[stage.nx:])


    def add_constraints(self, stage, opti):

        self.opti_advanced = self.opti.advanced

        xu = ca.vertcat(stage.x,stage.u)

        # Collect all inf constraints
        lbs = []
        ubs = []
        canons = []
        for c, meta, _ in stage._constraints["inf"]:
            assert not ca.depends_on(c, stage.t)
            (lb,canon,ub), mc = self.constraint_inspector.canon(c)
            
            assert ca.is_linear(canon, xu)

            lbs.append(lb)
            ubs.append(ub)
            canons.append(canon)
        
        lb = ca.vcat(lbs)
        ub = ca.vcat(ubs)
        canon = ca.vcat(canons)

        A, b = linear_coeffs(canon, xu)
        A = evalf(A)
        b = evalf(b)

        # Partition constraints into blocks per width
        for w in self.unique_widths:
            # Selector for specific width
            Sw = np.nonzero(self.widths==w)[0]
            Ablock = A[:,Sw]
            coeffs_epxr_block = [self.coeffs_epxr[e] for e in Sw]
            # Selector for nonempty rows
            Sr = ca.sum2(Ablock.sparsity()).row()
            # Selector for nonempty columns
            Sc = ca.sum1(Ablock.sparsity()).T.row()
            Ablock = Ablock[Sr,Sc]
            C = ca.vcat([coeffs_epxr_block[e] for e in Sc])


            # 
            Sr = ca.sum2(Ablock.sparsity()).row()
            if Sr:

                self.opti.subject_to(lb[Sr] - b[Sr] <= (Ablock @ C <= ub[Sr]-b[Sr]))
