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
        assert ode.sparsity_in("t").nnz()==0, "Time dependent variables not supported in SplineMethod"
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
        # Obtain chains of differentiations (scalarised)

        # Use combined index: v=[x;u]
        def node_x(i):
            return i
        def node_u(i):
            return i+stage.nx

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

        """
        Example
            p = ocp.state(2)
            v = ocp.state(2)
            a = ocp.control(2)

            ocp.set_der(p, v)

            c = ocp.control(order=5)
            c = ocp.control(order=3)

            ocp.set_der(v, 6*a)

        Scalarized indices
        v_0: x_0
        v_1: x_1
        ...
        v_11: x_11
        v_12: u_0
        v_13: u_1
        v_14: u_2
        v_15: u_3

        Chains:
            [0, 1.0, 2, 6.0, 12] L: 3
            [1, 1.0, 3, 6.0, 13] L: 3p = ocp.state()
v = ocp.state()
a = ocp.control()

ocp.set_der(p, v)

#c = ocp.control(order=5)
#c = ocp.control(order=3)

ocp.set_der(v, a)
            [4, 1.0, 5, 1.0, 6, 1.0, 7, 1.0, 8, 1.0, 14] L: 6
            [9, 1.0, 10, 1.0, 11, 1.0, 15] L: 4

        Interpretation:
            dot(v_0) = x_2; dot(v_2) = 6*v_12

        """

        # Group chains according to length
        # The goals is to re-identify some vector structure from the scalarised chains
        self.groups = defaultdict(list)
        for chain in chains:
            L = len(chain)//2+1
            self.groups[L].append(chain)
        """
         {3: [[0, 1.0, 2, 6.0, 12], [1, 1.0, 3, 6.0, 13]],
          6: [[4, 1.0, 5, 1.0, 6, 1.0, 7, 1.0, 8, 1.0, 14]],
          4: [[9, 1.0, 10, 1.0, 11, 1.0, 15]]})
        """

        # Needed to make SamplingMethod happy
        self.v = vvcat(stage.variables[''])
        self.free_time = False

        self.constraint_inspector = ConstraintInspector(self, stage)
        self.constraint_inspector.finalize()

    def sample_xu(self, stage, refine):
        # Cache for B,tau and results
        if not hasattr(self,"B"):
            self.B = defaultdict(dict)
            self.tau = {}
            self.time = {}
            self.XU_sampled = defaultdict(lambda : [None]*(stage.nx+stage.nu))
            self.XU0_sampled = defaultdict(lambda : [None]*(stage.nx+stage.nu))
            self.XUF_sampled = defaultdict(lambda : [None]*(stage.nx+stage.nu))
        if refine in self.B:
            return
        # Evaluations of BSplines on a grid happens by matrix multiplication:
        # values = coefficients @ B (basis matrix)
        # The width of B is determined by the size of the grid
        # The height of B is determined by the degree of BSpline
        #
        # We can construct all needed Bs upfront, regardless of groups
        # Store different Bs using width as a key
        # We need to cover the highest-order degree and all degrees lower than that
        Lmax = max(self.groups.keys())
        dmax = Lmax-1
        for i in range(dmax+1):
            d = dmax-i
            [tau,B] = eval_on_knots(self.xi,dmax-i,subsamples=refine-1)
            self.B[refine][self.N+d] = B
            self.tau[refine] = tau
        self.time[refine] = self.time_grid(self.t0, self.T, self.N*refine)

        # Evaluate spline on the control grid
        for L,chains in self.groups.items():
            d = L-1
            s = self.N+d
            for i in range(L):
                xu_sampled = self.coeffs_and_der[L][i] @ self.B[refine][s-i]
                xu0 = self.coeffs_and_der[L][i] @ self.B[refine][s-i][:,0]
                xuf = self.coeffs_and_der[L][i] @ self.B[refine][s-i][:,-1]

                xu_sampled_split = ca.vertsplit(xu_sampled)
                xu0_split = ca.vertsplit(xu0)
                xuf_split = ca.vertsplit(xuf)
                for k,c in enumerate(chains):
                    v_index = c[2*i]
                    self.XU_sampled[refine][v_index] = xu_sampled_split[k]
                    self.XU0_sampled[refine][v_index] = xu0_split[k]
                    self.XUF_sampled[refine][v_index] = xuf_split[k]

    def add_variables(self, stage, opti):

        self.add_variables_V(stage, opti)

        assert not self.time_grid.localize_t0 and not self.time_grid.localize_T

        self.control_grid = self.time_grid(self.t0, self.T, self.N)

        # Grid for B-spline
        xi = DM(self.time_grid(0, 1, self.N)).T
        self.xi = xi

        # Vectorized storage of coeffients and derivatives
        self.coeffs_and_der = defaultdict(list)
        # Scalarized storage of coeffients and derivatives
        # Scalarized using vertsplit, ideally gets whole again after vertcat
        self.coeffs_epxr = [None]*(stage.nx+stage.nu)

        # For each scalarized variable, store the width of the coefficient
        # prepare a substitute(stage.x,coeff)
        self.widths = ca.DM.zeros(stage.nx+stage.nu)

        self.origins = [None]*(stage.nx+stage.nu)

        def consume(chains):
            heads = [c[0] for c in chains]
            return heads, [c[2:] for c in chains]

        # For each group of chains (grouped by length)
        for L,chains in self.groups.items():
            # Compute the degree and size of a BSpline coefficient needed
            d = L-1; s = self.N+d
            # Create a decision variable for coefficients for the highest degree variable
            # (or group of variables) in the chain
            e = opti.variable(len(chains), s)
            # Loop over length of chain
            for i in range(L):
                # Store coeffient
                self.coeffs_and_der[L].append(e)
                # Scalarize coeffient
                esplit = ca.vertsplit(e)
                # Loop over chains in group
                for k,c in enumerate(chains):
                    # Current combined index
                    v_index = c[2*i]
                    # Store width
                    self.widths[v_index] = s-i
                    self.origins[v_index] = {"L": L, "i": i, "w": s-i, "k": k}
                    # Store scalarized coefficients
                    self.coeffs_epxr[v_index] = esplit[k]
                if d-i>0:
                    # Differentiate coefficient
                    e = bspline_derivative(e,xi,d-i)/self.T
        self.unique_widths = set(int(i) for i in self.widths.nonzeros())

        unique_refines = set([1]+[args["refine"] for _, _, args in stage._constraints["control"]])


        for refine in unique_refines:
            self.sample_xu(stage, refine)

        # We can know store states and controls evaluated on the control grid
        self.X = ca.horzsplit(ca.vcat(self.XU_sampled[1][:stage.nx]))
        self.U = ca.horzsplit(ca.vcat(self.XU_sampled[1][stage.nx:]))[:-1]

        # Below may improve efficiency, depends on the situation
        #self.X[0] = ca.vcat(self.XU0_expr[:stwidthstage.nx])
        #self.U[-1] = ca.vcat(self.XUF_expr[stage.nx:])

    def grid_control(self, stage, expr, grid, include_first=True, include_last=True, transpose=False, refine=1):
        # What scalarized variables are we dependent on?
        v = ca.vertcat(stage.x,stage.u)
        J = ca.jacobian(expr,v)
        deps = ca.sum1(J.sparsity()).T.row()

        self.sample_xu(stage, refine)

        [v_symbols,v_expressions] = self.xu_symbols(stage, deps, self.XU_sampled[refine])
        f = ca.Function("f",v_symbols+[stage.p,stage.t],[expr])
        F = f.map(self.N*refine+1,len(v_symbols)*[False]+ [True,False])
        results = F(*v_expressions,stage.p,self.time[refine])
        
        return self.time[refine],results




    def add_constraints(self, stage, opti):
        assert "integrator" not in stage._constraints

        self.opti_advanced = self.opti.advanced
        self.add_constraints_inf(stage, opti)
        self.add_constraints_noninf(stage, opti)
    
    def xu_symbols(self,stage,v_indices,pool):
        self.v_symbols = stage.states+stage.controls
        
        # Which symbols are needed?
        self.symbol_map = []
        for i,e in enumerate(self.v_symbols):
            for k in range(e.numel()):
                self.symbol_map.append((i,k))

        active_symbols = list(sorted(set([self.symbol_map[i][0] for i in v_indices])))
        v_active_symbols = [self.v_symbols[e] for e in active_symbols]

        v_expressions = [[0]*e.numel() for e in self.v_symbols]

        for i in v_indices:
            v_expressions[self.symbol_map[i][0]][self.symbol_map[i][1]] = pool[i]

        v_active_symbols = [self.v_symbols[e] for e in active_symbols]
        v_active_expressions = [ca.vcat(v_expressions[i]) for i in active_symbols]

        return v_active_symbols, v_active_expressions

    def add_constraints_noninf(self, stage, opti):
        # Lump constraints together, based on refine parameter
        lbs = defaultdict(list)
        ubs = defaultdict(list)
        canons = defaultdict(list)
        for c, meta, args in stage._constraints["control"]:
            key = (args["refine"],args["group_refine"])
            (lb,canon,ub), mc = self.constraint_inspector.canon(c)

            lbs[key].append(lb)
            ubs[key].append(ub)
            canons[key].append(canon)
        
        keys = list(lbs.keys())

        # Loop over lumps
        for k in keys:
            (refine,group_refine) = k
            lb = ca.vcat(lbs[k])
            ub = ca.vcat(ubs[k])
            canon = ca.vcat(canons[k])
            v = ca.vertcat(stage.x,stage.u)

            # What scalarized variables are we dependent on?
            J = ca.jacobian(canon,v)
            deps = ca.sum1(J.sparsity()).T.row()

            [v_symbols,v_expressions] = self.xu_symbols(stage, deps, self.XU_sampled[refine])
            [_,v0_expressions] = self.xu_symbols(stage, deps, self.XU0_sampled[refine])
            [_,vf_expressions] = self.xu_symbols(stage, deps, self.XUF_sampled[refine])

            f = ca.Function("f",v_symbols+[stage.p,stage.t],[canon])
            F = f.map(self.N*refine+1,[False,True,False])

            results = F(*v_expressions,stage.p,self.time[refine])
            assert canon.is_column()
            canon_sym = MX.sym("canon_sym",canon.size1(),refine)
            # Do a grouping along refinement grid if requested
            if group_refine:
                assert not ca.depends_on(canon, stage.t)

                # lb <= canon <= ub
                # Check for infinities
                try:
                    lb_inf = np.all(np.array(evalf(lb)==-np.inf))
                except:
                    lb_inf = False
                try:
                    ub_inf = np.all(np.array(evalf(ub)==np.inf))
                except:
                    ub_inf = False

                f_min_group = ca.Function("helper",[canon_sym],[-group_refine(-canon_sym,axis=1)])
                f_max_group = ca.Function("helper",[canon_sym],[group_refine(canon_sym,axis=1)])
                fm_min_group = f_min_group.map(self.N)
                fm_max_group = f_max_group.map(self.N)

                results_split = horzsplit(results,[0,self.N*refine,self.N*refine+1])
                results_max = fm_max_group(results_split[0])
                results_min = fm_min_group(results_split[0])
                results_end = f(*vf_expressions,stage.p,self.control_grid[-1])
                if not lb_inf:
                    self.opti.subject_to(lb <= results_min)
                    self.opti.subject_to(lb <= results_end)
                if not ub_inf:
                    self.opti.subject_to(results_max <= ub)
                    self.opti.subject_to(results_end <= ub)
            else:
                self.opti.subject_to(ca.vec(ca.repmat(lb,1,self.N*refine+1)) <= (ca.vec(results) <= ca.vec(ca.repmat(ub,1,self.N*refine+1))))

    def add_constraints_inf(self, stage, opti):

        v = ca.vertcat(stage.x,stage.u)

        # Collect all inf constraints
        lbs = []
        ubs = []
        canons = []
        for c, meta, _ in stage._constraints["inf"]:
            assert not ca.depends_on(c, stage.t)
            (lb,canon,ub), mc = self.constraint_inspector.canon(c)
            
            assert ca.is_linear(canon, v)

            lbs.append(lb)
            ubs.append(ub)
            canons.append(canon)
        if len(lbs)==0:
            return

        # Work towards lb <= Av+b <= ub
        
        lb = ca.vcat(lbs)
        ub = ca.vcat(ubs)
        canon = ca.vcat(canons)

        A, b = linear_coeffs(canon, v)
        A = evalf(A)
        b = evalf(b)

        # Goal is to put constraints on coefficients instead of on v
        # However, different entries of v have different widths of coefficients
        # Hence, we will have to add separate constraints for each width

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
