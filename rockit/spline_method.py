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
from .splines.micro_spline import bspline_derivative, eval_on_knots, get_greville_points
from casadi import sumsqr, vertcat, linspace, substitute, MX, evalf, vcat, horzsplit, veccat, DM, repmat, vvcat, vec
import numpy as np
import casadi as ca
from collections import defaultdict

from .casadi_helpers import vcat, ConstraintInspector, linear_coeffs, reshape_number

class SplineMethod(SamplingMethod):
    def __init__(self, **kwargs):
        SamplingMethod.__init__(self, **kwargs)
        self.clean()

    def clean(self):
        SamplingMethod.clean(self)
        self.constraint_inspector = None
        self.B = None
        self.tau = None
        self.G = None
        self.XU_sampled = None
        self.XU0_sampled = None
        self.XUF_sampled = None
        self.coeffs_and_der = None
        self.coeffs_epxr = None
        self.widths = None
        self.origins = None
        self.opti_advanced = None
        self.Q = None

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

        import networkx as nx
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
        assert nx.number_of_nodes(G)==0 or nx.is_forest(G)
        #nx.draw(G)
        #import pylab as plt
        #plt.show()

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
        self.v = vvcat(stage.variables['']+stage.variables['bspline'])
        self.free_time = False

        self.constraint_inspector = ConstraintInspector(self, stage)

        self.xu = ca.vertcat(stage.x,stage.u)

    def transcribe_event_after_varpar(self, stage, phase=1, **kwargs):
        self.constraint_inspector.finalize()

    def sample_xu(self, stage, refine):
        # Cache for B,tau and results
        if self.B is None:
            self.B = defaultdict(dict)
            self.tau = {}
            self.time = {}
            self.G = {}
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
        if self.groups:
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
            self.G[d] = get_greville_points(self.xi, d)

    def add_variables(self, stage, opti):

        self.add_variables_V(stage, opti)

        assert not self.time_grid.localize_t0 and not self.time_grid.localize_T

        self.control_grid = self.time_grid(self.t0, self.T, self.N)

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
                    self.origins[v_index] = {"L": L, "i": i, "w": s-i, "k": k, "d": d}
                    # Store scalarized coefficients
                    self.coeffs_epxr[v_index] = esplit[k]
                if d-i>0:
                    # Differentiate coefficient
                    e = bspline_derivative(e,self.xi,d-i)/self.T
        self.unique_widths = set(int(i) for i in self.widths.nonzeros())

        unique_refines = set([1]+[args["refine"] for _, _, args in stage._constraints["control"]])


        for refine in unique_refines:
            self.sample_xu(stage, refine)

        # We can know store states and controls evaluated on the control grid
        self.X = ca.horzsplit(ca.vcat(self.XU_sampled[1][:stage.nx])) if stage.nx else [DM(0,1)]*(self.N+1)
        self.U = ca.horzsplit(ca.vcat(self.XU_sampled[1][stage.nx:]))[:-1] if stage.nu else [DM(0,1)]*(self.N)

        # Below may improve efficiency, depends on the situation
        #self.X[0] = ca.vcat(self.XU0_expr[:stwidthstage.nx])
        #self.U[-1] = ca.vcat(self.XUF_expr[stage.nx:])

    def grid_gist(self, stage, expr, grid, include_first=True, include_last=True, transpose=False, refine=1):
        assert refine==1
        # What scalarized variables are we dependent on?
        v = self.xu
        arg1 = v
        arg2 = vvcat(self.signals.keys())
        arg = ca.vertcat(arg1,arg2)
        assert ca.is_linear(expr,arg)
        Jf = ca.Function('Jf',[],ca.linear_coeff(expr,ca.vertcat(arg1,arg2)))
        res = Jf.call([],False,False)
        Js = ca.horzsplit(res[0],[0,arg1.numel(),arg.numel()])
        bs = res[1]
        for J in Js:
            assert J.sparsity().is_selection(True)
        has_entries = [e.nnz()>0 for e in Js]
        assert has_entries.count(True)==1
        if has_entries[0]:
            deps = ca.sum1(Js[0].sparsity()).T.row()
            Jmul = Js[0][:,deps]
            widths = set([self.origins[i]["w"] for i in deps])
            assert len(widths)==1
            coeffs = ca.vcat([self.coeffs_epxr[i] for i in deps])
            d = self.origins[deps[0]]["d"]
            return self.t0+self.G[d]*self.T, (Jmul @ coeffs)+bs
        elif has_entries[1]:
            deps = ca.sum1(Js[1].sparsity()).T.row()
            vars = vvcat(self.signals.keys())[deps]
            Jmul = Js[1][:,deps]
            s = self.signals[vars]
            # Compute the degree and size of a BSpline coefficient needed
            G = get_greville_points(self.xi, s.degree)
            return self.t0+G*self.T, (Jmul @ s.coeff)+bs
        
    def grid_control(self, stage, expr, grid, include_first=True, include_last=True, transpose=False, refine=1):
        # What scalarized variables are we dependent on?
        v = self.xu
        J = ca.jacobian(expr,v)
        deps = ca.sum1(J.sparsity()).T.row()

        self.sample_xu(stage, refine)

        [v_symbols,v_expressions] = self.xu_symbols(stage, deps, self.XU_sampled[refine])

        # Modify expr, v_symbols and v_expressions when offsets are present
        v_expressions_with_offset = [(0,e) for e in v_expressions]

        syms = ca.symvar(expr)
        offsets = []
        modified_expr = []
        for s in syms:
            if s in stage._offsets:
                assert refine==1
                e, offset = stage._offsets[s]

                J = ca.jacobian(expr,v)
                deps = ca.sum1(J.sparsity()).T.row()

                [v_symbols_local,v_expressions] = self.xu_symbols(stage, deps, self.XU_sampled[refine])
                copy_v_symbols = [MX.sym(s.name()+"_offset_%d" % offset, s.sparsity()) for s in v_symbols]
                v_symbols+=copy_v_symbols
                v_expressions_with_offset+=[(offset,e) for e in v_expressions]
                
                offsets.append(s)
                modified_expr.append(substitute([e],v_symbols_local,copy_v_symbols)[0])

        expr = substitute([expr],offsets,modified_expr)[0]

        min_offset = int(np.min([o for o,_ in v_expressions_with_offset],initial=0))
        max_offset = int(np.max([o for o,_ in v_expressions_with_offset],initial=0))

        stop = self.N*refine+1

        v_expressions = []

        for o,e in v_expressions_with_offset:
            v_expressions.append(e[:,o-min_offset:o+stop-max_offset])

        time = self.time[refine][-min_offset:stop-max_offset]

        # End offset modifications

        fixed_parameters = MX(0, 1) if len(stage.parameters[''])==0 else vvcat(stage.parameters[''])

        spline_symbols = vvcat(self.signals.keys())
        spline_traj = []
        for p,v in self.signals.items():
            # Compute the degree and size of a BSpline coefficient needed
            s = self.N+v.degree
            assert p.size2()==1
            [_,B] = eval_on_knots(self.xi,v.degree,subsamples=refine-1)
            spline_traj.append(v.coeff @ B)
        spline_traj = vcat(spline_traj)

        f = ca.Function("f",v_symbols+[fixed_parameters,spline_symbols,stage.t],[expr])
        F = f.map(self.N*refine+1-max_offset+min_offset,len(v_symbols)*[False]+ [True,False,False])
        results = F(*v_expressions,fixed_parameters,spline_traj,time)

        return time, self.eval(stage, results)



    def add_constraints(self, stage, opti):
        self.add_constraints_before(stage, opti)
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

            _,results = self.grid_control(stage, canon, 'control', refine=refine)
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
                results_end = results_split[1]
                if not lb_inf:
                    self.opti.subject_to(self.eval(stage, lb <= results_min))
                    self.opti.subject_to(self.eval(stage, lb <= results_end))
                if not ub_inf:
                    self.opti.subject_to(self.eval(stage, results_max <= ub))
                    self.opti.subject_to(self.eval(stage, results_end <= ub))
            else:
                n = results.shape[1]
                lb = ca.repmat(lb,1,n)
                ub = ca.repmat(ub,1,n)
                self.opti.subject_to( self.eval(   stage, ca.vec(lb) <= (ca.vec(results) <= ca.vec(ub))   ) )

    def add_constraints_inf(self, stage, opti):

        v = self.xu

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

        A, Asignal, b = linear_coeffs(canon, v, vvcat(self.signals.keys()))
        A = evalf(A)
        Asignal = evalf(Asignal)
        b = evalf(b)

        assert A.nnz()==0 or Asignal.nnz()==0

        if A.nnz():

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

                if Sr:
                    self.opti.subject_to(self.eval(stage,lb[Sr] - b[Sr] <= (Ablock @ C <= ub[Sr]-b[Sr])))
        else:
            deps = ca.sum1(Asignal).T.row()
            vars = vvcat(self.signals.keys())[deps]
            Jmul = Asignal[:,deps]
            s = self.signals[vars]
            self.opti.subject_to(self.eval(stage,lb - b <= (Jmul @ s.coeff <= ub-b)))

    def set_initial(self, stage, master, initial):
        opti = master.opti if hasattr(master, 'opti') else master
        opti_initial = opti.initial()
        t0 = opti.debug.value(self.t0, opti_initial)
        T = opti.debug.value(self.T, opti_initial)
        initial_remainder = initial.__class__()
        for var, expr in initial.items():
            expr = reshape_number(var,expr)
            J = evalf(ca.jacobian(var,self.xu))

            # Selector for nonempty columns
            Sc = ca.sum1(J.sparsity()).T.row()
            if len(Sc)==0:
                initial_remainder[var] = expr
                continue

            expr = ca.inv(J[:,Sc]) @ expr
            if expr.shape[1]==1:
                f = ca.Function("f",[stage.t],[expr])

                Ls = set([e['L'] for i,e in enumerate(self.origins) if i in Sc])
                for L in Ls:
                    d = L-1
                    ks = []
                    js = []
                    j = 0
                    for i,e in enumerate(self.origins):
                        if i in Sc:
                            if e['L']==L:
                                assert e['i']==0
                                ks.append(e['k'])
                                js.append(j)
                            j+=1
                    target = self.coeffs_and_der[L][0][ks,:]
                    value = f(t0+self.G[d]*T)[js,:]
                    opti.set_initial(target, value)
            else:
                opti.set_initial(stage.sample(var,'gist')[1], expr)
        SamplingMethod.set_initial(self, stage, master, initial_remainder)