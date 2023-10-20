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
from casadi import sumsqr, horzcat, vertcat, linspace, substitute, MX, evalf,\
                   vcat, collocation_points, collocation_interpolators, hcat,\
                   repmat, DM, sum2, mtimes, vvcat, depends_on, Function
from .casadi_helpers import get_ranges_dict, HashOrderedDict, HashDict, is_numeric
import casadi as ca
from itertools import repeat
try:
    from casadi import collocation_coeff
except:
    def collocation_coeff(tau):
        [C, D] = collocation_interpolators(tau)
        d = len(tau)
        tau = [0]+tau
        F = [None]*(d+1)
        # Construct polynomial basis
        for j in range(d+1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(d+1):
                if r != j:
                    p *= np.poly1d([1, -tau[r]]) / (tau[j]-tau[r])
            
            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            F[j] = pint(1.0)
    
        return (hcat(C[1:]), vcat(D), hcat(F[1:]))

import numpy as np

class DirectCollocation(SamplingMethod):
    def __init__(self, degree=4, scheme='radau', **kwargs):
        SamplingMethod.__init__(self, **kwargs)
        self.degree = degree
        self.tau = collocation_points(degree, scheme)
        [self.C, self.D, self.B] = collocation_coeff(self.tau)
        self.clean()

    def clean(self):
        SamplingMethod.clean(self)
        self.Zc = []  # List that will hold algebraic decision variables list(N,list(M,nz x degree))
        self.Xc = []  # List that will hold helper collocation states list(N,list(M, degree+1))
        self.X_intg = []
        self.Xc_pure = []
        self.Zc0 = []
        self.Xc_vars = []
        self.Xc_vars0 = []
        self.Zc_vars_base = []
        self.Zc_vars_rest = []


    def add_variables(self, stage, opti):
        scale_x = stage._scale_x
        scale_z = stage._scale_z
        scale_u = stage._scale_u

        # We are creating variables in a special order such that the resulting constraint Jacobian
        # is block-sparse
        x = opti.variable(stage.nx, scale=scale_x)
        self.X.append(x)
        self.Q.append(DM.zeros(stage.nxq))
        self.add_variables_V(stage, opti)
        z = opti.variable(stage.nz, scale=scale_z)
        self.Zc_vars_base.append(z)

        for k in range(self.N):
            self.U.append(opti.variable(stage.nu, scale=scale_u) if stage.nu>0 else MX(0,1))
            xr = []
            zr = []
            Xc = []
            Zc = []
            for i in range(self.M):
                xc = opti.variable(stage.nx, self.degree, scale=repmat(scale_x, 1, self.degree))
                xr.append(xc)
                zc = opti.variable(stage.nz, self.degree-1, scale=repmat(scale_z, 1, self.degree-1))
                x0 = x if i==0 else opti.variable(stage.nx, scale=scale_x)
                self.X_intg.append(x0)
                self.Xc_pure.append(xc)
                Xc.append(horzcat(x0, xc))
                self.Xc_vars.append(xc if i==0 else horzcat(x0, xc))
                self.Xc_vars0.append(repmat(x, 1, self.degree if i==0 else self.degree+1))
                z0 = z if i==0 else opti.variable(stage.nz, scale=scale_z)
                zr.append(horzcat(z0, zc))
                Zc.append(horzcat(z0,zc))
                self.Zc_vars_rest.append(zc if i==0 else horzcat(z0, zc))
                self.Zc0.append(repmat(z, 1, self.degree-1 if i==0 else self.degree))

            self.xr.append(xr)
            self.zr.append(zr)
            self.Xc.append(Xc)
            self.Zc.append(Zc)
            x = opti.variable(stage.nx, scale=scale_x)
            self.X.append(x)
            self.Q.append(None)
            z = opti.variable(stage.nz, scale=scale_z)
            self.Zc_vars_base.append(z)
            self.add_variables_V_control(stage, opti, k)

        self.Xc_vars = hcat(self.Xc_vars)
        self.Xc_vars0 = hcat(self.Xc_vars0)
        self.Zc0 = hcat(self.Zc0)
        self.Zc_vars_base = hcat(self.Zc_vars_base)
        self.Zc_vars_rest = hcat(self.Zc_vars_rest)

        self.add_variables_V_control_finalize(stage, opti)

    def add_constraints(self, stage, opti):
        self.add_constraints_before(stage, opti)
        scale_x = stage._scale_x
        scale_der_x = stage._scale_der_x
        scale_z = stage._scale_z
        self.poly_coeff_q = None

        # Obtain the discretised system
        f = stage._ode()

        ps = []
        tau_root = [0] + self.tau
        # Construct polynomial basis
        for j in range(self.degree + 1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(self.degree + 1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])
            ps.append(hcat(p.coef[::-1]))
        poly = vcat(ps)
        ps_z = []
        # Construct polynomial basis for z cfr "2.5 Continuous Output for Optimal Control" from Rien's thesis
        for j in range(1, self.degree + 1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p_z = np.poly1d([1])
            for r in range(1, self.degree + 1):
                if r != j:
                    p_z *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])
            ps_z.append(hcat(p_z.coef[::-1]))
        poly_z = vcat(ps_z)
        self.q = 0

        # Make time-grid for roots
        for k in range(self.N):
            dt = (self.control_grid[k + 1] - self.control_grid[k])/self.M
            tr = []
            for i in range(self.M):
                tr.append([self.integrator_grid[k][i]+dt*self.tau[j] for j in range(self.degree)])        
            self.tr.append(tr)

        # Handle Bspline signals
        subgrid = []
        for i in range(self.M):
            subgrid+=list((i+np.array(self.tau))/self.M)

        v_sampled_store = []
        for e in self.signals.values():
            v_sampled = ca.horzsplit(e.sample(subgrid=subgrid,include_edges=False))
            v_sampled_store.append(v_sampled)
        signals_sampled = []
        for i in range(len(subgrid*self.N)):
            signals_sampled.append(ca.vertcat(*[e[i] for e in v_sampled_store]))

        dts = []
        # Fill in Z variables up-front, since they might be needed in constraints with ocp.next
        for k in range(self.N):
            dt = (self.control_grid[k + 1] - self.control_grid[k])/self.M
            dts.append(dt)
            S = 1/repmat(hcat([dt**i for i in range(self.degree + 1)]), self.degree + 1, 1)
            S_z = 1/repmat(hcat([dt**i for i in range(self.degree)]), self.degree, 1)
            self.Z.append(mtimes(self.Zc[k][0],poly_z[:,0]))
            for i in range(self.M):
                self.xk.append(self.Xc[k][i][:,0])
                self.zk.append(mtimes(self.Zc[k][i],poly_z[:,0]))
                self.poly_coeff.append(mtimes(self.Xc[k][i],poly*S))
                self.poly_coeff_z.append(mtimes(self.Zc[k][i],poly_z*S_z))

        self.Z.append(mtimes(self.Zc[-1][-1],sum2(poly_z)))

        self.xqk.append(DM.zeros(stage.nxq))
        count_f_eval = 0
        for k in range(self.N):
            dt = dts[k]
            p = self.get_p_sys(stage,k,include_signals=False)
            for i in range(self.M):
                for j in range(self.degree):
                    Pidot_j = mtimes(self.Xc[k][i],self.C[:,j])/ dt
                    p_total = vertcat(p, signals_sampled[count_f_eval])
                    res = f(x=self.Xc[k][i][:, j+1], u=self.U[k], z=self.Zc[k][i][:,j], p=p_total, t=self.tr[k][i][j])
                    count_f_eval += 1
                    # Collocation constraints
                    opti.subject_to(Pidot_j == res["ode"], scale=scale_der_x)
                    self.q = self.q + res["quad"]*dt*self.B[j]
                    if stage.nz:
                        opti.subject_to(0 == res["alg"], scale = scale_z)
                    for c, meta, args in stage._constraints["integrator_roots"]:
                        opti.subject_to(self.eval_at_integrator_root(stage, c, k, i, j), scale=args["scale"], meta=meta)

                # Continuity constraints
                x_next = self.X[k + 1] if i==self.M-1 else self.Xc[k][i+1][:,0]
                opti.subject_to(mtimes(self.Xc[k][i],self.D) == x_next, scale=scale_x)

                for c, meta, args in stage._constraints["integrator"]:
                    if k==0 and i==0 and not args["include_first"]: continue
                    opti.subject_to(self.eval_at_integrator(stage, c, k, i), scale=args["scale"], meta=meta)
                for c, meta, _ in stage._constraints["inf"]:
                    self.add_inf_constraints(stage, opti, c, k, i, meta)
                self.xqk.append(self.q)
            for c, meta, args in stage._constraints["control"]:  # for each constraint expression
                if k==0 and not args["include_first"]: continue
                # Add it to the optimizer, but first make x,u concrete.
                try:
                    opti.subject_to(self.eval_at_control(stage, c, k), scale=args["scale"], meta=meta)
                except IndexError:
                    pass # Can be caused by ocp.offset -> drop constraint
            self.Q[k+1] = self.q
        for c, meta, args in stage._constraints["control"]:  # for each constraint expression
            if not args["include_last"]: continue
            # Add it to the optimizer, but first make x,u concrete.
            try:
                opti.subject_to(self.eval_at_control(stage, c, -1), scale=args["scale"], meta=meta)
            except IndexError:
                pass # Can be caused by ocp.offset -> drop constraint

        for c, meta, args in stage._constraints["integrator"]:  # for each constraint expression
            if not args["include_last"]: continue
            # Add it to the optimizer, but first make x,u concrete.
            opti.subject_to(self.eval_at_control(stage, c, -1), scale=args["scale"], meta=meta)

    def set_initial(self, stage, master, initial):
        opti = master.opti if hasattr(master, 'opti') else master
        opti.cache_advanced()
        initial = HashOrderedDict(initial)
        algs = get_ranges_dict(stage.algebraics)
        initial_alg = HashDict()
        for a, v in list(initial.items()):
            if a in algs:
                initial_alg[a] = v
                del initial[a]
        for var, expr in initial.items():
            if ca.is_equal(var, stage.T):
                var = self.T
            if ca.is_equal(var, stage.t0):
                var = self.t0
            is_states = depends_on(var, stage.x)
            opti_initial = opti.initial()
            if is_numeric(expr):
                value = ca.evalf(expr)
                # Row vector if vector
                if value.is_column() and var.is_scalar(): value = value.T
                if is_states:
                    if var.numel()*(self.N)==value.numel() or var.numel()*(self.N+1)==value.numel():
                        value_integrator = kron(DM.ones(1,self.M),value[:,:self.N])
                        if var.numel()*(self.N+1)==value.numel(): value_integrator = horzcat(value_integrator.value[:,-1])
                        value_integrator_root = kron(DM.ones(1,self.M*self.degree),value[:,:self.N])
                    else:
                        value_integrator = repmat(value,1,self.N*self.M+1)
                        value_integrator_root = repmat(value,1,self.N*self.M*self.degree)
            else:
                if is_states:
                    NUM = vertcat(DM(range(10)).T, DM(range(10,20)).T)
                    expr_integrator = ca.hcat([self.eval_at_integrator(stage, expr, k, i) for k in list(range(self.N)) for i in range(self.M)]+[self.eval_at_control(stage, expr, -1)]) # HOT line
                    expr_integrator_root = ca.hcat([self.eval_at_integrator_root(stage, expr, k, i, j) for k in list(range(self.N)) for i in range(self.M) for j in range(self.degree) ]) # HOT line
                    value_integrator = DM(opti.debug.value(expr_integrator, opti_initial))
                    value_integrator_root = DM(opti.debug.value(expr_integrator_root, opti_initial))
                else:
                    expr = ca.hcat([self.eval_at_control(stage, expr, k) for k in list(range(self.N))+[-1]]) # HOT line
                    value = DM(opti.debug.value(expr, opti_initial))

            if is_states:
                target_integrator = ca.hcat([self.eval_at_integrator(stage, var, k, i) for k in list(range(self.N)) for i in range(self.M)]+[self.eval_at_control(stage, var, -1)])
                opti.set_initial(target_integrator, value_integrator, cache_advanced=True)
                target_integrator_root = ca.hcat([self.eval_at_integrator_root(stage, var, k, i, j) for k in list(range(self.N)) for i in range(self.M) for j in range(self.degree)]) # HOT line
                opti.set_initial(target_integrator_root, value_integrator_root, cache_advanced=True)
            else:
                # Row vector if vector
                if value.is_column() and var.is_scalar(): value = value.T
                for k in list(range(self.N))+[-1]:
                    target = self.eval_at_control(stage, var, k)
                    value_k = value
                    if target.numel()*(self.N)==value.numel() or target.numel()*(self.N+1)==value.numel():
                        value_k = value[:,k]
                    try:
                        #print(target,value_k)
                        opti.set_initial(target, value_k, cache_advanced=True)
                    except Exception as e:
                        # E.g for single shooting, set_initial of a state, for k>0
                        # Error message is usually "... arbitrary expression ..." but can also be
                        # "... You cannot set an initial value for a parameter ..."
                        # if the dynamics contains a parameter
                        if "arbitrary expression" in str(e) or (not target.is_valid_input() and "initial value for a parameter" in str(e)):
                            pass
                        else:
                            # Other type of error: 
                            raise e
        for var, expr in list(initial_alg.items()):
            opti_initial = opti.initial()
            for k in range(self.N):
                for i, e in enumerate(self.Zc[k]):
                    e_shape = e[algs[var],:].shape
                    value = DM(opti.debug.value(hcat([self.eval_at_integrator_root(stage, expr, k, i, j) for j in range(e_shape[1])]), opti_initial))                    
                    opti.set_initial(e[algs[var],:], value)

    def to_function(self, stage, name, args, results, *margs):
        args = list(args)
        add_zc = 0
        for i,e in enumerate(args):
            if isinstance(e,str) and e=="z":
                args[i] = self.Zc_vars_base
                add_zc = True
        all_args = vvcat(args)
        _,states = stage.sample(stage.x,grid='control')

        name_in = None
        if len(margs)>0 and isinstance(margs[0], list) and np.all([isinstance(e,str) for e in margs[0]]):
            name_in = list(margs[0])

        add_xc = depends_on(all_args, states) and not depends_on(all_args, self.Xc_vars)
        add_zc = add_zc and not depends_on(all_args, self.Zc_vars_rest)
        inner_args = list(args)
        if add_xc:
            inner_args += [self.Xc_vars]
            if name_in: name_in += ["Xc_vars"]
        if add_zc:
            inner_args += [self.Zc_vars_rest]
            if name_in: name_in += ["Zc_vars_rest"]

        inner_margs = list(margs)
        if name_in:
            inner_margs[0] = name_in

        f = SamplingMethod.to_function(self, stage, name, inner_args, results, *inner_margs)
        f_args = f.mx_in()[:len(args)]
        call_args = list(f_args)
        if add_xc:
            call_args+=[self.Xc_vars0]
        if add_zc:
            call_args+=[self.Zc0]

        return Function(name, f_args, f.call(call_args,True,False), *margs)



#first, find a rockit way  to initialize xc and zc using to_function
#parameter?
#parametric set_initial?