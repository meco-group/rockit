from casadi import *
import casadi as cs
import numpy as np

def eval_basis_knotindex(ind, knots, d):
    if knots.is_row():
      knots = knots.T
    # [first]*d+precursor+[last]*d
    basis = [0.0]*knots.numel()
    if ind==0:
      for j in range(d+1):
        basis[j] = 1.0+np.finfo(np.float64).eps # Workaround #2913
    else:
      basis[min(ind+d,knots.numel()-d-2)] = 1.0+np.finfo(np.float64).eps # Workaround #2913
    basis = sparsify(vcat(basis))
    x = knots[ind+d]
    for e in range(1, d + 1):
        i = DM(list(range(d-e+1,knots.numel() - d - 1)))
        L = knots.numel()-2*d-2+e
        ki = knots[i]
        kid = knots[i + e]
        norm = basis[i] / (kid - ki)
        dbg_ref = (x - ki) * norm
        dbg_ref2 = (kid - x) * norm
        basis = MX(knots.numel() - e - 1, 1)
        basis[d-e+1:d-e+1+L] += dbg_ref
        basis[d-e:d-e+L] += dbg_ref2
    return basis


def eval_basis_knotindex_subsampled(ind, N, knots, d):
    if knots.is_row():
      knots = knots.T
    basis = [0.0]*knots.numel()
    basis[min(ind+d,knots.numel()-d-2)] = 1.0+np.finfo(np.float64).eps # Workaround #2913
    basis = vcat(basis)
    basis = sparsify(repmat(basis,1,N))
    tau = cs.linspace(DM(0),1,N+2)[1:-1].T
    x = knots[ind+d]*(1-tau)+tau*knots[ind+d+1]
    for e in range(1, d + 1):
        i = DM(list(range(d-e+1,knots.numel() - d - 1)))
        L = knots.numel()-2*d-2+e
        ki = knots[i]
        kid = knots[i + e]
        norm = basis[i,:] / (kid - ki)

        xr = repmat(x, L ,1)
        dbg_ref = (xr - ki) * norm
        dbg_ref2 = (kid - xr) * norm
        basis = MX(knots.numel() - e - 1, N)
        basis[d-e+1:d-e+1+L,:] += dbg_ref
        basis[d-e:d-e+L,:] += dbg_ref2
          
    return basis

def eval_on_knots(xi,d,subsamples=0):
  knots = horzcat(repmat(xi[0],1,d),xi,repmat(xi[-1],1,d))
  basis = []
  k = []
  tau = cs.linspace(DM(0),1,subsamples+2)[1:-1].T
  for i in range(knots.numel()-2*d):
    basis.append(eval_basis_knotindex(i, knots, d ))
    k.append(xi[i])
    if subsamples>0 and i<knots.numel()-2*d-1:
      k_current = xi[i]
      k_next = xi[i+1]
      k.append(k_current*(1-tau)+tau*k_next)
      basis.append(eval_basis_knotindex_subsampled(i, subsamples, knots, d))
  basis = hcat(basis)
  k = hcat(k)
  try:
      basis = evalf(basis)
  except:
      pass
  try:
      k = evalf(k)
  except:
      pass
  if d==0:
    basis = basis[:-1,:]
  return k,basis

def bspline_derivative(c,xi,d):
  delta_xi = horzcat(xi[:,1:],repmat(xi[-1],1,d-1))-horzcat(repmat(xi[0],1,d-1),xi[:,:-1])
  scale = d/delta_xi
  return repmat(scale,c.shape[0],1)*(c[:,1:]-c[:,:-1])


def get_greville_points(xi,d):
    if d==0:
      return (xi[0,1:]+xi[0,:-1])/2
    # Greville: moving average of xi
    N = xi.shape[1]
    s = N-1+d

    import itertools
    row = []
    for i in range(N):
      row.extend(range(i,i+d))
    colind = np.array(range(N+1))*d

    source = (cs.DM(range(d,0,-1))/(cs.DM.ones(d,1)*d)).nonzeros()
    values  = source+[1/d]*(N*d-2*len(source))+source[::-1]
    S = cs.DM(cs.Sparsity(s,N,colind,row), values).T
    np.testing.assert_allclose(cs.sum1(S), 1, atol=1e-12)
    return xi @ S
  
if __name__ == "__main__":
    d = 3
    n = 10
    t = DM(np.linspace(0,1,n)).T
    print(eval_on_knots(t,d))       # works
    for dd in range(d+1):
      print(d-dd, [e.shape for e in eval_on_knots(t,d-dd,subsamples=2)])
      eval_on_knots(t,d-dd,subsamples=2)[1].sparsity().spy()

