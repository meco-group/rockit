# rockit
[![pipeline status](https://gitlab.mech.kuleuven.be/meco-software/rockit/badges/master/pipeline.svg)](https://gitlab.mech.kuleuven.be/meco-software/rockit/commits/master)
[![coverage report](https://gitlab.mech.kuleuven.be/meco-software/rockit/badges/master/coverage.svg)](https://meco-software.pages.mech.kuleuven.be/rockit/coverage/index.html)
[![html docs](https://img.shields.io/static/v1.svg?label=docs&message=online&color=informational)](http://meco-software.pages.mech.kuleuven.be/rockit)
[![pdf docs](https://img.shields.io/static/v1.svg?label=docs&message=pdf&color=red)](http://meco-software.pages.mech.kuleuven.be/rockit/documentation-rockit.pdf)

# Description

![Rockit logo](docs/logo.png)

Rockit (Rapid Optimal Control kit) is a software framework to quickly prototype optimal control problems (aka dynamic optimization) that may arise in engineering:
iterative learning (ILC), model predictive control (NMPC), motion planning.

Notably, the software allows free end-time problems and multi-stage optimal problems.
The software is currently focused on direct methods and relies heavily on [CasADi](http://casadi.org).
The software is developed by the [KU Leuven MECO research team](https://www.mech.kuleuven.be/en/pma/research/meco).

# Installation
Install using pip: `pip install rockit-meco`

# Hello world
(Taken from the [example gallery](https://meco-software.pages.mech.kuleuven.be/rockit/examples/))

You may try it live in your browser: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fgitlab.mech.kuleuven.be%2Fmeco-software%2Frockit.git/master?filepath=examples%2Fhello_world.ipynb).

Import the project:
```python
from rockit import *
```

Start an optimal control environment with a time horizon of 10 seconds (free time problems can be configured with `FreeTime(initial_guess)`):
```python
ocp = Ocp(T=10)
```

Define two scalar states (vectors and matrices also supported):
```python
x1 = ocp.state()
x2 = ocp.state()
```

Define one piece-wise constant control input (use `order=1` for piecewise linear):
```python
u = ocp.control()
```

Specify differential equations for states (time dependency supported with `ocp.t`, DAEs also supported with `ocp.algebraic` and `add_alg`):
```python
ocp.set_der(x1, (1 - x2**2) * x1 - x2 + u)
ocp.set_der(x2, x1)
```

Lagrange objective term:
```python
ocp.add_objective(ocp.integral(x1**2 + x2**2 + u**2))
```

```Mayer objective term
ocp.add_objective(ocp.at_tf(x1**2))
```

Path constraints (must be valid on the whole time domain running from `t0` to `tf=t0+T`, grid options available):
```python
ocp.subject_to(x1 >= -0.25)
ocp.subject_to(-1 <= (u <= 1 ))
```

Boundary constraints:
```python
ocp.subject_to(ocp.at_t0(x1) == 0)
ocp.subject_to(ocp.at_t0(x2) == 1)
```

Pick an NLP solver backend (CasADi `nlpsol` plugin):
```python
ocp.solver('ipopt')
```

Pick a solution method:
  N -- number of control intervals
  M -- number of integration steps per control interval
```python
method = MultipleShooting(N=10, M=2, intg='rk')
#method = DirectCollocation(N=10, M=2)
ocp.method(method)
```

Solve:
```python
sol = ocp.solve()
```

Show structure:
```python
ocp.spy()
```

![Structure of optimization problem](docs/hello_world_structure.png)

Post-processing:
```
tsa, x1a = sol.sample(x1, grid='control')
tsb, x1b = sol.sample(x1, grid='integrator')
tsc, x1c = sol.sample(x1, grid='integrator', refine=100)
plot(tsa, x1a, '-')
plot(tsb, x1b, 'o')
plot(tsc, x1c, '.')
```

![Solution trajectory of states](docs/hello_world_states.png)
