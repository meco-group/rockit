"""Dummy classes to explicilty define options for sampling ocp solutions.

Internal code, should never be imported by the user.
The Stage class uses these to tell the OcpxSolution.sample
how to sample the solution like this:
t_s, x1_s = sol.sample(stage, x1, grid=stage.grid_control)
"""


class GridControl:
    def __init__(self):
        pass

class GridIntegrator:
    def __init__(self):
        pass

class GridIntegratorFine:
    def __init__(self):
        pass
