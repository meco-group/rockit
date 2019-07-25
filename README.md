# OCPx
[![pipeline status](https://gitlab.mech.kuleuven.be/meco-software/ocpx/badges/master/pipeline.svg)](https://gitlab.mech.kuleuven.be/meco-software/ocpx/commits/master)
[![coverage report](https://gitlab.mech.kuleuven.be/meco-software/ocpx/badges/master/coverage.svg)](https://meco-software.pages.mech.kuleuven.be/ocpx/coverage/index.html)
[![html docs](https://img.shields.io/static/v1.svg?label=docs&message=online&color=informational)](http://meco-software.pages.mech.kuleuven.be/ocpx)
[![pdf docs](https://img.shields.io/static/v1.svg?label=docs&message=pdf&color=red)](http://meco-software.pages.mech.kuleuven.be/ocpx/documentation-ocpx.pdf)

# Get started
Some recommendations for a productive setup:

### Python environment

* Install https://docs.conda.io/en/latest/miniconda.html
  This allows you to create an isolated Python environment.

* `conda create --name ocpx python=3.6 matplotlib scipy ipython pylint`
* `conda activate ocpx`

### CasADi setup
* In your Python environment, do `pip install casadi`

### IDE

 * Install https://code.visualstudio.com/
 * Install the Python extension.
 * Open the `ocpx` cloned folder, navigate to the `hello_world.py` example.
 * A bar at the bottom of the screen should mention the Python environment,
  make sure to select 'ocpx' environment.
 * Add the `ocpx` cloned folder into PYTHONPATH with `export PYTHONPATH=$PYTHONPATH:/path/to/repo` (linux) or `set PYTHONPATH=%PYTHONPATH%;/path/to/repo` (windows)
 * Right-click the open file, select 'Run Python file in terminal'.

 ### Documentation

 Use the [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).
 For example:
 ```python
def sample(self, stage, expr, grid):
    """Sample expression at solution on a given grid.

    Parameters
    ----------
    stage : :obj:`~ocpx.stage.Stage`
        An optimal control problem stage.
    expr : :obj:`casadi.MX`
        Arbitrary expression containing states, controls, ...
    grid : :obj:`~ocpx.stage_options.GridOption`
        Type of time grid to use for sampling,
        options are available in ocpx.stage_options.

    Returns
    -------
    time : numpy.ndarray
        Time from zero to final time, same length as res
    res : numpy.ndarray
        Numerical values of evaluated expression at points in time vector.

    Examples
    --------
    Assume an ocp with a stage is already defined.

    >>> sol = ocp.solve()
    >>> tx, xs = sol.sample(stage, x, grid='control')
    """
```
