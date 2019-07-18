Some recommendations for a productive setup:

# Python environment

* Install https://docs.conda.io/en/latest/miniconda.html
  This allows you to create an isolated Python environment.

* `conda create --name ocpx python=3.7 matplotlib scipy ipython pylint`
* `conda activate ocpx`

# CasADi setup
* In your Python environment, do `pip install casadi`

# IDE

 * Install https://code.visualstudio.com/
 * Install the Python extension.
 * Open the `ocpx` cloned folder, navigate to the `hello_world.py` example.
 * A bar at the bottom of the screen should mention the Python environment,
  make sure to select 'ocpx' environment.
 * Add the `ocpx` cloned folder into PYTHONPATH with `export PYTHONPATH=$PYTHONPATH:/path/to/repo`
 * Right-click the open file, select 'Run Python file in terminal'.