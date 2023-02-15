import subprocess

import os
import sys
import fnmatch

my_env = os.environ.copy()
my_env["MPLBACKEND"] = "Agg"

def run(*args, **kwargs):
    print(args)
    pid = subprocess.Popen(*args, **kwargs)
    pid.wait()
    if pid.returncode!=0:
      sys.exit(pid.returncode)

# Run main examples
for root, dir, files in os.walk("examples"):
        if "wip" in root: continue
        for f in files:
          if f.endswith(".py"):
            run([sys.executable, f], cwd=root,env=my_env)

# Run external examples
for root, dir, files in os.walk("rockit/external"):
  if root.endswith("examples"):
    for f in files:
      if "wip" in root: continue
      if f.endswith(".py"):
        run([sys.executable, f], cwd=root,env=my_env)

run(["nosetests"])

