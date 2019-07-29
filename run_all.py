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

for root, dir, files in os.walk("examples"):
        if "wip" in root: continue
        for f in files:
          if f.endswith(".py"):
            run([sys.executable, os.path.join(root, f)], env=my_env)

run(["nosetests"])

