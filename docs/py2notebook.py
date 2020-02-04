import sys
import json

if len(sys.argv)<=2:
  out = sys.argv[1][:-2]+"ipynb"
else:
  out = sys.argv[2]

print(out)

with open(sys.argv[1],"r") as f_in:
  data = json.load(open("docs/template.ipynb", "r"))
  
  for l in f_in:
    if not l.startswith("#"):
      break

  preamble = []
  for l in f_in:
    if not l.startswith('"""'):
      break
  
  for l in f_in:
    preamble.append(l)
    if l.startswith('"""'): break

  data["cells"][-4]["source"] = preamble
  data["cells"][-1]["source"] = f_in.readlines()
  json.dump(data, open(out, "w"),indent=True)


