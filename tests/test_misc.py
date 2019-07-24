import unittest

from ocpx import *

class MiscTests(unittest.TestCase):

    def test_stage_constr(self):
      ocp = OcpMultiStage()
      stage = ocp.stage(t0=0,T=10)

if __name__ == '__main__':
    unittest.main()
