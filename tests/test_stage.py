import unittest

from rockit import Ocp, DirectMethod, MultipleShooting, FreeTime, Stage
import numpy as np

class StageTests(unittest.TestCase):

    def test_stage_cloning_t0_T(self):
        for t0_stage, t0_sol_stage in [(None, 0), (-1, -1), (FreeTime(-1), -1)]:
            for T_stage, T_sol_stage in [(None, 2), (2, 2), (FreeTime(1), 2)]:
                kwargs = {}
                if t0_stage is not None:
                    kwargs["t0"] = t0_stage
                if T_stage is not None:
                    kwargs["T"] = T_stage
                stage = Stage(**kwargs)

                p = stage.state()
                v = stage.state()
                u = stage.control()

                stage.set_der(p, v)
                stage.set_der(v, u)

                stage.subject_to(u <= 1)
                stage.subject_to(-1 <= u)

                stage.add_objective(stage.tf)
                stage.subject_to(stage.at_t0(p) == 0)
                stage.subject_to(stage.at_t0(v) == 0)
                stage.subject_to(stage.at_tf(p) == 1)
                stage.subject_to(stage.at_tf(v) == 0)
                stage.method(MultipleShooting(N=2))

                for t0, t0_sol in ([] if t0_stage is None else [(None, t0_sol_stage)]) + [(-1, -1), (FreeTime(-1), -1)]:
                    for T, T_sol in ([] if T_stage is None else [(None, T_sol_stage)]) + [(2, 2), (FreeTime(1), 2)]:
                        ocp = Ocp()

                        kwargs = {}
                        if t0 is not None:
                            kwargs["t0"] = t0
                        if T is not None:
                            kwargs["T"] = T
                        mystage = ocp.stage(stage, **kwargs)

                        if mystage.is_free_starttime():
                            ocp.subject_to(mystage.t0 >= t0_sol)

                        ocp.solver('ipopt')

                        sol = ocp.solve()

                        tolerance = 1e-6

                        ts, ps = sol(mystage).sample(p, grid='integrator', refine=10)

                        ps_ref = np.hstack(((0.5*np.linspace(0,1, 10+1)**2)[:-1],np.linspace(0.5,1.5,10+1)-0.5*np.linspace(0,1, 10+1)**2)) 
                        np.testing.assert_allclose(ps, ps_ref, atol=tolerance)

                        ts_ref = t0_sol + np.linspace(0, 2, 10*2+1)

                        ts, vs = sol(mystage).sample(v, grid='integrator', refine=10)
                        np.testing.assert_allclose(ts, ts_ref, atol=tolerance)

                        vs_ref = np.hstack((np.linspace(0,1, 10+1)[:-1],np.linspace(1,0, 10+1))) 
                        np.testing.assert_allclose(vs, vs_ref, atol=tolerance)


                        u_ref = np.array([1.0]*10+[-1.0]*11)
                        ts, us = sol(mystage).sample(u, grid='integrator', refine=10)
                        np.testing.assert_allclose(us, u_ref, atol=tolerance)

if __name__ == '__main__':
    unittest.main()
