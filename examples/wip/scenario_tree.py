ocp = Ocp()

stage = ocp.stage()

x1 = stage.state()
x2 = stage.state()

stage.set_der(x1, x2)
stage.set_der(x2, -0.1 * (1 - x1**2 + delta) * x2 - x1 + u + delta)
