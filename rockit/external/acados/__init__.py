try:
    import acados_template
except:
    raise Exception("'import acados_template' failed.\n"
                    "Check https://docs.acados.org/interfaces/index.html#python-templates for install instructions.")

from .method import AcadosMethod

method = AcadosMethod