class DirichletBC:
    def __init__(self):
        pass

    def apply(self, model, x_boundary, u_boundary, n_out):
        u_boundary_pred = model(x_boundary)[:, n_out]

        return u_boundary_pred, u_boundary[:, n_out]
