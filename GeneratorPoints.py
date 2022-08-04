import torch
import numpy as np
import sobol_seq


def generator_points(samples, dim, random_seed, type_of_points, boundary):
    if type_of_points == "random":
        torch.random.manual_seed(random_seed)
        return torch.rand([samples, dim]).type(torch.FloatTensor)
    elif type_of_points == "gauss":
        if samples != 0:

            x, _ = np.polynomial.legendre.leggauss(samples)
            x = 0.5 * (x.reshape(-1, 1) + 1)

            if dim == 1:
                return torch.from_numpy(x).type(torch.FloatTensor)
            if dim == 2:
                x = x.reshape(-1, )
                x = np.transpose([np.repeat(x, len(x)), np.tile(x, len(x))])
                return torch.from_numpy(x).type(torch.FloatTensor)
        else:
            return torch.zeros([0, dim])
    elif type_of_points == "grid":
        if samples != 0:
            samples_per_dim = int(samples ** (1 / dim))

            if dim == 1:
                x = np.linspace(0, 1, samples_per_dim + 2)
                x = x[1:-1]
                return torch.from_numpy(x).type(torch.FloatTensor)
            if dim == 2:
                if not boundary:
                    x = np.linspace(0, 1, samples_per_dim + 2)
                    x = x[1:-1]
                    x2 = np.copy(x)
                    inputs = np.transpose([np.tile(x, len(x2)), np.repeat(x2, len(x))])
                else:
                    x = np.linspace(0, 1, samples)
                    inputs = np.concatenate([x.reshape(-1, 1), x.reshape(-1, 1)], 1)
                print(inputs)
                print(inputs.shape)
                return torch.from_numpy(inputs).type(torch.FloatTensor)
        else:
            return torch.zeros([0, dim])

    elif type_of_points == "sobol":
        skip = random_seed
        data = np.full((samples, dim), np.nan)
        for j in range(samples):
            seed = j + skip
            data[j, :], next_seed = sobol_seq.i4_sobol(dim, seed)
        return torch.from_numpy(data).type(torch.FloatTensor)
