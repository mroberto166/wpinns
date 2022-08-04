import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader


class DefineDataset:
    def __init__(self,
                 Ec,
                 n_collocation,
                 n_boundary,
                 n_initial,
                 n_internal,
                 batches,
                 random_seed,
                 shuffle=False,
                 ):

        self.Ec = Ec
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.n_initial = n_initial
        self.n_internal = n_internal
        self.batches = batches
        self.random_seed = random_seed
        self.shuffle = shuffle

        self.space_dimensions = self.Ec.space_dimensions
        self.time_dimensions = self.Ec.time_dimensions
        self.input_dimensions = self.Ec.space_dimensions + self.Ec.time_dimensions
        self.output_dimension = self.Ec.output_dimension
        self.n_samples = self.n_collocation + 2 * self.n_boundary * self.space_dimensions + self.n_initial * self.time_dimensions + self.n_internal
        self.BC = None
        self.data_coll = None
        self.data_boundary = None
        self.data_initial_internal = None

        if self.batches == "full":
            self.batches = int(self.n_samples)
        else:
            self.batches = int(self.batches)

    def assemble_dataset(self):

        fraction_coll = int(self.batches * self.n_collocation / self.n_samples)
        fraction_boundary = int(self.batches * 2 * self.n_boundary * self.space_dimensions / self.n_samples)
        fraction_initial = int(self.batches * self.n_initial / self.n_samples)
        fraction_internal = int(self.batches * self.n_internal / self.n_samples)

        x_coll, y_coll = self.Ec.add_collocation_points(self.n_collocation, self.random_seed)
        x_b, y_b = self.Ec.add_boundary_points(self.n_boundary, self.random_seed)

        if self.n_initial == 0:
            x_time_internal = torch.zeros((self.n_initial, self.input_dimensions))
            y_time_internal = torch.zeros((self.n_initial, self.output_dimension))
        else:
            x_time_internal, y_time_internal = self.Ec.add_initial_points(self.n_initial, self.random_seed)

        if self.n_internal != 0:
            x_internal, y_internal = self.Ec.add_internal_points(self.n_internal, self.random_seed)
            x_time_internal = torch.cat([x_time_internal, x_internal])
            y_time_internal = torch.cat([y_time_internal, y_internal])

        # print("###################################")
        # print(x_coll, x_coll.shape, y_coll.shape)
        # print(x_time_internal, x_time_internal.shape, y_time_internal.shape)
        # print(x_b, x_b.shape, y_b.shape)
        # print("###################################")

        # print(fraction_coll, fraction_initial, fraction_internal, fraction_boundary)

        if self.n_collocation == 0:
            self.data_coll = DataLoader(torch.utils.data.TensorDataset(x_coll, y_coll), batch_size=1, shuffle=False)
        else:
            self.data_coll = DataLoader(torch.utils.data.TensorDataset(x_coll, y_coll), batch_size=fraction_coll, shuffle=self.shuffle)

        if self.n_boundary == 0:
            self.data_boundary = DataLoader(torch.utils.data.TensorDataset(x_b, y_b), batch_size=1, shuffle=False)
        else:
            self.data_boundary = DataLoader(torch.utils.data.TensorDataset(x_b, y_b), batch_size=fraction_boundary, shuffle=self.shuffle)

        if fraction_internal == 0 and fraction_initial == 0:
            self.data_initial_internal = DataLoader(torch.utils.data.TensorDataset(x_time_internal, y_time_internal), batch_size=1, shuffle=False)
        else:
            self.data_initial_internal = DataLoader(torch.utils.data.TensorDataset(x_time_internal, y_time_internal), batch_size=fraction_initial + fraction_internal,
                                                    shuffle=self.shuffle)
