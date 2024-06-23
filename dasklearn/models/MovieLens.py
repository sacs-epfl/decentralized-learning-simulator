import torch

from dasklearn.models import Model


class MatrixFactorization(Model):
    """
    Class for a Matrix Factorization model for MovieLens.
    """

    def __init__(self, n_users=610, n_items=9724, n_factors=20):
        """
        Instantiates the Matrix Factorization model with user and item embeddings.

        Parameters
        ----------
        n_users
            The number of unique users.
        n_items
            The number of unique items.
        n_factors
            The number of columns in embeddings matrix.
        """
        super().__init__()
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # logging.info("Device: {}".format(self.device))
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.user_factors.weight.data.uniform_(-0.05, 0.05)
        self.item_factors.weight.data.uniform_(-0.05, 0.05)

    def forward(self, data):
        """
        Forward pass of the model, it does matrix multiplication and returns predictions for given users and items.
        """
        users = data[:, 0].to(torch.long) - 1
        items = data[:, 1].to(torch.long) - 1
        u, it = self.user_factors.to(self.device)(users), self.item_factors.to(
            self.device
        )(items)
        x = (u * it).sum(dim=1, keepdim=True)
        return x.squeeze(1)
