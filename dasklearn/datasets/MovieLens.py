import logging
import math
import os
import zipfile

import pandas as pd
import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dasklearn.datasets import Dataset
from dasklearn.datasets.data import Data
from dasklearn.mappings import Mapping


class MovieLens(Dataset):
    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        random_seed: int = 1234,
        train_dir="",
        test_dir="",
        sizes="",
        test_batch_size=1,
    ):
        """
        Constructor which reads the data files, instantiates and partitions the dataset

        Parameters
        ----------
        rank : int
            Rank of the current process (to get the partition).
        machine_id : int
            Machine ID
        mapping : decentralizepy.mappings.Mapping
            Mapping to convert rank, machine_id -> uid for data partitioning
            It also provides the total number of global processes
        random_seed : int, optional
            Random seed for the dataset
        only_local : bool, optional
            True if the dataset needs to be partioned only among local procs, False otherwise
        train_dir : str, optional
            Path to the training data files. Required to instantiate the training set
            The training set is partitioned according to the number of global processes and sizes
        test_dir : str. optional
            Path to the testing data files Required to instantiate the testing set
        sizes : list(int), optional
            A list of fractions specifying how much data to alot each process. Sum of fractions should be 1.0
            By default, each process gets an equal amount.
        test_batch_size : int, optional
            Batch size during testing. Default value is 64

        """
        super().__init__(
            rank,
            machine_id,
            mapping,
            train_dir,
            test_dir,
            sizes,
            test_batch_size,
        )

        self.random_seed = random_seed

        # Download the dataset
        self.train_dir = self.train_dir or self.test_dir
        zip_file = os.path.join(self.train_dir, "ml-latest-small.zip")
        if not os.path.isfile(zip_file):
            download_movie_lens(self.train_dir)

        self.n_users, self.n_items, df_train, df_test = self._load_data()
        self.train_data, self.test_data = self._split_data(
            df_train, df_test, self.n_procs
        )

        # [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        # [0,   1,   2,   3,   4,   5,   6,   7,   8,   9  ]
        self.NUM_CLASSES = 10
        self.RATING_DICT = {
            0.5: 0,
            1.0: 1,
            1.5: 2,
            2.0: 3,
            2.5: 4,
            3.0: 5,
            3.5: 6,
            4.0: 7,
            4.5: 8,
            5.0: 9,
        }

    def _load_data(self):
        f_ratings = os.path.join(self.train_dir, "ml-latest-small", "ratings.csv")
        names = ["user_id", "item_id", "rating", "timestamp"]
        df_ratings = pd.read_csv(f_ratings, sep=",", names=names, skiprows=1).drop(
            columns=["timestamp"]
        )
        # map item_id properly
        items_count = df_ratings["item_id"].nunique()
        items_ids = sorted(list(df_ratings["item_id"].unique()))
        id_map = {item: i+1 for i, item in enumerate(items_ids)}
        df_ratings["item_id"] = df_ratings["item_id"].replace(id_map)

        # split train, test - 70% : 30%
        grouped_users = df_ratings.groupby(["user_id"])
        users_count = len(grouped_users)
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        for i in range(0, users_count):
            df_user = df_ratings[df_ratings["user_id"] == i + 1]
            df_user_train = df_user.sample(frac=0.7, random_state=self.random_seed)
            df_user_test = pd.concat([df_user, df_user_train]).drop_duplicates(
                keep=False
            )
            assert len(df_user_train) + len(df_user_test) == len(df_user)

            df_train = pd.concat([df_train, df_user_train])
            df_test = pd.concat([df_test, df_user_test])

        # 610, 9724
        return users_count, items_count, df_train, df_test

    def _split_data(self, train_data, test_data, world_size):
        # SPLITTING BY USERS: group by users and split the data accordingly
        mod = self.n_users % world_size
        users_count = self.n_users // world_size
        if self.uid < mod:
            users_count += 1
            offset = users_count * self.uid
        else:
            offset = users_count * self.uid + mod

        my_train_data = pd.DataFrame()
        my_test_data = pd.DataFrame()
        for i in range(offset, offset + users_count):
            my_train_data = pd.concat(
                [my_train_data, train_data[train_data["user_id"] == i + 1]]
            )
            my_test_data = pd.concat(
                [my_test_data, test_data[test_data["user_id"] == i + 1]]
            )

        logging.info("Data split for test and train.")
        return my_train_data, my_test_data

    def get_trainset(self, batch_size=1, shuffle=False):
        if self.__training__:
            train_x = self.train_data[["user_id", "item_id"]].to_numpy()
            train_y = self.train_data.rating.values.astype("float32")
            return DataLoader(
                Data(train_x, train_y), batch_size=batch_size, shuffle=shuffle
            )
        raise RuntimeError("Training set not initialized!")

    def get_testset(self):
        if self.__testing__:
            test_x = self.test_data[["user_id", "item_id"]].to_numpy()
            test_y = self.test_data.rating.values
            return DataLoader(Data(test_x, test_y), batch_size=self.test_batch_size)
        raise RuntimeError("Test set not initialized!")

    def test(self, model, device_name: str = "cpu"):
        """
        Evaluate the model on the test set.

        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate.
        loss : torch.nn.loss
            Loss function to evaluate

        Returns
        -------
        tuple(float, float)

        """
        device = torch.device(device_name)
        model.to(device)
        model.eval()
        test_set = self.get_testset()
        loss = nn.MSELoss()
        logging.debug("Test Loader instantiated.")

        correct_pred = [0 for _ in range(self.NUM_CLASSES)]
        total_pred = [0 for _ in range(self.NUM_CLASSES)]
        total_correct = 0
        total_predicted = 0

        with torch.no_grad():
            loss_val = 0.0
            count = 0

            for test_x, test_y in test_set:
                test_x, test_y = test_x.to(device), test_y.to(device)

                output = model(test_x)
                loss_val += loss(output, test_y).item()
                count += 1

                # threshold values to range [0.5, 5.0]
                o1 = (output > 5.0).nonzero(as_tuple=False)
                output[o1] = 5.0
                o1 = (output < 0.5).nonzero(as_tuple=False)
                output[o1] = 0.5
                # round a number to the closest half integer
                output = torch.round(output * 2) / 2

                for rating, prediction in zip(test_y.tolist(), output):
                    # print(rating, prediction)
                    logging.debug("{} predicted as {}".format(rating, prediction))
                    if rating == prediction:
                        correct_pred[self.RATING_DICT[rating]] += 1
                        total_correct += 1
                    total_pred[self.RATING_DICT[rating]] += 1
                    total_predicted += 1

        logging.debug("Predicted on the test set")
        for key, value in enumerate(correct_pred):
            if total_pred[key] != 0:
                accuracy = 100 * float(value) / total_pred[key]
            else:
                accuracy = 100.0
            logging.debug("Accuracy for class {} is: {:.1f} %".format(key, accuracy))

        accuracy = 100 * float(total_correct) / total_predicted
        loss_val = math.sqrt(loss_val / count)
        logging.info("Overall accuracy is: {:.1f} %".format(accuracy))
        return accuracy, loss_val


def download_movie_lens(dest_path):
    """
    Downloads the movielens latest small dataset.
    This data set consists of:
        * 100836 ratings from 610 users on 9742 movies.
        * Each user has rated at least 20 movies.

    https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html
    """
    url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    req = requests.get(url, stream=True)

    os.makedirs(dest_path, exist_ok=True)
    logging.info("Downloading MovieLens Latest Small data...")
    with open(os.path.join(dest_path, "ml-latest-small.zip"), "wb") as fd:
        for chunk in req.iter_content(chunk_size=None):
            fd.write(chunk)
    with zipfile.ZipFile(os.path.join(dest_path, "ml-latest-small.zip"), "r") as z:
        z.extractall(dest_path)
    logging.info("Downloaded MovieLens Latest Small dataset at %s" % dest_path)
