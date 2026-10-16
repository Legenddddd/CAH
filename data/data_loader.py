import torch
import os

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFile

import data.cub_2011 as cub2011
import data.nabirds as nabirds
import data.aircraft as aircraft
import data.food101 as food101
import data.vegfru as vegfru
from data.transform import train_transform, encode_onehot

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(dataset, root, num_query, num_train, batch_size, num_workers):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
        num_query(int): Number of query data points.
        num_train(int): Number of training data points.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
    """
    if dataset == 'cub':
        query_dataloader, train_dataloader, retrieval_dataloader, num_query, num_train = cub2011.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    elif dataset == 'food101':
        query_dataloader, train_dataloader, retrieval_dataloader, num_query, num_train = food101.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    elif dataset == 'nabirds':
        query_dataloader, train_dataloader, retrieval_dataloader, num_query, num_train = nabirds.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    elif dataset == 'air':
        query_dataloader, train_dataloader, retrieval_dataloader, num_query, num_train = aircraft.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    elif dataset == 'vegfru':
        query_dataloader, train_dataloader, retrieval_dataloader, num_query, num_train = vegfru.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    else:
        raise ValueError("Invalid dataset.")

    return query_dataloader, train_dataloader, retrieval_dataloader, num_query, num_train