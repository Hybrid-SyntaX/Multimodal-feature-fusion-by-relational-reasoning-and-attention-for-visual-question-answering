from typing import Sequence, List

import torch
# from ..GQAFeatures import GQAFeatures
# import os
# from glob import glob
from torch import Tensor
from torch._C import Generator
from torch.types import _int
from torch.utils.data import random_split, TensorDataset, Dataset, Subset

#
from utils import nonrandom_split


def tests_nonrandom_split():
    dataset=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # dataset = TensorDataset(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #                         , torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    ds_length=len(dataset)
    #[0.7, 0.15, 0.15]
    lengths = [int(ds_length * 0.5), int(ds_length * 0.25) + 1, int(ds_length * 0.25) + 1]
    train_data, valid_data, test_data =nonrandom_split(dataset,lengths)
    #train_data, valid_data, test_data =torch.split(dataset, lengths)

    assert len(train_data)==5
    assert len(valid_data)==3
    assert  len(test_data)==3
    assert list(train_data)==[0,1,2,3,4]
    assert list(valid_data) == [5,6,7]
    assert list(test_data) == [8,9,10]

# class StaticGenerator(torch.Generator):
#     def get_state(self) -> Tensor:
#         return super().get_state()
#
#     def set_state(self, _new_state: Tensor) -> Generator:
#         return super().set_state(_new_state)
#
#     def manual_seed(self, seed: _int) -> Generator:
#         return super().manual_seed(seed)
#
#     def seed(self) -> _int:
#         return super().seed()
#
#     def initial_seed(self) -> _int:
#         return super().initial_seed()
#
#     def __init__(self):
#         super(StaticGenerator,self).__init__()

def tests_random_split_splits_properly():
    dataset=[0,1,2,3,4,5, 6,7,8,9,10]
    # dataset=TensorDataset(torch.tensor([0,1,2,3,4,5, 6,7,8,9,10])
    #                       ,torch.tensor([0,0,0,0,0,0,0,0,0,0,0]))


    train_data=dataset[0:5]
    valid_data= dataset[5:8]
    test_data = dataset[8:]
    # train_data, valid_data, test_data = random_split(dataset = dataset, lengths = [5,3,3],
    #                                                  generator=torch.Generator().manual_seed(0))

    assert len(train_data)==5
    assert len(valid_data)==3
    assert  len(test_data)==3
    assert list(train_data)==[0,1,2,3,4]
    assert list(valid_data) == [5,6,7]
    assert list(test_data) == [8,9,10]
# def test_gqa_spatial_featureset_constructed_success():
#     spatial_loc=f'E:\Datasets\GQA\spatial'
#     spatial_h5=glob(os.path.join(spatial_loc, 'gqa_spatial_*.h5'))
#     gqa_spatial_featureset = GQAFeatures(spatial_h5[0])
#     batch_size = 100
#     train_loader = torch.utils.data.DataLoader(gqa_spatial_featureset,
#                                            batch_size = batch_size,
#                                           shuffle = False)
#
#     data_iter = iter(train_loader)
#     features = data_iter.next()
#     assert features.shape==(100, 2048, 7, 7)
