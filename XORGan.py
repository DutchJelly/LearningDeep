# This is from a tutorial: https://towardsdatascience.com/build-a-super-simple-gan-in-pytorch-54ba349920e4
from typing import List, Tuple
import math
import numpy as np

import torch
import torch.nn as nn

def translateToBinaryVector(x: int) -> List[int]:
    
    if(x < 0 or type(x) is not int):
        raise ValueError('the supplied parameter must be positive and of type int')
    return [int(y) for y in bin(x)[2:]]

def generateEvenDataset(max_int: int, batch_size: int=16) -> Tuple[List[int], List[List[int]]]:
    # Get the number of binary places needed to represent the maximum number
    max_length = int(math.log(max_int, 2))

    # Sample batch_size number of integers in range 0-max_int
    sampled_integers = np.random.randint(0, int(max_int / 2), batch_size)

    # create a list of labels all ones because all numbers are even
    labels = [1] * batch_size

    # Generate a list of binary numbers for training.
    data = [translateToBinaryVector(int(x * 2)) for x in sampled_integers]
    data = [([0] * (max_length - len(x))) + x for x in data]

    return labels, data


