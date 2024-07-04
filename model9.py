import json
import tensorflow as tf
import numpy as np
import math
import random


def positionalEncoding(length, depth):
    ret = []
    for i in range(length):
        r = []
        ret.append(r)
        for j in range(depth):
            r.append(
                math.sin(i / 10000 ** (j / depth))
                if j % 2
                else math.cos(i / 10000 ** (j / depth))
            )
    return tf.constant(ret)


def useRecursiveBERT(
    dModel,
    dFF,
    h,
    pDropout,
    depthInput,
    depthOutput,
):
    pass
