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
    maxLen,
    pDropout,
    depthInput,
    depthOutput,
):
    input = tf.keras.Input((maxLen,))
    embedding = tf.keras.layers.Embedding(depthInput, dModel)(input)
    mul = tf.keras.layers.Multiply()([embedding, tf.constant([1 / math.sqrt(dModel)])])
    positionalEncodingOut = tf.keras.layers.Add(
        [mul, positionalEncoding(maxLen, dModel)[tf.newaxis]]
    )
    inModel = tf.keras.Model(input, positionalEncodingOut)
    return (inModel,)
