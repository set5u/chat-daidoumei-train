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
    layers,
):
    input = tf.keras.Input((maxLen,))
    embedding = tf.keras.layers.Embedding(depthInput, dModel)(input)
    positionalEncodingOut = tf.keras.layers.Add()(
        [embedding, positionalEncoding(maxLen, dModel)[tf.newaxis]]
    )
    inModel = tf.keras.Model(input, positionalEncodingOut)
    middleIn = tf.keras.Input((maxLen, dModel))
    upperStateInput = tf.keras.Input((maxLen, dModel))
    lowerStateInput = tf.keras.Input((maxLen, dModel))
    lastInput = middleIn
    bypass = []
    for i in range(layers):
        attn0 = tf.keras.layers.MultiHeadAttention(h, dModel // h, dropout=pDropout)(
            lastInput, lastInput
        )
        add0 = tf.keras.layers.Add()([attn0, lastInput])
        norm0 = tf.keras.layers.LayerNormalization()(add0)
        temp = lastInput
        for _ in range(maxLen):
            lastInput = tf.keras.layers.Conv1D(dModel, 2, padding="same")(lastInput)
            lastInput = tf.keras.layers.EinsumDense(
                "abc,bcd->abd", (maxLen, dModel), "relu"
            )(lastInput)
        add1 = tf.keras.layers.Add()([lastInput, temp])
        norm1 = tf.keras.layers.LayerNormalization()(add1)
        attn1 = tf.keras.layers.MultiHeadAttention(h, dModel // h, dropout=pDropout)(
            norm0, norm1
        )
        add2 = tf.keras.layers.Add()([attn1, norm0])
        norm2 = tf.keras.layers.LayerNormalization()(add2)
        attn2 = tf.keras.layers.MultiHeadAttention(h, dModel // h, dropout=pDropout)(
            norm2, upperStateInput
        )
        add3 = tf.keras.layers.Add()([attn2, norm2])
        norm3 = tf.keras.layers.LayerNormalization()(add3)
        attn3 = tf.keras.layers.MultiHeadAttention(h, dModel // h, dropout=pDropout)(
            lowerStateInput, norm3
        )
        add4 = tf.keras.layers.Add()([attn3, norm3])
        norm4 = tf.keras.layers.LayerNormalization()(add4)
        dense0 = tf.keras.layers.EinsumDense("abc,bcd->abd", (maxLen, dFF), "relu")(
            norm4
        )
        dense1 = tf.keras.layers.EinsumDense(
            "abd,bcd->abc", (maxLen, dModel), "linear"
        )(dense0)
        add5 = tf.keras.layers.Add()([dense1, norm4])
        norm5 = tf.keras.layers.LayerNormalization()(add5)
        bypass.append(lastInput)
        lastInput = norm5
        j = 1
        while (i + 1) % j == 0:
            temp = tf.keras.layers.Add()((bypass[i - j + 1], lastInput))
            lastInput = tf.keras.layers.LayerNormalization()(temp)
            j *= 2
    middleModel = tf.keras.Model(
        (middleIn, upperStateInput, lowerStateInput), lastInput
    )
    maxPool = tf.keras.layers.MaxPool2D((maxLen, maxLen), (maxLen, maxLen))
    outIn = tf.keras.Input((maxLen, dModel))
    outDense = tf.keras.layers.EinsumDense("abc,bcd->ad", (depthOutput,), "softmax")
    outModel = tf.keras.Model(outIn, outDense(outIn))
    return inModel, middleModel, maxPool, outModel


with open("./num2word.json", "r", -1, "utf-8") as f:
    num2word = json.loads("".join(f.readlines()))
with open("./word2num.json", "r", -1, "utf-8") as f:
    word2num = json.loads("".join(f.readlines()))
with open("./wordTokens.json", "r", -1, "utf-8") as f:
    tokens = json.loads("".join(f.readlines()))

depth = len(num2word)
dModel = 128
dFF = 256
h = 8
maxLen = 8
pDropout = 0.2
layers = 8
models = useRecursiveBERT(dModel, dFF, h, maxLen, pDropout, depth, depth, layers)
models[0].summary()
models[1].summary()
models[3].summary()
tf.keras.utils.plot_model(models[0], "in.png")
tf.keras.utils.plot_model(models[1], "middle.png")
tf.keras.utils.plot_model(models[3], "out.png")
funcs = [
    tf.function(lambda x, **kwargs: models[0](x, **kwargs)),
    tf.function(lambda x, **kwargs: models[1](x, **kwargs)),
    tf.function(lambda x, **kwargs: models[2](x, **kwargs)),
    tf.function(lambda x, **kwargs: models[3](x, **kwargs)),
]
toTrain = False
batchSize = 32 if toTrain else 1


def predict():
    pass


def train():
    pass


if toTrain:
    train()
else:
    predict()
