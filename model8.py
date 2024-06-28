import json
import pickle
import tensorflow as tf
import numpy as np
import math
import random

toTrain = False

dtype = "float32"


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
    return tf.constant(ret, dtype)


def useExtendedBERT(
    dModel,
    dFF,
    pDropout,
    h,
    maxLen,
    depthInput,
    depthOutput,
    layers,
):
    positionalEncodingInput = tf.keras.Input((maxLen**2, dModel), dtype=dtype)
    input = tf.keras.Input((maxLen**2,))
    ones = tf.keras.layers.Dense(
        units=maxLen**2,
        kernel_initializer="zeros",
        bias_initializer="ones",
        trainable=False,
    )(input)
    attentionMask = tf.keras.layers.Minimum()([input, ones])
    embedding = tf.keras.layers.Embedding(
        input_dim=depthInput,
        output_dim=dModel,
        mask_zero=True,
    )(input)
    positionalEncoding = tf.keras.layers.Add()((embedding, positionalEncodingInput))
    start = tf.keras.Model(
        (positionalEncodingInput, input),
        (positionalEncoding, attentionMask),
    )
    outLayers = []
    for _ in range(layers):
        layersInput = tf.keras.Input((maxLen**2, dModel))
        layersMaskInput = tf.keras.Input((maxLen**2,))
        layersStateInput = tf.keras.Input((maxLen**2, dModel))
        attn0 = tf.keras.layers.MultiHeadAttention(h, dModel // h)(
            layersInput, attention_mask=layersMaskInput, use_causal_mask=True
        )
        add0 = tf.keras.layers.Add()([attn0, layersInput])
        norm0 = tf.keras.layers.LayerNormalization()(add0)
        dropout0 = tf.keras.layers.Dropout(pDropout)(norm0)
        attn1 = tf.keras.layers.MultiHeadAttention(h, dModel // h)(
            dropout0,
            layersStateInput,
            attention_mask=layersMaskInput,
            use_causal_mask=True,
        )
        add1 = tf.keras.layers.Add()([attn1, norm0])
        norm1 = tf.keras.layers.LayerNormalization()(add1)
        dropout1 = tf.keras.layers.Dropout(pDropout)(norm1)
        dense0 = tf.keras.layers.EinsumDense(
            "abc,bcd->abd", (maxLen**2, dFF), "relu"
        )(dropout1)
        dense1 = tf.keras.layers.EinsumDense("abd,bcd->abc", (maxLen**2, dModel))(
            dense0
        )
        add2 = tf.keras.layers.Add()([dense1, dropout1])
        norm2 = tf.keras.layers.LayerNormalization()(add2)
        dropout2 = tf.keras.layers.Dropout(pDropout)(norm2)
        outLayers.append(
            tf.keras.Model((layersInput, layersMaskInput, layersStateInput), dropout2)
        )
    convInput = tf.keras.Input((maxLen**2, dModel))
    permute0 = tf.keras.layers.Permute((2, 1))(convInput)
    conv = tf.keras.layers.Conv1D(maxLen, 1)(permute0)
    permute1 = tf.keras.layers.Permute((2, 1))(conv)
    convModel = tf.keras.Model(convInput, permute1)
    outInput = tf.keras.Input((maxLen**2, dModel))
    outDense = tf.keras.layers.Dense(depthOutput, "softmax")
    outModel = tf.keras.Model(outInput, outDense)
    return start, outLayers, convModel, outModel


with open("./num2word.json", "r", -1, "utf-8") as f:
    num2word = json.loads("".join(f.readlines()))
with open("./word2num.json", "r", -1, "utf-8") as f:
    word2num = json.loads("".join(f.readlines()))
with open("./wordTokens.json", "r", -1, "utf-8") as f:
    tokens = json.loads("".join(f.readlines()))
depth = len(num2word)
maxLen = 4
# params =
dModel = 256
dFF = 512
layers = 16
h = 8
numRecur = 4
models = useExtendedBERT(
    dModel,
    dFF,
    0.2,
    h,
    maxLen,
    depth,
    depth,
    layers,
)
funcs = []
funcs.append(tf.function(lambda x, **kwargs: models[0](x, **kwargs)))
funcs.append([tf.function(lambda x, **kwargs: m(x, **kwargs)) for m in models[1]])
funcs.append(tf.function(lambda x, **kwargs: models[2](x, **kwargs)))
funcs.append(tf.function(lambda x, **kwargs: models[3](x, **kwargs)))


batchSize = 64 if toTrain else 1
