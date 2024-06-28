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
    layersInput = tf.keras.Input((maxLen**2, dModel))
    layersMaskInput = tf.keras.Input((maxLen**2,))
    layersStateInput = tf.keras.Input((maxLen**2, dModel))
    lastInput = layersInput
    for _ in range(layers):
        attn0 = tf.keras.layers.MultiHeadAttention(h, dModel // h)(
            lastInput,
            lastInput,
            attention_mask=layersMaskInput,
            use_causal_mask=True,
        )
        add0 = tf.keras.layers.Add()([attn0, lastInput])
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
        lastInput = dropout2
    layerModel = tf.keras.Model(
        (layersInput, layersMaskInput, layersStateInput), lastInput
    )

    convInput = tf.keras.Input((maxLen**2, dModel))
    permute0 = tf.keras.layers.Permute((2, 1))(convInput)
    conv = tf.keras.layers.Conv1D(maxLen, 1)(permute0)
    permute1 = tf.keras.layers.Permute((2, 1))(conv)
    convDense = tf.keras.layers.EinsumDense("abc,bcd->adc", (maxLen, dModel))(permute1)
    convModel = tf.keras.Model(convInput, convDense)
    outInput = tf.keras.Input((maxLen**2, dModel))
    outDense = tf.keras.layers.Dense(depthOutput, "softmax")(outInput)
    outModel = tf.keras.Model(outInput, outDense)
    return start, layerModel, convModel, outModel


with open("./num2word.json", "r", -1, "utf-8") as f:
    num2word = json.loads("".join(f.readlines()))
with open("./word2num.json", "r", -1, "utf-8") as f:
    word2num = json.loads("".join(f.readlines()))
with open("./wordTokens.json", "r", -1, "utf-8") as f:
    tokens = json.loads("".join(f.readlines()))
depth = len(num2word)
maxLen = 8
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
models[0].summary()
models[1].summary()
models[2].summary()
models[3].summary()
tf.keras.utils.plot_model(models[0], "start.png", show_shapes=True)
tf.keras.utils.plot_model(models[1], "attn.png", show_shapes=True)
tf.keras.utils.plot_model(models[2], "conv.png", show_shapes=True)
tf.keras.utils.plot_model(models[3], "out.png", show_shapes=True)
funcs = []
funcs.append(tf.function(lambda x, **kwargs: models[0](x, **kwargs)))
funcs.append(tf.function(lambda x, **kwargs: models[1](x, **kwargs)))
funcs.append(tf.function(lambda x, **kwargs: models[2](x, **kwargs)))
funcs.append(tf.function(lambda x, **kwargs: models[3](x, **kwargs)))


batchSize = 64 if toTrain else 1

numRecur = 3  # len = 4096
