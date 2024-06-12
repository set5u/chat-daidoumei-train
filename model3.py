import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import random


def draw_heatmap(*data, rows=1, cols=1):
    fig = plt.figure()
    for i, d in enumerate(data):
        ax = plt.subplot(rows, cols, i + 1)
        c = ax.pcolor(d, cmap=plt.cm.jet)
        fig.colorbar(c, ax=ax)
    plt.show()


def save(model):
    weights = model.get_weights()
    ret = []
    for weight in weights:
        ret.append(json.dumps(weight.tolist()))
    return "\n".join(ret)


def load(weights: str):
    weights = weights.split("\n")
    ret = []
    for weight in weights:
        ret.append(np.array(json.loads(weight), "float32"))
    return ret


class AddNorm(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        self.norm.build(input_shape)

    def call(self, *inputs):
        return self.norm(inputs[0] + inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class FF(tf.keras.Model):
    def __init__(self, dModel, dFF, maxLen, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ff0 = tf.keras.layers.Dense(dFF, activation="relu")
        self.ff1 = tf.keras.layers.Dense(dModel, activation="linear")
        self.dModel = dModel
        self.maxLen = maxLen

    def build(self, input_shape):
        self.ff0.build(input_shape)
        self.ff1.build(self.ff0.compute_output_shape(input_shape))

    def call(self, input):
        ret = self.ff0(input)
        ret = self.ff1(ret)
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape[0][0:1] + (self.maxLen, self.dModel)


class Reducer(tf.keras.Model):
    def call(self, input):
        return tf.reduce_sum(input, 2)


def useBERTLarge(
    depthInput,
    depthOutput,
    embeddingD=512,
    dModel=512,
    dFF=2048,
    h=16,
    maxLen=8,
    layers=24,
):
    input = tf.keras.layers.Input((None, maxLen))
    embedding = tf.keras.layers.Embedding(depthInput, embeddingD)(input)
    bypass = []
    lastOutput = embedding
    for i in range(layers):
        multiHeadAttention = tf.keras.layers.MultiHeadAttention(h, dModel)(
            lastOutput, lastOutput
        )
        addNorm0 = AddNorm()(multiHeadAttention, lastOutput)
        ff = tf.keras.layers.TimeDistributed(FF(dModel, dFF, maxLen))(addNorm0)
        addNorm1 = AddNorm()(addNorm0, ff)
        bypass.append(addNorm1)
        lastOutput = addNorm1
        j = 1
        while (i + 1) % j == 0:
            lastOutput = AddNorm()(bypass[i - j + 1], lastOutput)
            j *= 2
    attn = tf.keras.layers.MultiHeadAttention(h, dModel)(lastOutput, lastOutput)
    reducer = Reducer()(attn)
    gru = tf.keras.layers.GRU(dModel)(reducer)
    dense = tf.keras.layers.Dense(depthOutput, activation="softmax")(gru)
    model = tf.keras.Model(input, dense)
    optimizer = tf.keras.optimizers.Adadelta(1.0)
    model.compile(
        optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


with open("./num2word.json") as f:
    num2word = json.loads("".join(f.readlines()))
with open("./word2num.json") as f:
    word2num = json.loads("".join(f.readlines()))
with open("./wordTokens.json") as f:
    tokens = json.loads("".join(f.readlines()))
depth = len(num2word)
# large = useBERTLarge(depth, depth)
# large.summary()
# tf.keras.utils.plot_model(large, "large.png", show_shapes=True)

batchSize = 4
stepsPerEpoch = 4


def loader():
    while True:
        input = []
        output = []
        for _ in range(batchSize * stepsPerEpoch):
            startIndex = math.floor(random.random() * (len(tokens) - 257))
            input.append(tokens[startIndex : startIndex + 256])
            endIndex = startIndex + 256
            while tokens[endIndex] == 3:
                endIndex += 1
            output.append(tokens[endIndex])
        yield np.array(input).reshape((batchSize, -1, 8)), np.array(output)


data = loader()


def useBERTTeacher(
    depthInput,
    depthOutput,
    embeddingD=128,
    dModelInter=256,
    dModelIntra=512,
    dFF=1024,
    h=4,
    maxLen=8,
    layers=24,
):
    pass


def useBERTStudent(
    depthInput,
    depthOutput,
    embeddingD=128,
    dModelInter=256,
    dModelIntra=128,
    dFF=512,
    h=4,
    r=2,
    maxLen=8,
    layers=24,
):
    pass
    # xF == queue horizonal
