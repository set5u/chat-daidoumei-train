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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        self.norm.build(input_shape[0])

    def call(self, *inputs):
        return self.norm(inputs[0] + inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class FF(tf.keras.Model):
    def __init__(self, dModel, maxLen, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ff0 = tf.keras.layers.Dense(dModel, activation="relu")
        self.ff1 = tf.keras.layers.Dense(dModel, activation="linear")
        self.dModel = dModel
        self.maxLen = maxLen

    def build(self, input_shape):
        self.ff0.build(input_shape[0])
        self.ff1.build(input_shape[0])

    def call(self, *inputs):
        ret = self.ff0(inputs[0])
        ret = self.ff1(ret)
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape[0][0:1] + (self.maxLen, self.dModel)
