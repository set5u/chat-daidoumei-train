import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import random

batchSize = 16
numRecur = 8
log4Size = 3  # 256,64,16,4 = 3, 1小さい値
timeSteps = None


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


class PositionalEncoding(tf.keras.Model):
    def __init__(self, length, depth):
        super().__init__(trainable=False)
        self.length = length
        self.depth = depth

    def call(self, input):
        batchSize = input.shape[0]
        return tf.tile(
            positionalEncoding(self.length, self.depth)[np.newaxis, :, :],
            (batchSize, 1, 1),
        )

    def compute_output_shape(self, inputShape):
        return inputShape


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


class AddNorm(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        self.norm.build(
            input_shape[0] if isinstance(input_shape[0], tuple) else input_shape
        )

    def call(self, *inputs):
        return self.norm(inputs[0] + inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0] if isinstance(input_shape[0], tuple) else input_shape


class FF(tf.keras.Model):
    def __init__(self, dModel, dFF, maxLen, use_causal_mask=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = (
            tf.linalg.LinearOperatorLowerTriangular(
                tf.ones((maxLen, maxLen))
            ).to_dense()
            if use_causal_mask
            else tf.ones((maxLen, maxLen))
        )
        self.ff0 = tf.keras.layers.EinsumDense(
            "acfd,de->ace", (maxLen, dFF), activation="relu"
        )
        self.ff1 = tf.keras.layers.EinsumDense(
            "acfe,dc->acd", (maxLen, dModel), activation="linear"
        )
        self.maxLen = maxLen

    def build(self, input_shape):
        input_shape = (
            input_shape[0:1] + input_shape[1:2] + input_shape[1:2] + input_shape[2:]
        )
        self.ff0.build(input_shape)
        input_shape = self.ff0.compute_output_shape(input_shape)
        input_shape = (
            input_shape[0:1] + input_shape[1:2] + input_shape[1:2] + input_shape[2:]
        )
        self.ff1.build(input_shape)

    def call(self, *inputs):
        mask = self.mask[tf.newaxis, :, :, tf.newaxis]
        input = tf.tile(inputs[0][:, tf.newaxis], (1, self.maxLen, 1, 1))
        ret = input * mask
        ret = self.ff0(ret)
        ret = tf.tile(ret[:, tf.newaxis], (1, self.maxLen, 1, 1))
        ret = ret * mask
        ret = self.ff1(ret)
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape


class MiddleLayer(tf.keras.Model):
    def __init__(self, h, keyDim, maxLen, use_causal_mask=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dModel = h * keyDim
        self.dModel = dModel
        self.dModelLen = dModel * maxLen
        self.maxLen = maxLen
        self.use_causal_mask = use_causal_mask
        self.masks = (
            tf.linalg.LinearOperatorLowerTriangular(
                tf.ones((maxLen, maxLen))
            ).to_dense()
            if use_causal_mask
            else tf.ones((maxLen, maxLen))
        )
        self.reshape0 = tf.keras.layers.Reshape(target_shape=(-1, maxLen * dModel))
        self.rnn = tf.keras.layers.GRU(
            dModel * maxLen,
            return_sequences=True,
            return_state=True,
        )
        self.reshape1 = tf.keras.layers.Reshape(target_shape=(-1, maxLen, dModel))
        self.concat = tf.keras.layers.Concatenate(2)

    def build(self, input_shapes):
        input_shape = input_shapes[0]
        computed = input_shape[0:2] + (self.dModelLen,)
        self.reshape0.build(input_shape)
        self.rnn.build(computed)
        self.reshape1.build(computed)
        self.concat.build(
            (
                input_shape[0:2] + (None,) + input_shape[3:],
                input_shape[0:2] + (1,) + input_shape[3:],
            )
        )

    def call(self, *inputs):
        input = inputs[0]
        initialState = inputs[1] if len(inputs) == 2 else None
        if self.use_causal_mask:
            ret = tf.constant(
                0.0,
                "float32",
                (batchSize if input.shape[0] is None else input.shape[0],)
                + input.shape[1:2]
                + (0,)
                + input.shape[3:],
            )
            for i in range(self.maxLen):
                c, state = self.rnn(
                    self.reshape0(
                        input * self.masks[i][tf.newaxis, tf.newaxis, :, tf.newaxis]
                    ),
                    initial_state=initialState,
                )
                ret = self.concat((ret, self.reshape1(c)[:, :, i : i + 1, :]))
        else:
            ret, state = self.rnn(
                self.reshape0(input),
                initial_state=initialState,
            )
            ret = self.reshape1(ret)
        return ret, state

    def compute_output_shape(self, input_shapes):
        return input_shapes[0], input_shapes[0][:2] + (self.dModelLen,)


class InvSoftmax(tf.keras.Model):
    supports_masking = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, input, mask=None):
        b = -tf.argsort(tf.zeros((input.shape[2],))) / input.shape[2]
        ret = tf.einsum("abc,d->abcd", input, b) + 1
        return self.softmax(-tf.math.log(ret), mask=mask)

    def build(self, input_shape):
        self.softmax.build(input_shape)

    def compute_output_shape(self, input_shapes):
        return input_shapes


class InvSoftmaxTiler(tf.keras.Model):
    supports_masking = True

    def call(self, input):
        return tf.tile(
            input[:, :, :, tf.newaxis],
            (1, 1, 1, input.shape[1]),
        )

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2], input_shape[2]]


def toTimebasedTensor(r, size):
    r = tf.reshape(r, (-1, size, size, size // 4, 4))
    r = tf.transpose(r, (0, 1, 3, 2, 4))
    r = tf.reshape(r, (-1, size, (size // 4) ** 2, 4, 4))
    r = tf.transpose(r, (0, 2, 1, 3, 4))
    r = tf.reshape(r, (-1, (size // 4) ** 3, 4, 4, 4))
    r = tf.transpose(r, (0, 1, 4, 3, 2))
    return r


def fromTimebasedTensor(r, size):
    r = tf.reshape(r, (-1, (size // 4) ** 2, size, 4, 4))
    r = tf.transpose(r, (0, 2, 1, 3, 4))
    r = tf.reshape(r, (-1, size, (size // 4) ** 1, size, 4))
    r = tf.transpose(r, (0, 1, 3, 2, 4))
    r = tf.reshape(r, (-1, size, size, size))
    r = tf.transpose(r, (0, 3, 2, 1))
    return r


def upscaleTensor(r, size):
    r = tf.reshape(r, (-1, (size // 4) ** 3, 1, 1, 1))
    r = tf.tile(r, (1, 1, 4, 4, 4))
    r = fromTimebasedTensor(r, size)
    return r


def downscaleTensor(r, size):
    r = toTimebasedTensor(r, size)
    r = tf.reduce_sum(r, (2, 3, 4)) / 64
    r = tf.reshape(r, (-1, size // 4, size // 4, size // 4))
    r = tf.transpose(r, (0, 3, 2, 1))
    return r


class Averager(tf.keras.Model):
    supports_masking = True

    def call(self, input):
        level = input.shape[1]
        ret = []
        for j in range(level):
            r = input
            r = tf.transpose(input, (1, 0, 2, 3, 4))[j]
            for i in range(j):
                size = 4 ** (level - i + 1)
                r = downscaleTensor(r, size)
            for i in range(j):
                r = size = 4 ** ((level - i) + j)
                upscaleTensor(r, size)
            ret.append(r)
        return tf.transpose(ret, (1, 0, 2, 3, 4))

    def compute_output_shape(self, input_shape):
        return input_shape


class AveragedTiler(tf.keras.Model):
    supports_masking = True

    def __init__(self, level, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.level = level

    def call(self, input):
        cLevel = math.floor(math.log2(input.shape[1] - 0.1) / 2)
        ret = [input]
        i = cLevel + 2
        while self.level > i - 2:
            r = ret[0]
            size = 4**i
            r = upscaleTensor(r, size)
            ret.insert(0, r)
            i += 1
        while cLevel > 0:
            size = 4 ** (cLevel + 1)
            r = ret[-1]
            r = downscaleTensor(r, size)
            ret.append(r)
            cLevel -= 1
        for i in range(self.level):
            r = ret[i + 1]
            for j in range(i + 1):
                size = 4 ** (len(ret) - i + j)
                r = upscaleTensor(r, size)
            ret[i + 1] = r
        return tf.reshape(
            tf.transpose(ret, (1, 0, 2, 3, 4)),
            (input.shape[0], -1, (input.shape[1] * 4) ** 3),
        )

    def compute_output_shape(self, input_shape):
        return input_shape[0:1] + (None,) + (input_shape[1] ** 3,)


def useConverter(dModel, h, pDropout, layers):
    pass


def useRecursiveTransformer(
    dModel,
    h,
    pDropout,
    depthInput,
    depthOutput,
    depth,
    layers,
    numRecur,
    log4Size,
):
    input = tf.keras.Input(shape=(timeSteps, 4 ** (log4Size + 1)))
    mask = tf.keras.layers.Minimum()([input, tf.constant([1.0])])
    embedding = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Embedding(
            input_dim=depthInput, output_dim=depth, mask_zero=True
        )
    )(input)


models = useRecursiveTransformer(32, 4, 0.1, 2300, 2300, 256, 16, numRecur, log4Size)
print(models)


def draw_heatmap(data):
    fig, ax = plt.subplots()
    ax.pcolor(data, cmap=plt.cm.jet)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    plt.show()


# s = 64
# tiler = AveragedTiler(3)
# x = tf.reshape(tf.argsort(tf.zeros((1 * s * s * s))), (1, s, s, s)) / (1 * s * s * s)
# draw_heatmap(tf.reduce_sum(x[0], 0))
# y = tiler(x)
# y = tf.reshape(y, (1, -1, 256, 256, 256))
# draw_heatmap(tf.reduce_sum(y[0][0], 0))
# draw_heatmap(tf.reduce_sum(y[0][1], 0))
# draw_heatmap(tf.reduce_sum(y[0][2], 0))
# draw_heatmap(tf.reduce_sum(y[0][3], 0))
