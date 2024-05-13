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


def toTimebasedTensor(r, size, base):
    r = tf.reshape(r, (-1, base, size // base, size, size))
    r = tf.transpose(r, (0, 2, 1, 3, 4))
    r = tf.reshape(r, (-1, size // base, base**2, size // base, size))
    r = tf.transpose(r, (0, 1, 3, 2, 4))
    r = tf.reshape(r, (-1, size // base, size // base, base**3, size // base))
    r = tf.transpose(r, (0, 3, 1, 2, 4))
    return r


def fromTimebasedTensor(r, size, base):
    r = tf.transpose(r, (0, 2, 3, 1, 4))
    r = tf.reshape(r, (-1, size // base, size // base, base**2, size))
    r = tf.transpose(r, (0, 1, 3, 2, 4))
    r = tf.reshape(r, (-1, size // base, base, size, size))
    r = tf.transpose(r, (0, 2, 1, 3, 4))
    r = tf.reshape(r, (-1, size, size, size))
    return r


def upscaleTensor(r, size):
    r = tf.reshape(r, (-1, (size // 4) ** 3, 1, 1, 1))
    r = tf.tile(r, (1, 1, 4, 4, 4))
    r = fromTimebasedTensor(r, size, size // 4)
    return r


def downscaleTensor(r, size):
    r = toTimebasedTensor(r, size, size // 4)
    r = tf.reduce_sum(r, (2, 3, 4)) / 64
    r = tf.reshape(r, (-1, size // 4, size // 4, size // 4))
    return r


class Averager(tf.keras.Model):
    supports_masking = True

    def call(self, input):
        level = input.shape[2]
        input = tf.transpose(input, (2, 0, 1, 3))
        ret = []
        for i in range(level):
            r = input[i]
            size = 4**level
            base = 4 ** (level - i)
            r = tf.reshape(r, (-1, (size // 4) ** 3, 4, 4, 4))
            r = fromTimebasedTensor(r, size, size // 4)
            r = tf.reshape(r, (-1, (size // base) ** 3, base, base, base))
            r = fromTimebasedTensor(r, size, size // base)
            for k in range(i):
                size = 4 ** (level - k)
                r = downscaleTensor(r, size)
            for k in range(i):
                r = tf.tile(r, (1, 4, 4, 4))
            r = toTimebasedTensor(r, 4**level, 4 ** (level - 1))
            r = tf.reshape(r, (-1, (4 ** (level - 1)) ** 3, 4**3))
            ret.append(r)
        return tf.transpose(ret, (1, 2, 0, 3))

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
        while cLevel > 0:
            size = 4 ** (cLevel + 1)
            r = ret[-1]
            r = downscaleTensor(r, size)
            ret.append(r)
            cLevel -= 1
        for i in range(self.level + 1):
            r = ret[i]
            for j in range(i):
                r = tf.tile(r, (1, 4, 4, 4))
            r = toTimebasedTensor(r, 4 ** (self.level + 1), 4**self.level)
            r = tf.reshape(r, (-1, (4**self.level) ** 3, 4**3))
            ret[i] = r
        return tf.transpose(ret, (1, 2, 0, 3))

    def compute_output_shape(self, input_shape):
        return input_shape[0:1] + ((4**self.level) ** 3, self.level + 1, 4**3)


class Splitter(tf.keras.Model):
    def __init__(self, split, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split

    def call(self, input):
        return tf.split(input, self.split, 1)

    def compute_output_shape(self, input):
        ret = ()
        for c in self.split:
            ret += ((input.shape[0:1] + (c,) + input.shape[2:],),)
        return ret


def useConverterCell(dModel, h, pDropout, layers):
    input = tf.keras.layers.Input(shape=(4**3 + layers * 2, dModel))
    stateInput = tf.keras.layers.Input(shape=(layers * 2 * dModel))
    splittedInput, splittedState = Splitter([4**3, layers * 2])(input)
    reshapedState = tf.keras.layers.Reshape(target_shape=(layers * 2 * dModel,))(
        splittedState
    )
    mergedState = stateInput + reshapedState


class ConverterCell(tf.keras.Model):
    def __init__(self, dModel, h, pDropout, layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell = useConverterCell(dModel, h, pDropout, layers)
        self.state_size = 4**log4Size * layers * dModel

    def call(self, inputs):
        return self.cell(inputs)

    def compute_output_shape(self, inputShape):
        return inputShape


def useConverter(dModel, h, pDropout, layers, log4Size, numRecur):
    # ConverterCell: reshape(T=log4Size+1,4**3,dModel) -> tile(1,1,dModel) -> cell -> dense(1) -> reshape(T,4**3)
    input = tf.keras.layers.Input(shape=(4**log4Size, log4Size + 1, 4**3))
    stateInput = tf.keras.layers.Input(shape=(4**log4Size * layers * 2 * dModel))
    averagerLayer = Averager()
    # concat input and state
    encoderCellLayer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.RNN(
            ConverterCell(dModel, h, pDropout, layers), return_sequences=True
        )
    )
    # state.unstack.reverse -> concat output and state
    decoderCellLayer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.RNN(
            ConverterCell(dModel, h, pDropout, layers), return_sequences=True
        )
    )


class Converter(tf.keras.Model):
    def __init__(
        self, dModel, h, pDropout, layers, log4Size, numRecur, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.converter = useConverter(dModel, h, pDropout, layers, log4Size, numRecur)
        self.state_size = 2 * dModel * layers

    def call(self, inputs):
        return self.converter(inputs)

    def compute_output_shape(self, inputShape):
        return inputShape


def useRecursiveTransformer(
    dModel,
    h,
    pDropout,
    depthInput,
    depthOutput,
    layers,
    numRecur,
    log4Size,
):
    input = tf.keras.Input(shape=(timeSteps, 4 ** (log4Size + 1)))
    stateInput = tf.keras.Input(shape=(numRecur, 2 * 4**log4Size * dModel * layers))
    mask = tf.keras.layers.Minimum()([input, tf.constant([1.0])])
    embedding = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Embedding(
            input_dim=depthInput, output_dim=4 ** (log4Size + 1), mask_zero=True
        )
    )(input)
    invSoftmaxTiler = tf.keras.layers.TimeDistributed(InvSoftmaxTiler())(embedding)
    invSoftmax = tf.keras.layers.TimeDistributed(InvSoftmax())(invSoftmaxTiler)
    averagedTiler = tf.keras.layers.TimeDistributed(AveragedTiler(log4Size))(invSoftmax)
    converterLayer = tf.keras.layers.RNN(
        Converter(dModel, h, pDropout, layers, log4Size, numRecur)
    )


models = useRecursiveTransformer(32, 4, 0.1, 2300, 2300, 16, numRecur, log4Size)
print(models)


def draw_heatmap(*data, rows=1, cols=1):
    fig = plt.figure()
    for i, d in enumerate(data):
        ax = plt.subplot(rows, cols, i + 1)
        c = ax.pcolor(d, cmap=plt.cm.jet)
        fig.colorbar(c, ax=ax)
    plt.show()


# s = 64
# tiler = AveragedTiler(2)
# x = tf.reshape(tf.argsort(tf.zeros((1 * s * s * s))), (1, s, s, s)) / (1 * s * s * s)
# draw_heatmap(tf.reduce_sum(x[0], 0))
# y = tiler(x)
# averager = Averager()
# z = averager(y)
# draw_heatmap(
#     tf.reduce_sum(
#         fromTimebasedTensor(tf.reshape(y[:, :, 0], (1, -1, 4, 4, 4)), s, s // 4)[0], 0
#     ),
#     tf.reduce_sum(
#         fromTimebasedTensor(tf.reshape(y[:, :, 1], (1, -1, 4, 4, 4)), s, s // 4)[0], 0
#     ),
#     tf.reduce_sum(
#         fromTimebasedTensor(tf.reshape(y[:, :, 2], (1, -1, 4, 4, 4)), s, s // 4)[0], 0
#     ),
#     # tf.reduce_sum(
#     #     fromTimebasedTensor(tf.reshape(y[:, :, 3], (1, -1, 4, 4, 4)), s, s // 4)[0], 0
#     # ),
#     tf.reduce_sum(
#         fromTimebasedTensor(tf.reshape(z[:, :, 0], (1, -1, 4, 4, 4)), s, s // 4)[0], 0
#     ),
#     tf.reduce_sum(
#         fromTimebasedTensor(tf.reshape(z[:, :, 1], (1, -1, 4, 4, 4)), s, s // 4)[0], 0
#     ),
#     tf.reduce_sum(
#         fromTimebasedTensor(tf.reshape(z[:, :, 2], (1, -1, 4, 4, 4)), s, s // 4)[0], 0
#     ),
#     # tf.reduce_sum(
#     #     fromTimebasedTensor(tf.reshape(z[:, :, 3], (1, -1, 4, 4, 4)), s, s // 4)[0], 0
#     # ),
#     rows=2,
#     cols=3,
# )

# w = tf.reshape(tf.argsort(tf.zeros((1 * 64 * 64 * 64))), (1, 64, 64, 64)) / (
#     1 * 64 * 64 * 64
# )

# draw_heatmap(tf.reduce_sum(w[0], 0))
# v = downscaleTensor(w, 64)
# draw_heatmap(tf.reduce_sum(v[0], 0))
# v = downscaleTensor(w, 16)
# draw_heatmap(tf.reduce_sum(v[0], 0))

# w = tf.reshape(tf.argsort(tf.zeros((1 * 4 * 4 * 4))), (1, 4, 4, 4)) / (1 * 4 * 4 * 4)
# draw_heatmap(tf.reduce_sum(w[0], 0))
# v = upscaleTensor(w, 16)
# draw_heatmap(tf.reduce_sum(v[0], 0))
# v = upscaleTensor(v, 64)
# draw_heatmap(tf.reduce_sum(v[0], 0))
