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
            input_shape[0][0] if isinstance(input_shape[0][0], tuple) else input_shape
        )

    def call(self, *inputs):
        return self.norm(inputs[0] + inputs[1])

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0][0] if isinstance(input_shape[0][0], tuple) else input_shape
        )


class FF(tf.keras.Model):
    def __init__(self, dModel, maxLen, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ff0 = tf.keras.layers.EinsumDense(
            "abc,de->ade",
            (maxLen, dModel),
            activation="relu",
        )
        self.ff1 = tf.keras.layers.EinsumDense(
            "abc,de->ade", (maxLen, dModel), activation="linear"
        )
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
            ret += (input[0:1] + (c,) + input[2:],)
        return ret


class StatePermuter(tf.keras.Model):
    def call(self, input):
        return tf.transpose(input, (1, 0, 2, 3))

    def compute_output_shape(self, input_shape):
        return input_shape[0][0][0:1] + (len(input_shape) * 2,) + input_shape[0][0][1:]


def useConverterCell(dModel, h, pDropout, layers):
    input = tf.keras.layers.Input(shape=(4**3 + layers * 2, dModel))
    stateInput = tf.keras.layers.Input(shape=(layers * 2 * dModel,))
    splittedInput, splittedState = Splitter([4**3, layers * 2])(input)
    reshapedState = tf.keras.layers.Reshape(target_shape=(layers * 2 * dModel,))(
        splittedState
    )
    mergedState = stateInput + reshapedState
    mergedReshapedState = tf.keras.layers.Reshape(target_shape=(layers, 2, dModel))(
        mergedState
    )
    bypass = []
    stateOuts = []
    lastInput = splittedInput
    for i in range(layers):
        reshape444Layer = tf.keras.layers.Reshape(target_shape=(4, 4, 4, -1))
        reshape64Layer = tf.keras.layers.Reshape(target_shape=(64, -1))
        reshape27Layer = tf.keras.layers.Reshape(target_shape=(27, -1))
        reshape8Layer = tf.keras.layers.Reshape(target_shape=(8, -1))
        reshape1Layer = tf.keras.layers.Reshape(target_shape=(1, -1))
        concatLayer0 = tf.keras.layers.Concatenate(1)
        reshape444 = reshape444Layer(lastInput)
        conv4 = tf.keras.layers.Conv3D(dModel, 4)(reshape444)
        conv3 = tf.keras.layers.Conv3D(dModel, 3)(reshape444)
        conv2 = tf.keras.layers.Conv3D(dModel, 2)(reshape444)
        conv1 = tf.keras.layers.Conv3D(dModel, 1)(reshape444)
        rconv4 = reshape1Layer(conv4)
        rconv3 = reshape8Layer(conv3)
        rconv2 = reshape27Layer(conv2)
        rconv1 = reshape64Layer(conv1)
        convConcatted = concatLayer0([rconv4, rconv3, rconv2, rconv1])
        attn0 = tf.keras.layers.MultiHeadAttention(h, dModel // h)(
            convConcatted, convConcatted
        )
        ff0 = FF(dModel, 64)(attn0)
        addNorm0 = AddNorm()([ff0, reshape444])
        dropout0 = tf.keras.layers.Dropout(pDropout)(addNorm0)
        fore, foreState = tf.keras.layers.GRU(
            dModel, return_sequences=True, return_state=True, go_backwards=False
        )(dropout0, initial_state=mergedReshapedState[i, 0])
        back, backState = tf.keras.layers.GRU(
            dModel, return_sequences=True, return_state=True, go_backwards=True
        )(dropout0, initial_state=mergedReshapedState[i, 1])
        stateOuts.append([foreState, backState])
        concatLayer1 = tf.keras.layers.Concatenate(1)
        foreback = concatLayer1([fore, back])
        attn1 = tf.keras.layers.MultiHeadAttention(h, dModel // h)(foreback, foreback)
        ff1 = FF(dModel, 64)(attn1)
        addNorm1 = AddNorm()([ff1, foreback])
        dropout1 = tf.keras.layers.Dropout(pDropout)(addNorm1)
        bypass.append(dropout1)
        lastInput = dropout1
        j = 1
        while (i + 1) % j == 0:
            lastInput = AddNorm()([bypass[i - j + 1], lastInput])
            j *= 2
    permutedState = StatePermuter()(stateOuts)
    outStateReshaped = tf.keras.layers.Reshape(target_shape=(layers * 2 * dModel,))(
        permutedState
    )
    outStateDModelReshaped = tf.keras.layers.Reshape(target_shape=(layers * 2, dModel))(
        permutedState
    )
    out = concatLayer0([lastInput, outStateDModelReshaped])
    return tf.keras.Model(inputs=[input, stateInput], outputs=[out, outStateReshaped])


cell = useConverterCell(32, 4, 0.1, 16)
cell.summary()
tf.keras.utils.plot_model(cell, "cell.png", show_shapes=True)


class ConverterCell(tf.keras.Model):
    def __init__(self, dModel, h, pDropout, layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell = useConverterCell(dModel, h, pDropout, layers)
        self.state_size = 4**log4Size * layers * dModel

    def call(self, *inputs):
        return self.cell(*inputs)

    def compute_output_shape(self, inputShape):
        return inputShape


class StateConcatter(tf.keras.Model):
    def __init__(self, dModel, layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dModel = dModel
        self.layers = layers

    def call(self, inputs):
        input0Shape = inputs[0].shape
        input0 = tf.tile(inputs[0][:, :, :, :, tf.newaxis], (1, 1, 1, 1, self.dModel))
        input1 = tf.reshape(
            inputs[1], (input0Shape[0], input0Shape[1], 1, self.layers * 2, self.dModel)
        )
        input1Pad = tf.zeros(
            (
                input0Shape[0],
                input0Shape[1],
                input0Shape[2] - 1,
                self.layers * 2,
                self.dModel,
            )
        )
        input1Concatted = tf.concat((input1, input1Pad), 2)
        return tf.concat((input0, input1Concatted), 3)

    def compute_output_shape(self, inputShapes):
        input0Shape = inputShapes[0]
        return (
            input0Shape[0],
            input0Shape[1],
            input0Shape[2],
            4**3 + self.layers * 2,
            self.dModel,
        )


class StateUnstacker(tf.keras.Model):
    def __init__(self, numRecur, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numRecur = numRecur

    def call(self, input):
        ret = tf.reshape(input, (input.shape[0], self.numRecur, -1))
        ret = tf.transpose(input, (1, 0, 2))
        ret = tf.unstack(input)
        return ret

    def compute_output_shape(self, inputShape):
        return inputShape[0:1] + (self.numRecur,) + (inputShape[2] // self.numRecur,)


def useConverter(dModel, h, pDropout, layers, log4Size, numRecur):
    # ConverterCell: reshape(T=log4Size+1,4**3,dModel) -> tile(1,1,dModel) -> cell -> dense(1) -> reshape(T,4**3)
    input = tf.keras.layers.Input(
        shape=(
            4**log4Size,
            log4Size + 1,
            4**3,
        )
    )
    stateInput = tf.keras.layers.Input(
        shape=(numRecur * 4**log4Size * layers * 2 * dModel,)
    )
    unstackerLayer = StateUnstacker(numRecur)(stateInput)
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
        self.state_size = numRecur * 4**log4Size * layers * 2 * dModel

    def call(self, *inputs):
        return self.converter(*inputs)

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
