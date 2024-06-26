import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import random

tf.config.run_functions_eagerly(True)
batchSize = 3
numRecur = 8
log4Size = 1  # 16,4 = 2, 1小さい値
timeSteps = 2


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
        self.norm.build(input_shape[0])

    def call(self, *inputs):
        return self.norm(inputs[0] + inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, input, mask=None):
        b = tf.cast(
            -tf.argsort(tf.zeros((input.shape[2],))) / input.shape[2], "float32"
        )
        ret = tf.abs(input - b[tf.newaxis, tf.newaxis, tf.newaxis]) + 1.0
        return self.softmax(-tf.math.log(ret), mask=mask)

    def build(self, input_shape):
        self.softmax.build(input_shape)

    def compute_output_shape(self, input_shapes):
        return input_shapes


class InvSoftmaxTiler(tf.keras.Model):

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


class Extractor(tf.keras.Model):
    def __init__(self, log4Size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log4Size = log4Size

    def call(self, inputs):
        return fromTimebasedTensor(
            tf.reshape(inputs[:, :, 0, :], (-1, (4**log4Size) ** 3, 4, 4, 4)),
            4 ** (log4Size + 1),
            4**log4Size,
        )

    def compute_output_shape(self, inputShape):
        return (
            inputShape[0],
            4 ** (log4Size + 1),
            4 ** (log4Size + 1),
            4 ** (log4Size + 1),
        )


class Splitter(tf.keras.Model):
    def __init__(self, split, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split

    def call(self, input):
        a, b = tf.split(input, self.split, 2)
        b = tf.transpose(b, (0, 2, 1))
        return a, b

    def compute_output_shape(self, input):
        ret = ()
        for c in self.split:
            ret += (input[0:2] + (c,) + input[3:],)
        return ret


class StatePermuter(tf.keras.Model):
    def call(self, input):
        return tf.transpose(input, (0, 1, 2, 3))

    def compute_output_shape(self, input_shape):
        return input_shape[0][0][0:1] + (len(input_shape) * 2,) + input_shape[0][0][1:]


def useConverterCell(dModel, h, pDropout, layers):
    input = tf.keras.layers.Input(shape=(4**3 * (dModel + layers * 2),))
    stateInput = tf.keras.layers.Input(shape=(layers * 2 * 4**3,))
    reshapedInput = tf.keras.layers.Reshape(target_shape=(4**3, dModel + layers * 2))(
        input
    )
    splittedInput, splittedState = Splitter([dModel, layers * 2])(reshapedInput)
    reshapedState = tf.keras.layers.Reshape(target_shape=(layers * 2 * 4**3,))(
        splittedState
    )
    mergedState = stateInput + reshapedState
    mergedReshapedState = tf.keras.layers.Reshape(target_shape=(layers, 2, 4**3))(
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
        addNorm0 = AddNorm()(ff0, lastInput)
        dropout0 = tf.keras.layers.Dropout(pDropout)(addNorm0)
        gruPermute = tf.keras.layers.Permute((2, 1))(dropout0)
        fore, foreState = tf.keras.layers.GRU(
            4**3, return_sequences=True, return_state=True, go_backwards=False
        )(gruPermute, initial_state=mergedReshapedState[:, i, 0])
        back, backState = tf.keras.layers.GRU(
            4**3, return_sequences=True, return_state=True, go_backwards=True
        )(gruPermute, initial_state=mergedReshapedState[:, i, 1])
        stateOuts.append([foreState, backState])
        concatLayer1 = tf.keras.layers.Concatenate(2)
        foreback = concatLayer1([fore, back])
        attnPermute = tf.keras.layers.Permute((2, 1))(foreback)
        attn1 = tf.keras.layers.MultiHeadAttention(h, dModel // h)(
            attnPermute, attnPermute
        )
        ff1 = FF(dModel, 64)(attn1)
        addNorm1 = AddNorm()(ff1, dropout0)
        dropout1 = tf.keras.layers.Dropout(pDropout)(addNorm1)
        bypass.append(dropout1)
        lastInput = dropout1
        j = 1
        while (i + 1) % j == 0:
            lastInput = AddNorm()(bypass[i - j + 1], lastInput)
            j *= 2
    permutedState = StatePermuter()(stateOuts)
    outStateReshaped = tf.keras.layers.Reshape(target_shape=(layers * 2 * 4**3,))(
        permutedState
    )
    outStateDModelReshaped = tf.keras.layers.Permute((2, 1))(
        tf.keras.layers.Reshape(target_shape=(layers * 2, 4**3))(permutedState)
    )
    out = tf.keras.layers.Concatenate(2)([lastInput, outStateDModelReshaped])
    outReshaped = tf.keras.layers.Reshape(
        target_shape=(4**3 * (dModel + layers * 2),)
    )(out)
    return tf.keras.Model(
        inputs=[input, stateInput], outputs=[outReshaped, outStateReshaped]
    )


class ConverterCell(tf.keras.Model):
    def __init__(self, dModel, h, pDropout, layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell = useConverterCell(dModel, h, pDropout, layers)
        self.state_size = layers * 2 * 4**3
        self.output_size = 4**3 * (layers * 2 + dModel)

    def build(self, inputShape):
        self.cell.build(inputShape)

    def call(self, *inputs):
        return self.cell(inputs)

    def compute_output_shape(self, inputShape):
        return inputShape


class StateConcatter(tf.keras.Model):
    def __init__(self, dModel, layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dModel = dModel
        self.layersCount = layers

    # 4**3, dModel + layers * 2
    def call(self, inputs):
        input0Shape = inputs[0].shape
        batch_size = input0Shape[0] if batchSize is None else batchSize
        input0 = tf.tile(inputs[0][:, :, :, :, tf.newaxis], (1, 1, 1, 1, self.dModel))
        input1 = (
            inputs[1]
            if len(inputs) == 2
            else tf.zeros((batch_size * input0Shape[1] * 4**3 * self.layersCount * 2,))
        )
        input1 = tf.transpose(
            tf.reshape(
                input1,
                (
                    batch_size,
                    input0Shape[1],
                    -1,
                    self.layersCount * 2,
                    4**3,
                ),
            )[:, :, 0:1, :, :],
            (0, 1, 2, 4, 3),
        )
        input1Pad = tf.zeros(
            (
                batch_size,
                input0Shape[1],
                input0Shape[2] - 1,
                4**3,
                self.layersCount * 2,
            )
        )
        input1Concatted = tf.concat((input1, input1Pad), 2)
        return tf.concat((input0, input1Concatted), 4)

    def compute_output_shape(self, inputShapes):
        input0Shape = inputShapes[0]
        return (
            input0Shape[0],
            input0Shape[1],
            input0Shape[2],
            4**3,
            self.dModel + self.layersCount * 2,
        )


class StateSplitter(tf.keras.Model):
    def __init__(self, dModel, layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dModel = dModel
        self.layersCount = layers

    def call(self, input):
        ret, state = tf.split(input, [self.dModel, self.layersCount * 2], 4)
        return ret, tf.transpose(state[:, :, -1, :, :], (0, 1, 3, 2))

    def compute_output_shape(self, inputShapes):
        input0Shape = inputShapes
        batch_size = input0Shape[0] if batchSize is None else batchSize
        return (
            (batch_size, input0Shape[1], input0Shape[2], 4**3, self.dModel),
            (batch_size, input0Shape[1], self.layersCount * 2, 4**3),
        )


class DecoderStatePermuter(tf.keras.Model):
    def call(self, input):
        return tf.transpose(input[0], (1, 0, 2, 3, 4))

    def compute_output_shape(self, input_shape):
        return input_shape[0][0][0:1] + (len(input_shape) * 2,) + input_shape[0][0][1:]


def useConverter(dModel, h, pDropout, layers, log4Size, numRecur):
    input = tf.keras.layers.Input(shape=((4**log4Size) ** 3 * (log4Size + 1) * 4**3,))
    inputReshaped = tf.keras.layers.Reshape(
        target_shape=((4**log4Size) ** 3, (log4Size + 1), 4**3)
    )(input)
    stateInput = tf.keras.layers.Input(
        shape=(numRecur * (4**log4Size) ** 3 * layers * 2 * 4**3,)
    )
    concatterLayer = StateConcatter(dModel, layers)
    splitterLayer = StateSplitter(dModel, layers)
    averagerLayer = Averager()
    # concat input and state
    encoderCellLayer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.RNN(
            ConverterCell(dModel, h, pDropout, layers), return_sequences=True
        )
    )
    encoderDenseLayer = tf.keras.layers.EinsumDense(
        "abcde,de->abcd", ((4**log4Size) ** 3, log4Size + 1, 4**3)
    )
    concatter = concatterLayer((inputReshaped, stateInput))
    lastEncoderOutput = concatter
    lastEncoderBypass = inputReshaped
    encoderStates = []
    bypass = []
    reshapeFlatten = tf.keras.layers.Reshape(
        target_shape=((4**log4Size) ** 3, log4Size + 1, 4**3 * (layers * 2 + dModel))
    )
    reshapeUnflatten = tf.keras.layers.Reshape(
        target_shape=((4**log4Size) ** 3, log4Size + 1, 4**3, (layers * 2 + dModel))
    )
    for i in range(numRecur):
        encoderCell = reshapeUnflatten(
            encoderCellLayer(reshapeFlatten(lastEncoderOutput))
        )
        splitter, state = splitterLayer(encoderCell)
        encoderStates.append(state)
        encoderDense = encoderDenseLayer(splitter)
        averager = averagerLayer(encoderDense)
        bypass.append(lastEncoderBypass)
        lastEncoderBypass = averager
        j = 1
        while (i + 1) % j == 0:
            lastEncoderBypass = AddNorm()(bypass[i - j + 1], lastEncoderBypass)
            j *= 2
        lastDecoderOutput = lastEncoderBypass
        concatter = concatterLayer((lastEncoderBypass,))
        lastEncoderOutput = concatter

    # state.unstack.reverse -> concat output and state
    decoderCellLayer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.RNN(
            ConverterCell(dModel, h, pDropout, layers), return_sequences=True
        )
    )
    decoderDenseLayer = tf.keras.layers.EinsumDense(
        "abcde,de->abcd", ((4**log4Size) ** 3, log4Size + 1, 4**3)
    )
    bypass = []
    encoderStates.reverse()
    decoderStates = []
    for i in range(numRecur):
        concatter = concatterLayer((lastDecoderOutput, encoderStates[i]))
        decoderCell = reshapeUnflatten(decoderCellLayer(reshapeFlatten(concatter)))
        splitter, state = splitterLayer(decoderCell)
        decoderStates.append(state)
        decoderDense = decoderDenseLayer(splitter)
        averager = averagerLayer(decoderDense)
        bypass.append(lastDecoderOutput)
        lastDecoderOutput = averager
        j = 1
        while (i + 1) % j == 0:
            lastDecoderOutput = AddNorm()(bypass[i - j + 1], lastDecoderOutput)
            j *= 2
    decoderStates.reverse()
    reshapedOutput = tf.keras.layers.Reshape(
        target_shape=((4**log4Size) ** 3 * (log4Size + 1) * 4**3,)
    )(lastDecoderOutput)
    permutedStates = DecoderStatePermuter()((decoderStates,))
    reshapedStates = tf.keras.layers.Reshape(target_shape=(-1,))(permutedStates)
    return tf.keras.Model(
        inputs=(input, stateInput), outputs=[reshapedOutput, reshapedStates]
    )


class Converter(tf.keras.Model):
    def __init__(
        self, dModel, h, pDropout, layers, log4Size, numRecur, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.converter = useConverter(dModel, h, pDropout, layers, log4Size, numRecur)
        self.state_size = numRecur * (4**log4Size) ** 3 * layers * 2 * 4**3
        self.output_size = (4**log4Size) ** 3 * (log4Size + 1) * 4**3

    def build(self, inputShape):
        self.converter.build(inputShape)

    def call(self, *inputs):
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
    input = tf.keras.Input(batch_shape=(batchSize, timeSteps, 4 ** (log4Size + 1)))
    stateInput = tf.keras.Input(
        batch_shape=(
            batchSize,
            numRecur * (4**log4Size) ** 3 * layers * 2 * 4**3,
        )
    )
    embedding = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Embedding(input_dim=depthInput, output_dim=4 ** (log4Size + 1))
    )(input)
    invSoftmaxTiler = tf.keras.layers.TimeDistributed(InvSoftmaxTiler())(embedding)
    invSoftmax = tf.keras.layers.TimeDistributed(InvSoftmax())(invSoftmaxTiler)
    averagedTiler = tf.keras.layers.TimeDistributed(AveragedTiler(log4Size))(invSoftmax)
    tileReshaped = tf.keras.layers.Reshape(
        target_shape=(timeSteps, (4**log4Size) ** 3 * (log4Size + 1) * 4**3)
    )(averagedTiler)
    converterLayer, state = tf.keras.layers.RNN(
        Converter(dModel, h, pDropout, layers, log4Size, numRecur),
        return_state=True,
        return_sequences=True,
    )(tileReshaped, initial_state=stateInput)
    reshape = tf.keras.layers.Reshape(
        target_shape=(timeSteps, (4**log4Size) ** 3, log4Size + 1, 4**3)
    )(converterLayer)
    extract = tf.keras.layers.TimeDistributed(Extractor(log4Size))(reshape)
    outputDense = tf.keras.layers.TimeDistributed(
        tf.keras.layers.EinsumDense(
            "abcd,de->abe",
            (4 ** (log4Size + 1), depthOutput),
            activation="softmax",
        )
    )(extract)
    return {
        "predicter": tf.keras.Model(
            inputs=[input, stateInput], outputs=[outputDense, state]
        ),
        "trainer": tf.keras.Model(inputs=[input, stateInput], outputs=outputDense),
    }


with open("./num2char.json") as f:
    num2char = json.loads("".join(f.readlines()))
with open("./char2num.json") as f:
    char2num = json.loads("".join(f.readlines()))
with open("./tokens.json") as f:
    tokens = json.loads("".join(f.readlines()))
flattenTokens = sum(tokens, [])
depth = len(num2char)
stepsPerEpoch = 1


def loader():
    while True:
        xs = []
        ys = []
        for i in range(batchSize * stepsPerEpoch):
            startIndex = math.floor(
                random.random()
                * (len(flattenTokens) - 4 ** (log4Size + 1) * (timeSteps + 1))
            )
            count = (
                math.floor(random.random() * 4 ** (log4Size + 1))
                + 4 ** (log4Size + 1) * timeSteps
            )
            r = flattenTokens[startIndex : startIndex + count]
            r = [0] * (4 ** (log4Size + 1) * (timeSteps + 1) - count) + r
            r = np.array(r).reshape(timeSteps + 1, 4 ** (log4Size + 1))
            xs.append(r[0:-1])
            ys.append(r[1:])
        yield (np.array(xs), np.array(ys))


models = useRecursiveTransformer(32, 4, 0.1, depth, depth, 16, numRecur, log4Size)
models["trainer"].summary()


class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, _):
        toSave = save(models["trainer"])
        with open(
            "/content/drive/MyDrive/chat-daidoumei-train/weights/weight-"
            + str(1)
            + ".jsonl",
            "w",
        ) as f:
            f.write(toSave)


def train():
    state = tf.zeros((batchSize * stepsPerEpoch, 1048576))
    trainDatas = loader()
    epoch = 0
    optimizer = tf.keras.optimizers.Adadelta(1.0)
    models["trainer"].compile(
        optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True,
    )
    while True:
        print("epoch" + str(epoch))
        data = next(trainDatas)
        models["trainer"].fit(
            [data[0], state],
            data[1],
            batch_size=batchSize,
            steps_per_epoch=stepsPerEpoch,
            epochs=1,
            callbacks=[Callback()] if epoch % 200 == 1 else [],
        )


def predict():
    pass


toTrain = True
if toTrain:
    train()
else:
    predict()
