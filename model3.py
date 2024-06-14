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
        self.ff0 = tf.keras.layers.EinsumDense(
            "abcd,de->abde",
            (None, dModel, dFF),
            activation="relu",
        )
        self.ff1 = tf.keras.layers.EinsumDense(
            "abde,dc->abcd", (None, maxLen, dModel), activation="linear"
        )
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


class Splitter(tf.keras.Model):
    def call(self, input):
        return tf.split(input, 2, 3)


class FFPermuter(tf.keras.Model):
    def call(self, input):
        return tf.transpose(input, (1, 2, 3, 4, 0))


def useBERTTeacher(
    depthInput,
    depthOutput,
    dModelInter=128,
    dFF=1024,
    h=4,
    maxLen=8,
    layers=24,
):
    input = tf.keras.layers.Input((None, maxLen))
    embedding = tf.keras.layers.Embedding(depthInput, dModelInter, mask_zero=True)(
        input
    )
    bypass = []
    bypassStandalone = []
    lastOutput = embedding
    lastOutputStandalone = embedding
    attns = []
    stateInputs = []
    stateOutputs = []
    for i in range(layers):
        conv0Layer = tf.keras.layers.Conv2D(dModelInter, 1)
        conv0 = conv0Layer(lastOutput)
        conv0Standalone = conv0Layer(lastOutputStandalone)
        attnLayer = tf.keras.layers.MultiHeadAttention(h, dModelInter)
        attns.append(attnLayer)
        attn = attnLayer(lastOutput, lastOutput)
        attnStandalone = attnLayer(lastOutputStandalone, lastOutputStandalone)
        addNorm0Layer = AddNorm()
        addNorm0 = addNorm0Layer(conv0, attn)
        addNorm0Standalone = addNorm0Layer(conv0Standalone, attnStandalone)
        ffLayer = FF(dModelInter, dFF, maxLen)
        ff = ffLayer(addNorm0)
        ffStandalone = ffLayer(addNorm0Standalone)
        addNorm1Layer = AddNorm()
        addNorm1 = addNorm1Layer(ff, addNorm0)
        addNorm1Standalone = addNorm1Layer(ffStandalone, addNorm0Standalone)
        conv1Layer = tf.keras.layers.Conv2D(dModelInter, 1)
        conv1 = conv1Layer(addNorm1)
        conv1Standalone = conv1Layer(addNorm1Standalone)
        addNorm2Layer = AddNorm()
        addNorm2 = addNorm2Layer(lastOutput, conv1)
        addNorm2Standalone = addNorm2Layer(lastOutputStandalone, conv1Standalone)
        bypass.append(lastOutput)
        bypassStandalone.append(lastOutputStandalone)
        lastOutput = addNorm2
        lastOutputStandalone = addNorm2Standalone
        j = 2
        while (i + 1) % j == 0:
            lastOutputLayer = AddNorm()
            lastOutput = lastOutputLayer(bypass[i - j + 1], lastOutput)
            lastOutputStandalone = lastOutputLayer(
                bypassStandalone[i - j + 1], lastOutputStandalone
            )
            j *= 2
    attnLayer = tf.keras.layers.MultiHeadAttention(h, dModelInter)
    attn = attnLayer(lastOutput, lastOutput)
    attnStandalone = attnLayer(lastOutputStandalone, lastOutputStandalone)
    reducer = Reducer()(attn)
    reducerStandalone = Reducer()(attnStandalone)
    gruLayer = tf.keras.layers.GRU(dModelInter, return_state=True)
    gru, _ = gruLayer(reducer)
    denseLayer = tf.keras.layers.Dense(depthOutput, activation="softmax")
    dense = denseLayer(gru)
    model = tf.keras.Model(input, dense)
    optimizer = tf.keras.optimizers.Adadelta(1.0)
    model.compile(
        optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, attns, input, reducer, gruLayer, denseLayer


def useBERTStudent(
    depthInput,
    depthOutput,
    dModelInter=128,
    dModelIntra=64,
    dFF=128,
    h=4,
    r=2,
    maxLen=8,
    layers=24,
):
    input = tf.keras.layers.Input((None, maxLen))
    embedding = tf.keras.layers.Embedding(depthInput, dModelInter, mask_zero=True)(
        input
    )
    bypass = []
    lastOutput = embedding
    attns = []
    for i in range(layers):
        conv0 = tf.keras.layers.Conv2D(dModelInter, 1)(lastOutput)
        attnLayer = tf.keras.layers.MultiHeadAttention(h, dModelInter)
        attns.append(attnLayer)
        attn = attnLayer(lastOutput, lastOutput)
        addNorm0 = AddNorm()(conv0, attn)
        ff = []
        for splitted in Splitter()(addNorm0):
            ff.append(FF(dModelIntra, dFF, maxLen)(splitted))
        permuter = FFPermuter()(ff)
        reshape = tf.keras.layers.Reshape((-1, maxLen, dModelInter))(permuter)
        addNorm1 = AddNorm()(reshape, addNorm0)
        conv1 = tf.keras.layers.Conv2D(dModelInter, 1)(addNorm1)
        addNorm2 = AddNorm()(lastOutput, conv1)
        bypass.append(lastOutput)
        lastOutput = addNorm2
        j = 2
        while (i + 1) % j == 0:
            lastOutput = AddNorm()(bypass[i - j + 1], lastOutput)
            j *= 2
    attn = tf.keras.layers.MultiHeadAttention(h, dModelInter)(lastOutput, lastOutput)
    reducer = Reducer()(attn)
    gruLayer = tf.keras.layers.GRU(dModelInter, return_state=True)
    gru, _ = gruLayer(reducer)
    denseLayer = tf.keras.layers.Dense(depthOutput, activation="softmax")
    dense = denseLayer(gru)
    model = tf.keras.Model(input, dense)
    optimizer = tf.keras.optimizers.Adadelta(1.0)
    model.compile(
        optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, attns, input, reducer, gruLayer, denseLayer


with open("./num2word.json") as f:
    num2word = json.loads("".join(f.readlines()))
with open("./word2num.json") as f:
    word2num = json.loads("".join(f.readlines()))
with open("./wordTokens.json") as f:
    tokens = json.loads("".join(f.readlines()))
depth = len(num2word)
teacher, _, input, reducer, gruLayer, denseLayer = useBERTTeacher(depth, depth)
teacher.summary()
tf.keras.utils.plot_model(teacher, "teacher.png", show_shapes=True)
# with open("./weights/weight-2.jsonl") as f:
#     weights = load("".join(f.readlines()))
#     teacher.set_weights(weights)
# student, _ = useBERTStudent(depth, depth)
# student.summary()
# tf.keras.utils.plot_model(student, "student.png", show_shapes=True)
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


def predictTeacher():
    output = []
    predictModel = tf.keras.Model(input, reducer)
    state = tf.constant([[0.0] * 128])
    while True:
        inArray = tf.constant(
            [[[0] * (len(output) // 8 * 8 + 8 - len(output)) + output]]
        )
        result = predictModel(inArray)
        result, newState = gruLayer(result, initial_state=state)
        result = denseLayer(result)
        result = tf.argmax(result, 1)[0].numpy()
        output.append(result)
        print(num2word[result], end="\n" if result == 1 else "", flush=True)
        if len(output) == 8:
            state = newState
            output = []


predictTeacher()
