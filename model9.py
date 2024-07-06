import json
import tensorflow as tf
import numpy as np
import math
import random
import pickle


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
            lastInput, lastInput, use_causal_mask=True
        )
        add0 = tf.keras.layers.Add()([attn0, lastInput])
        norm0 = tf.keras.layers.LayerNormalization()(add0)
        conv0 = tf.keras.layers.Conv1D(dModel, 2, padding="same")(lastInput)
        dense2 = tf.keras.layers.EinsumDense("abc,bcd->abd", (maxLen, dModel), "relu")(
            conv0
        )
        add1 = tf.keras.layers.Add()([dense2, lastInput])
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
    maxPool = tf.keras.layers.MaxPool1D(maxLen, maxLen)
    outIn = tf.keras.Input((maxLen, dModel))
    outDense = tf.keras.layers.EinsumDense(
        "abc,cd->abd",
        (maxLen, depthOutput),
        "softmax",
    )
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
    None,
    tf.function(lambda x, **kwargs: models[3](x, **kwargs)),
]
toTrain = False
batchSize = 32 if toTrain else 1

trainableVariables = [
    models[0].trainable_variables
    + models[1].trainable_variables
    + models[3].trainable_variables
]


def predict():
    inputs = []
    while True:
        inputsReshaped = tf.reshape(
            [0] * (maxLen - len(inputs) % maxLen) + inputs, (-1, maxLen)
        )
        numRecur = math.floor(math.log(max(len(inputs), 1), maxLen)) + 1
        inLayerOut = funcs[0](inputsReshaped)
        states = [
            tf.constant(
                [
                    [[1.0 for _ in range(dModel)] for _ in range(maxLen)]
                    for _ in range(maxLen ** max(i - 1, 0))
                ]
            )
            for i in range(numRecur + 2)
        ]
        middleIns = tf.stack(
            [
                tf.constant([[0.0 for _ in range(dModel)] for _ in range(maxLen)])
                for _ in range(maxLen ** (numRecur - 1) - len(inputs) // maxLen - 1)
            ]
            + tf.unstack(inLayerOut)
        )
        middleOutput = [middleIns[-(maxLen**j) :] for j in range(numRecur)]
        for i in range(numRecur):
            for j in range(numRecur - i):
                middleInput = middleOutput[j]
                upperState = states[j + 2]
                upperState = models[2](upperState)
                upperState = tf.reshape(upperState, (-1, maxLen, dModel))
                lowerState = states[j]
                lowerState = tf.tile(states[j], (1 if j == 0 else maxLen, 1, 1))
                middleOutput[j] = funcs[1]((middleInput, upperState, lowerState))
                states[j + 1] = middleOutput[j]
        out = funcs[3](middleOutput[0])
        decoderSorted = tf.argsort(out[0][-1])
        results = []
        sum = 0
        for l in range(5):
            c = out[0][-1][decoderSorted[~l]].numpy()
            sum += c
            results.append(c)
        r = random.random() * sum
        t = 0
        m = 0
        while t < r:
            t += results[m]
            m += 1
        result = decoderSorted[~m + 1].numpy()
        inputs.append(result)
        print(num2word[result], end="\n" if result == 1 else "", flush=True)


numRecur = 3
lenRecur = 0


def train_step(optimizer, loader):
    global lenRecur
    lenRecur = random.randint(1, 3)
    xs, ys = next(loader)


def loader():
    while True:
        inputs = []
        outputs = []
        for _ in range(batchSize):
            startIndex = random.randint(0, len(tokens) - maxLen**numRecur - 1)
            inputs.append(tokens[startIndex : startIndex + maxLen**numRecur])
            outputs.append(tokens[startIndex + 1 : startIndex + maxLen**numRecur + 1])
        yield tf.constant(inputs), tf.constant(outputs)


def train():
    step = 0
    datas = loader()
    while True:
        print("step:", step, "loss:", train_step(optimizer, datas))
        step += 1
        if step % 10 == 0:
            models[0].save_weights("./weights/in")
            models[1].save_weights("./weights/middle")
            models[3].save_weights("./weights/out")
            with open(".weights/optimizer") as f:
                pickle.dump(optimizer.get_weights(), f)


optimizer = tf.keras.optimizers.Adadelta(1.0)
# optimizer.apply_gradients(
#     zip([tf.zeros_like(m) for m in trainableVariables], trainableVariables)
# )
# with open("./weights/optimizer", "wb") as f:
#     optimizer.set_weights(pickle.load(f))
# models[0].load_weights("./weights/in")
# models[1].load_weights("./weights/middle")
# models[3].load_weights("./weights/out")

if toTrain:
    train()
else:
    predict()
