import json
import pickle
import tensorflow as tf
import numpy as np
import math
import random

toTrain = True

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
    embeddingScale = tf.keras.layers.Multiply()((embedding, tf.constant([dModel**0.5])))
    positionalEncoding = tf.keras.layers.Add()(
        (embeddingScale, positionalEncodingInput)
    )
    start = tf.keras.Model(
        (positionalEncodingInput, input),
        (positionalEncoding, attentionMask),
    )
    layersInput = tf.keras.Input((maxLen**2, dModel))
    layersMaskInput = tf.keras.Input((maxLen**2, 1))
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
    bridgeInput = tf.keras.Input((maxLen**2, dModel))
    bridgePermute0 = tf.keras.layers.Permute((2, 1))(bridgeInput)
    bridgeConv0 = tf.keras.layers.Conv1D(maxLen, 1)(bridgePermute0)
    bridgeDense = tf.keras.layers.EinsumDense("abc,bcde->ade", (maxLen, dModel))(
        bridgeConv0
    )
    bridgeModel = tf.keras.Model(bridgeInput, bridgeDense)
    collectorInput = tf.keras.Input((maxLen**2, dModel))
    collectorPositionalEncodingInput = tf.keras.Input((maxLen**2, dModel))
    collectorPositionalEncoding = tf.keras.layers.Add()(
        (collectorInput, collectorPositionalEncodingInput)
    )
    lastInput = collectorPositionalEncoding
    for _ in range(layers):
        attn0 = tf.keras.layers.MultiHeadAttention(h, dModel // h)(
            lastInput,
            lastInput,
        )
        add0 = tf.keras.layers.Add()([attn0, lastInput])
        norm0 = tf.keras.layers.LayerNormalization()(add0)
        dropout0 = tf.keras.layers.Dropout(pDropout)(norm0)
        dense0 = tf.keras.layers.EinsumDense(
            "abc,bcd->abd", (maxLen**2, dFF), "relu"
        )(dropout0)
        dense1 = tf.keras.layers.EinsumDense("abd,bcd->abc", (maxLen**2, dModel))(
            dense0
        )
        add2 = tf.keras.layers.Add()([dense1, dropout0])
        norm2 = tf.keras.layers.LayerNormalization()(add2)
        dropout2 = tf.keras.layers.Dropout(pDropout)(norm2)
        lastInput = dropout2
    collectorModel = tf.keras.Model(
        (collectorPositionalEncodingInput, collectorInput), lastInput
    )
    convInput = tf.keras.Input((maxLen**2, dModel))
    permute0 = tf.keras.layers.Permute((2, 1))(convInput)
    conv = tf.keras.layers.Conv1D(maxLen, 1)(permute0)
    convDense = tf.keras.layers.EinsumDense("abc,bcde->ade", (maxLen, dModel))(conv)
    convModel = tf.keras.Model(convInput, convDense)
    outInput = tf.keras.Input((maxLen**2, dModel))
    outDense = tf.keras.layers.Dense(depthOutput, "softmax")(outInput)
    outModel = tf.keras.Model(outInput, outDense)
    return start, layerModel, bridgeModel, convModel, collectorModel, outModel


with open("./num2word.json", "r", -1, "utf-8") as f:
    num2word = json.loads("".join(f.readlines()))
with open("./word2num.json", "r", -1, "utf-8") as f:
    word2num = json.loads("".join(f.readlines()))
with open("./wordTokens.json", "r", -1, "utf-8") as f:
    tokens = json.loads("".join(f.readlines()))
depth = len(num2word)
maxLen = 8
# params =
dModel = 128
dFF = 256
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
models[4].summary()
models[5].summary()
tf.keras.utils.plot_model(models[0], "start.png", show_shapes=True)
tf.keras.utils.plot_model(models[1], "attn.png", show_shapes=True)
tf.keras.utils.plot_model(models[2], "bridge.png", show_shapes=True)
tf.keras.utils.plot_model(models[3], "conv.png", show_shapes=True)
tf.keras.utils.plot_model(models[4], "collector.png", show_shapes=True)
tf.keras.utils.plot_model(models[5], "out.png", show_shapes=True)
funcs = []
funcs.append(tf.function(lambda x, **kwargs: models[0](x, **kwargs), jit_compile=True))
funcs.append(tf.function(lambda x, **kwargs: models[1](x, **kwargs), jit_compile=True))
funcs.append(tf.function(lambda x, **kwargs: models[2](x, **kwargs), jit_compile=True))
funcs.append(tf.function(lambda x, **kwargs: models[3](x, **kwargs), jit_compile=True))
funcs.append(tf.function(lambda x, **kwargs: models[4](x, **kwargs), jit_compile=True))
funcs.append(tf.function(lambda x, **kwargs: models[5](x, **kwargs), jit_compile=True))


batchSize = 32 if toTrain else 1

numRecur = 3  # len = 4096

zeroState = tf.constant(
    [[[0.0 for _ in range(dModel)] for _ in range(maxLen**2)] for _ in range(batchSize)]
)
zeroPad = tf.constant(
    [[[0.0 for _ in range(dModel)] for _ in range(maxLen)] for _ in range(batchSize)]
)
positionalEncodingInput = positionalEncoding(maxLen**2, dModel)[tf.newaxis]


def collectArray(x):
    return tf.reshape(
        tf.transpose(x, (1, 2, 0, 3)),
        (batchSize, maxLen**2, dModel),
    )


def predict():
    state = zeroState
    states = [[]]
    while True:
        inputs = []
        for i in range(maxLen**2):
            r, m = funcs[0](
                (
                    positionalEncodingInput,
                    tf.constant([inputs + [0] * (maxLen**2 - len(inputs))]),
                )
            )
            rs = funcs[1]((r, m[:, :, tf.newaxis], state))
            r = funcs[5](rs)
            decoderSorted = tf.argsort(r[0][i])
            results = []
            sum = 0
            for l in range(5):
                c = r[0][i][decoderSorted[~l]].numpy()
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
            print(num2word[result], end="", flush=True)
        states[0].append(funcs[2](rs))
        i = 0
        while len(states) != i:
            ss = states[i]
            if len(ss) == maxLen:
                state = funcs[4]((positionalEncodingInput, collectArray(ss)))
                if len(states) == i + 1:
                    states.append([])
                states[i + 1].append(funcs[3](state))
                states[i] = []
            else:
                p = (
                    funcs[3](
                        collectArray(
                            states[i - 1] + [zeroPad] * (maxLen - len(states[i - 1]))
                        )
                    )
                    if len(states) != 1
                    else zeroPad
                )
                state = funcs[4](
                    (
                        positionalEncodingInput,
                        collectArray(ss + [p] + [zeroPad] * (maxLen - len(ss) - 1)),
                    )
                )
            i += 1


trainableVariables = (
    models[0].trainable_variables
    + models[1].trainable_variables
    + models[2].trainable_variables
    + models[3].trainable_variables
    + models[4].trainable_variables
    + models[5].trainable_variables
)


def loader():
    while True:
        input = []
        output = []
        for _ in range(batchSize):
            startIndex = random.randint(0, len(tokens) - maxLen ** (numRecur + 1) - 1)
            input.append(tokens[startIndex : startIndex + maxLen ** (numRecur + 1)])
            output.append(
                tokens[startIndex + 1 : startIndex + maxLen ** (numRecur + 1) + 1]
            )
        yield (
            tf.constant(input),
            tf.constant(output),
        )


def train_step(optimizer, data):
    xs, ys = data
    state = zeroState
    states = [[]]
    startOuts = []
    maskOuts = []
    stateIns = []
    attnOuts = []
    bridgeOuts = []
    for j in range(maxLen ** (numRecur - 1)):
        inputs = xs[:, j * maxLen**2 : (j + 1) * maxLen**2]
        stateIns.append(state)
        r, m = funcs[0](
            (
                positionalEncodingInput,
                inputs,
            )
        )
        m = m[:, :, tf.newaxis]
        startOuts.append(r)
        maskOuts.append(m)
        rs = funcs[1]((r, m, state))
        attnOuts.append(rs)
        r = funcs[2](rs)
        bridgeOuts.append(r)
        states[0].append(r)
        i = 0
        while len(states) != i:
            ss = states[i]
            if len(ss) == maxLen:
                state = funcs[4]((positionalEncodingInput, collectArray(ss)))
                if len(states) == i + 1:
                    states.append([])
                states[i + 1].append(funcs[3](state))
                states[i] = []
            else:
                p = (
                    funcs[3](
                        collectArray(
                            states[i - 1] + [zeroPad] * (maxLen - len(states[i - 1]))
                        )
                    )
                    if len(states) != 1
                    else zeroPad
                )
                state = funcs[4](
                    (
                        positionalEncodingInput,
                        collectArray(ss + [p] + [zeroPad] * (maxLen - len(ss) - 1)),
                    )
                )
            i += 1
    loss = []
    totalGrads = [tf.zeros_like(m) for m in trainableVariables]
    attnGrads = []
    bridgeGrads = [0 for _ in range(maxLen ** (numRecur - 1))]
    for j, out in enumerate(attnOuts):
        with tf.GradientTape() as tape:
            tape.watch(out)
            r = funcs[5](out)
            r = tf.keras.losses.sparse_categorical_crossentropy(
                ys[:, j * maxLen**2 : (j + 1) * maxLen**2], r
            )
        grads, nextGrads = tape.gradient(r, (trainableVariables, out), None, "zero")
        loss.append(r)
        totalGrads = [g + gt for g, gt in zip(totalGrads, grads)]
        attnGrads.append(nextGrads)
    nextGrads = 0
    for ij in range(maxLen ** (numRecur - 1) - 1):
        j = maxLen ** (numRecur - 1) - ij - 1
        inputs = xs[:, j * maxLen**2 : (j + 1) * maxLen**2]
        with tf.GradientTape() as tape:
            tape.watch(stateIns[j])
            r, m = funcs[0](
                (
                    positionalEncodingInput,
                    inputs,
                )
            )
            r = funcs[1]((r, m, stateIns[j]))
        grads, nextGrads = tape.gradient(
            r, (trainableVariables, stateIns[j]), attnGrads[j] + nextGrads, "zero"
        )
        totalGrads = [g + gt for g, gt in zip(totalGrads, grads)]
        with tf.GradientTape() as tape:
            outs = bridgeOuts[:j]
            for m in outs:
                tape.watch(m)
            while len(outs) != 1:
                newOuts = []
                for i in range(len(outs) // maxLen + 1):
                    r = outs[i * maxLen : (i + 1) * maxLen]
                    if len(r) == 0:
                        break
                    r = collectArray(r + [zeroPad] * (maxLen - len(r)))
                    rs = funcs[4]((positionalEncodingInput, r))
                    r = funcs[3](rs)
                    newOuts.append(r)
                outs = newOuts
        grads, nextGrads = tape.gradient(
            rs, (trainableVariables, bridgeOuts[:j]), nextGrads, "zero"
        )
        totalGrads = [g + gt for g, gt in zip(totalGrads, grads)]
        for i, nextGrad in enumerate(nextGrads):
            bridgeGrads[i] += nextGrad
        with tf.GradientTape() as tape:
            tape.watch(attnOuts[j - 1])
            r = funcs[2](attnOuts[j - 1])
        grads, nextGrads = tape.gradient(
            r, (trainableVariables, attnOuts[j - 1]), bridgeGrads[j - 1], "zero"
        )
        totalGrads = [g + gt for g, gt in zip(totalGrads, grads)]
    inputs = xs[:, 0 : maxLen**2]
    with tf.GradientTape() as tape:
        tape.watch(stateIns[0])
        r, m = funcs[0](
            (
                positionalEncodingInput,
                inputs,
            )
        )
        r = funcs[1]((r, m, stateIns[0]))
    grads = tape.gradient(r, trainableVariables, attnGrads[0] + nextGrads, "zero")
    totalGrads = [g + gt for g, gt in zip(totalGrads, grads)]
    optimizer.apply_gradients(zip(totalGrads, trainableVariables))
    return tf.reduce_mean(loss)


optimizer = tf.keras.optimizers.Adadelta(1.0)
optimizer.apply_gradients(
    zip([tf.zeros_like(m) for m in trainableVariables], trainableVariables)
)
with open("./weights/optimizer", "rb") as f:
    weights = pickle.load(f)
optimizer.set_weights(weights)
models[0].load_weights("./weights/start")
models[1].load_weights("./weights/attn")
models[2].load_weights("./weights/bridge")
models[3].load_weights("./weights/conv")
models[4].load_weights("./weights/collector")
models[5].load_weights("./weights/out")

if toTrain:
    trainDatas = loader()
    step = 0
    while True:
        print("step:", step, ",loss:", train_step(optimizer, next(trainDatas)).numpy())
        step += 1
        if step % 10 == 0:
            models[0].save_weights("./weights/start")
            models[1].save_weights("./weights/attn")
            models[2].save_weights("./weights/bridge")
            models[3].save_weights("./weights/conv")
            models[4].save_weights("./weights/collector")
            models[5].save_weights("./weights/out")
            with open("./weights/optimizer", "wb") as f:
                pickle.dump(optimizer.get_weights(), f)
else:
    predict()
