import json
import pickle
import tensorflow as tf
import numpy as np
import math
import random
import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model

# policy = tf.keras.mixed_precision.Policy("mixed_float16")
# tf.keras.mixed_precision.set_global_policy(policy)
toTrain = True

dtype = "float32"


def save(weights):
    ret = []
    for weight in weights:
        ret.append(json.dumps(weight.value().tolist()))
    return "\n".join(ret)


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


def load(weights: str):
    weights = weights.split("\n")
    ret = []
    for weight in weights:
        ret.append(np.array(json.loads(weight), "float32"))
    return ret


def useExtendedTransformer(
    dModel,
    dFF,
    pDropout,
    h,
    maxLen,
    depthEncoder,
    depthDecoder,
    depthTarget,
    layers,
):
    positionalEncodingInput = tf.keras.Input((maxLen, dModel), dtype=dtype)
    encoderInput = tf.keras.Input((maxLen,))
    encoderOnes = tf.keras.layers.Dense(
        units=maxLen,
        kernel_initializer="zeros",
        bias_initializer="ones",
        trainable=False,
    )(encoderInput)
    encoderAttentionMask = tf.keras.layers.Minimum()([encoderInput, encoderOnes])
    encoderEmbedding = tf.keras.layers.Embedding(
        input_dim=depthEncoder,
        output_dim=dModel,
        mask_zero=True,
    )(encoderInput)
    encoderPositionalEncoding = tf.keras.layers.Add()(
        (encoderEmbedding, positionalEncodingInput)
    )
    encoderStart = tf.keras.Model(
        (positionalEncodingInput, encoderInput),
        (encoderPositionalEncoding, encoderAttentionMask),
    )
    encoders = []
    for i in range(layers):
        encoderLayerInput = tf.keras.Input((maxLen, dModel), dtype=dtype)
        encoderMaskInput = tf.keras.Input((maxLen, 1))
        encoderMultiHeadAttention = tf.keras.layers.MultiHeadAttention(h, dModel)(
            encoderLayerInput, encoderLayerInput, attention_mask=encoderMaskInput
        )
        encoderAdd0 = tf.keras.layers.Add()(
            (encoderLayerInput, encoderMultiHeadAttention)
        )
        encoderNorm0 = tf.keras.layers.LayerNormalization()(encoderAdd0)
        encoderDropout0 = tf.keras.layers.Dropout(pDropout)(encoderNorm0)
        encoderDense0 = tf.keras.layers.EinsumDense(
            "abc,bcd->abd", (maxLen, dFF), "relu"
        )(encoderDropout0)
        encoderDense1 = tf.keras.layers.EinsumDense(
            "abd,bcd->abc", (maxLen, dModel), "linear"
        )(encoderDense0)
        encoderAdd1 = tf.keras.layers.Add()((encoderDense1, encoderDropout0))
        encoderNorm1 = tf.keras.layers.LayerNormalization()(encoderAdd1)
        encoderDropout1 = tf.keras.layers.Dropout(pDropout)(encoderNorm1)
        encoderStateInput = tf.keras.Input((maxLen, dModel), dtype=dtype)
        encoderMean2 = tf.keras.layers.Average()((encoderStateInput, encoderDropout1))
        encoderNorm2 = tf.keras.layers.LayerNormalization()(encoderMean2)
        encoderActivation = tf.keras.layers.Activation("tanh")(encoderNorm2)
        encoderDropout2 = tf.keras.layers.Dropout(pDropout)(encoderActivation)
        encoders.append(
            tf.keras.Model(
                (encoderLayerInput, encoderMaskInput, encoderStateInput),
                encoderDropout2,
            )
        )
    encoderEndInput = tf.keras.Input((maxLen**2, dModel), dtype=dtype)
    encoderEndAttn = tf.keras.layers.MultiHeadAttention(h, dModel)(
        encoderEndInput, encoderEndInput
    )
    encoderEndActivation = tf.keras.layers.Activation("tanh")(encoderEndAttn)
    encoderEndDropout0 = tf.keras.layers.Dropout(pDropout)(encoderEndActivation)
    encoderEndPermute0 = tf.keras.layers.Permute((2, 1))(encoderEndDropout0)
    encoderEndConv = tf.keras.layers.Conv1D(maxLen, 1)(encoderEndPermute0)
    encoderEndPermute1 = tf.keras.layers.Permute((2, 1))(encoderEndConv)
    encoderEndDropout1 = tf.keras.layers.Dropout(pDropout)(encoderEndPermute1)
    encoderEnd = tf.keras.Model(encoderEndInput, encoderEndDropout1)

    decoderInput = tf.keras.Input((maxLen,))
    decoderOnes = tf.keras.layers.Dense(
        units=maxLen,
        kernel_initializer="zeros",
        bias_initializer="ones",
        trainable=False,
    )(decoderInput)
    decoderAttentionMask = tf.keras.layers.Minimum()((decoderInput, decoderOnes))
    decoderEmbedding = tf.keras.layers.Embedding(
        input_dim=depthDecoder,
        output_dim=dModel,
        mask_zero=True,
    )(decoderInput)
    decoderPositionalEncoding = tf.keras.layers.Add()(
        (decoderEmbedding, positionalEncodingInput)
    )
    decoderStart = tf.keras.Model(
        (positionalEncodingInput, decoderInput),
        (decoderPositionalEncoding, decoderAttentionMask),
    )
    decoders = []
    for i in range(layers):
        decoderLayerInput = tf.keras.Input((maxLen, dModel), dtype=dtype)
        decoderEncoderInput = tf.keras.Input((maxLen, dModel), dtype=dtype)
        decoderMaskInput = tf.keras.Input((maxLen, 1))
        decoderMultiHeadAttention0 = tf.keras.layers.MultiHeadAttention(h, dModel)(
            decoderLayerInput,
            decoderLayerInput,
            use_causal_mask=True,
            attention_mask=decoderMaskInput,
        )
        decoderAdd0 = tf.keras.layers.Add()(
            (decoderLayerInput, decoderMultiHeadAttention0)
        )
        decoderNorm0 = tf.keras.layers.LayerNormalization()(decoderAdd0)
        decoderDropout0 = tf.keras.layers.Dropout(pDropout)(decoderNorm0)
        decoderMultiHeadAttention1 = tf.keras.layers.MultiHeadAttention(h, dModel)(
            decoderDropout0, decoderEncoderInput, attention_mask=decoderMaskInput
        )
        decoderAdd1 = tf.keras.layers.Add()(
            (decoderDropout0, decoderMultiHeadAttention1)
        )
        decoderNorm1 = tf.keras.layers.LayerNormalization()(decoderAdd1)
        decoderDropout1 = tf.keras.layers.Dropout(pDropout)(decoderNorm1)
        decoderDense0 = tf.keras.layers.EinsumDense(
            "abc,bcd->abd", (maxLen, dFF), "relu"
        )(decoderDropout1)
        decoderDense1 = tf.keras.layers.EinsumDense(
            "abd,bcd->abc", (maxLen, dModel), "linear"
        )(decoderDense0)
        decoderAdd2 = tf.keras.layers.Add()((decoderDense1, decoderDropout1))
        decoderNorm2 = tf.keras.layers.LayerNormalization()(decoderAdd2)
        decoderDropout2 = tf.keras.layers.Dropout(pDropout)(decoderNorm2)
        decoderStateInput = tf.keras.Input((maxLen, dModel), dtype=dtype)
        decoderMean3 = tf.keras.layers.Average()((decoderStateInput, decoderDropout2))
        decoderNorm3 = tf.keras.layers.LayerNormalization()(decoderMean3)
        decoderActivation = tf.keras.layers.Activation("tanh")(decoderNorm3)
        decoderDropout3 = tf.keras.layers.Dropout(pDropout)(decoderActivation)
        decoders.append(
            tf.keras.Model(
                (
                    decoderLayerInput,
                    decoderMaskInput,
                    decoderEncoderInput,
                    decoderStateInput,
                ),
                decoderDropout3,
            )
        )
    decoderEndInput = tf.keras.Input((maxLen, dModel), dtype=dtype)
    decoderEndDense = tf.keras.layers.Dense(depthTarget, "softmax")(decoderEndInput)
    decoderEnd = tf.keras.Model(decoderEndInput, decoderEndDense)
    return {
        "encoderStart": encoderStart,
        "encoders": encoders,
        "encoderEnd": encoderEnd,
        "decoderStart": decoderStart,
        "decoders": decoders,
        "decoderEnd": decoderEnd,
    }


with open("./num2word.json", "r", -1, "utf-8") as f:
    num2word = json.loads("".join(f.readlines()))
with open("./word2num.json", "r", -1, "utf-8") as f:
    word2num = json.loads("".join(f.readlines()))
with open("./wordTokens.json", "r", -1, "utf-8") as f:
    tokens = json.loads("".join(f.readlines()))
depth = len(num2word)
maxLen = 8
# params = 171,957,328
dModel = 256
dFF = 512
layers = 16
h = 8
numRecur = 4
models = useExtendedTransformer(
    dModel,
    dFF,
    0.2,
    h,
    maxLen,
    depth,
    depth,
    depth,
    layers,
)
models["encoderStart"].summary()
models["encoders"][0].summary()
models["encoderEnd"].summary()
models["decoderStart"].summary()
models["decoders"][0].summary()
models["decoderEnd"].summary()
tf.keras.utils.plot_model(models["encoderStart"], "encoderStart.png", show_shapes=True)
tf.keras.utils.plot_model(models["encoders"][0], "encoder.png", show_shapes=True)
tf.keras.utils.plot_model(models["encoderEnd"], "encoderEnd.png", show_shapes=True)
tf.keras.utils.plot_model(models["decoderStart"], "decoderStart.png", show_shapes=True)
tf.keras.utils.plot_model(models["decoders"][0], "decoder.png", show_shapes=True)
tf.keras.utils.plot_model(models["decoderEnd"], "decoderEnd.png", show_shapes=True)
funcs = {}


funcs["encoderStart"] = tf.function(
    lambda xs, **kwargs: models["encoderStart"](xs, **kwargs),
    jit_compile=True,
    reduce_retracing=True,
)
funcs["encoders"] = [
    tf.function(
        lambda xs, **kwargs: models["encoders"][i](xs, **kwargs),
        jit_compile=True,
        reduce_retracing=True,
    )
    for i in range(layers)
]
funcs["encoderEnd"] = tf.function(
    lambda xs, **kwargs: models["encoderEnd"](xs, **kwargs),
    jit_compile=True,
    reduce_retracing=True,
)
funcs["decoderStart"] = tf.function(
    lambda xs, **kwargs: models["decoderStart"](xs, **kwargs),
    jit_compile=True,
    reduce_retracing=True,
)
funcs["decoders"] = [
    tf.function(
        lambda xs, **kwargs: models["decoders"][i](xs, **kwargs),
        jit_compile=True,
        reduce_retracing=True,
    )
    for i in range(layers)
]
funcs["decoderEnd"] = tf.function(
    lambda xs, **kwargs: models["decoderEnd"](xs, **kwargs),
    jit_compile=True,
    reduce_retracing=True,
)


batchSize = 256 if toTrain else 1


def predict():
    zeroState = tf.constant(
        [[[0.0 for _ in range(dModel)] for _ in range(maxLen)]], dtype=dtype
    )
    states = [[]]

    encoderInput = []
    positionalEncodingInput = positionalEncoding(maxLen, dModel)[tf.newaxis, :, :]
    encoderStates = [zeroState for _ in range(layers)]
    while True:
        once = True
        while once or len(encoderInput) > maxLen:
            e, eMask = funcs["encoderStart"](
                (
                    positionalEncodingInput,
                    tf.constant(
                        [[0] * (8 - len(encoderInput)) + encoderInput[:maxLen]]
                    ),
                )
            )
            for i, state in enumerate(encoderStates):
                e = funcs["encoders"][i]((e, eMask[:, :, tf.newaxis], state))
                if len(encoderInput) > maxLen:
                    encoderInput = encoderInput[maxLen:]
                    encoderStates[i] = e[:, ::-1]
            states[0].append(e[:, ::-1])
            for i, state in enumerate(states):
                if len(state) == maxLen:
                    if len(states) == i + 1:
                        states.append([])
                    states[i + 1].append(
                        funcs["encoderEnd"](
                            tf.reshape(
                                tf.transpose(state, (1, 0, 2, 3)),
                                (1, maxLen**2, dModel),
                            )
                        )[:, ::-1]
                    )
                    states[i] = []
            currentState = zeroState
            for i, state in enumerate(states):
                currentState = funcs["encoderEnd"](
                    tf.reshape(
                        tf.transpose(
                            [zeroState] * (7 - len(state)) + [currentState] + state,
                            (1, 0, 2, 3),
                        ),
                        (1, maxLen**2, dModel),
                    )[:, ::-1]
                )
            once = False
        decoderInput = []
        decoderStates = [zeroState for _ in range(layers)]
        while True:
            d, dMask = funcs["decoderStart"](
                (
                    positionalEncodingInput,
                    tf.constant([[0] * (8 - len(decoderInput)) + decoderInput]),
                )
            )
            for i, state in enumerate(decoderStates):
                d = funcs["decoders"][i](
                    (d, dMask[:, :, tf.newaxis], currentState, state)
                )
                if len(decoderInput) == maxLen:
                    decoderInput = []
                    decoderStates[i] = d[:, ::-1]
            decoderOut = funcs["decoderEnd"](d[:, ::-1])
            decoderSorted = tf.argsort(decoderOut[0][(len(decoderInput) + 1) % maxLen])
            results = []
            sum = 0
            for l in range(5):
                c = decoderOut[0][(len(decoderInput) + 1) % maxLen][
                    decoderSorted[~l]
                ].numpy()
                sum += c
                results.append(c)
            r = random.random() * sum
            t = 0
            m = 0
            while t < r:
                t += results[m]
                m += 1
            result = decoderSorted[~m + 1].numpy()
            print(num2word[result], end="\n" if result == 1 else "", flush=True)
            if result == 1:
                encoderInput.extend(decoderInput)
                break
            decoderInput.append(result)


encoderRecurrentCount = 3
encoderLength = maxLen**encoderRecurrentCount
decoderLength = 256


def loader():
    while True:
        input = []
        output = []
        input2 = []
        for _ in range(batchSize):
            startIndex = math.floor(
                random.random() * (len(tokens) - (encoderLength + decoderLength * 2))
            )
            input.append(tokens[startIndex : startIndex + encoderLength])
            endIndex = startIndex + encoderLength
            out = []
            while len(out) != decoderLength:
                if tokens[endIndex] != 3:
                    out.append(tokens[endIndex])
                endIndex += 1
            output.append(out)
            input2.append([0] + out[:-1])

        yield (
            np.array(input).reshape((batchSize, -1, 8)),
            np.array(input2).reshape((batchSize, -1, 8)),
        ), np.array(output).reshape((batchSize, -1, 8))


trainableVariables = (
    models["encoderStart"].trainable_variables
    + sum([m.trainable_variables for m in models["encoders"]], [])
    + models["encoderEnd"].trainable_variables
    + models["decoderStart"].trainable_variables
    + sum([m.trainable_variables for m in models["decoders"]], [])
    + models["decoderEnd"].trainable_variables
)


# @tf.function(jit_compile=True)
def decoderEndBack(decodersOut, ys, i):
    with tf.GradientTape() as tape:
        tape.watch(decodersOut[i])
        d = funcs["decoderEnd"](decodersOut[i], training=True)
        d = tf.keras.losses.sparse_categorical_crossentropy(ys[:, i], d)
    loss = tf.reduce_mean(d)
    return tape.gradient(
        d,
        (models["decoderEnd"].trainable_variables, decodersOut[i]),
        None,
        "zero",
    ) + (loss,)


# @tf.function(jit_compile=True)
def decodersBack(
    encoderEndOut, decoderStartOuts, decodersStates, decoderEndNextGrads, i, j
):
    with tf.GradientTape() as tape:
        tape.watch(encoderEndOut)
        decodersIn = decoderStartOuts[decoderLength // maxLen - i - 1][0]
        tape.watch(decodersIn)
        for j in range(layers):
            d = funcs["decoders"][j](
                (
                    decodersIn,
                    decoderStartOuts[decoderLength // maxLen - i - 1][1],
                    encoderEndOut,
                    decodersStates[decoderLength // maxLen - i - 1][j],
                )
            )
            decodersIn = d
    return tape.gradient(
        d,
        (
            [m.trainable_variables for m in models["decoders"]],
            decodersIn,
            encoderEndOut,
        ),
        decoderEndNextGrads[decoderLength // maxLen - i - 1],
        "zero",
    )


# @tf.function(jit_compile=True)
def decoderStartBack(positionalEncodingInput, dx, i, decodersNextGrads):
    with tf.GradientTape() as tape:
        d, _ = funcs["decoderStart"](
            (
                positionalEncodingInput,
                dx[:, i],
            ),
            training=True,
        )
    return tape.gradient(
        d, models["decoderStart"].trainable_variables, decodersNextGrads[i], "zero"
    )


# @tf.function(jit_compile=True)
def encoderEndBack(encoderEndIns, encoderEndNextGrads, i, j):
    with tf.GradientTape() as tape:
        tape.watch(encoderEndIns[i][j])
        d = funcs["encoderEnd"](encoderEndIns[i][j], training=True)
    return tape.gradient(
        d,
        (models["encoderEnd"].trainable_variables, encoderEndIns[i][j]),
        encoderEndNextGrads[j],
        "zero",
    )


# @tf.function(jit_compile=True)
def encodersBack(
    encoderEndOut, encoderStartOuts, encodersStates, encoderEndNextGrads, i, j
):
    with tf.GradientTape() as tape:
        tape.watch(encoderEndOut)
        encodersIn = encoderStartOuts[encoderLength // maxLen - i - 1][0]
        tape.watch(encodersIn)
        for j in range(layers):
            d = funcs["encoders"][j](
                (
                    encodersIn,
                    encoderStartOuts[encoderLength // maxLen - i - 1][1],
                    encodersStates[encoderLength // maxLen - i - 1][j],
                )
            )
            encodersIn = d
    return tape.gradient(
        d,
        (
            [m.trainable_variables for m in models["encoders"]],
            encodersIn,
        ),
        encoderEndNextGrads[encoderLength // maxLen - i - 1],
        "zero",
    )


# @tf.function(jit_compile=True)
def encoderStartBack(positionalEncodingInput, encodersNextGrads, ex, i):
    with tf.GradientTape() as tape:
        d, _ = funcs["encoderStart"](
            (
                positionalEncodingInput,
                ex[:, i],
            ),
            training=True,
        )
    return tape.gradient(
        d, models["encoderStart"].trainable_variables, encodersNextGrads[i], "zero"
    )


def train_step(optimizer, data):
    positionalEncodingInput = positionalEncoding(maxLen, dModel)[tf.newaxis, :, :]
    zeroState = tf.constant(
        [[[0.0 for _ in range(dModel)] for _ in range(maxLen)]], dtype=dtype
    )
    xs = data[0]
    ex = xs[0]
    dx = xs[1]
    ys = data[1]
    # fore
    encoderStartOuts = []  # ((numRecur,B,maxLen,dModel),(numRecur,B,maxLen))[]
    for i in range(encoderLength // maxLen):
        encoderStartOuts.append(
            funcs["encoderStart"]((positionalEncodingInput, ex[:, i]), training=True)
        )
    encodersStates = [
        [zeroState for _ in range(layers)]
    ]  # (numRecur,layers,B,maxLen,dModel)
    encodersOut = [o[0] for o in encoderStartOuts]  # (numRecur,B,maxLen,dModel)
    encoderStates = [zeroState for _ in range(layers)]  # (layers,B,maxLen,dModel)
    for i in range(encoderLength // maxLen):
        encodersStates.append([])
        for j in range(layers):
            out = funcs["encoders"][j](
                (encodersOut[i], encoderStartOuts[i][1], encoderStates[j])
            )
            encodersOut[i] = out
            encoderStates[j] = out[:, ::-1]
            encodersStates[i].append(out[:, ::-1])
    encoderEndIn = tf.reshape(
        tf.transpose(encodersOut, (1, 0, 2, 3)),
        (-1, encoderLength // (maxLen**2), maxLen**2, dModel),
    )  # (B,numRecur//maxLen,maxLen**2,dModel)
    encoderEndOut = encoderEndIn  # (B,numRecur//maxLen,maxLen**2,dModel)
    encoderEndIns = []  # (numRecur,B,maxLen**2,dModel)[]
    for i in range(encoderRecurrentCount - 1):
        encoderEndIns.append([])
        temp = []  # (numRecur//maxLen,B,maxLen,dModel)
        for j in range(maxLen ** (encoderRecurrentCount - i - 1)):
            encoderEndIns[i].append(encoderEndOut[:, i, :, :])
            temp.append(funcs["encoderEnd"](encoderEndOut[:, i, :, :], training=True))
        encoderEndOut = tf.reshape(
            tf.transpose(temp, (1, 0, 2, 3)),
            (batchSize, -1, maxLen**2, dModel),
        )
    encoderEndIns.append([encoderEndOut[:, 0, :, :]])
    encoderEndOut = funcs["encoderEnd"](
        encoderEndOut[:, 0, :, :], training=True
    )  # (B,maxLen,dModel)
    decoderStartOuts = []  # ((numRecur,B,maxLen,dModel),(numRecur,B,maxLen))[]
    for i in range(decoderLength // maxLen):
        decoderStartOuts.append(
            funcs["decoderStart"]((positionalEncodingInput, dx[:, i]), training=True)
        )
    decodersStates = [
        [zeroState for _ in range(layers)]
    ]  # (numRecur,layers,B,maxLen,dModel)
    decodersOut = [o[0] for o in decoderStartOuts]  # (numRecur,B,maxLen,dModel)
    decoderStates = [zeroState for _ in range(layers)]  # (layers,B,maxLen,dModel)
    for i in range(decoderLength // maxLen):
        decodersStates.append([])
        for j in range(layers):
            out = funcs["decoders"][j](
                (
                    decodersOut[i],
                    decoderStartOuts[i][1],
                    encoderEndOut,
                    decoderStates[j],
                )
            )
            decodersOut[i] = out
            decoderStates[j] = out[:, ::-1]
            decodersStates[i].append(out[:, ::-1])
    # back
    decoderEndGrads = [
        tf.zeros_like(m) for m in models["decoderEnd"].trainable_variables
    ]
    decoderEndNextGrads = []
    ret = 0
    for i in range(decoderLength // maxLen):
        grads, nextGrad, loss = decoderEndBack(decodersOut, ys, i)
        ret += loss
        decoderEndNextGrads.append(nextGrad)
        decoderEndGrads = [gs + g for gs, g in zip(decoderEndGrads, grads)]
    decodersGrads = [
        [tf.zeros_like(m) for m in models["decoders"][i].trainable_variables]
        for i in range(layers)
    ]
    decodersNextGrads = []
    encoderGrads = 0
    for i in range(decoderLength // maxLen):
        grads, nextGrad, eGrad = decodersBack(
            encoderEndOut, decoderStartOuts, decodersStates, decoderEndNextGrads, i, j
        )
        for j in range(layers):
            decodersGrads[j] = [gs + g for gs, g in zip(decodersGrads[j], grads[j])]
        encoderGrads += eGrad
        decodersNextGrads.append(nextGrad)
    decoderStartGrads = [
        tf.zeros_like(m) for m in models["decoderStart"].trainable_variables
    ]
    for i in range(decoderLength // maxLen):
        grads = decoderStartBack(positionalEncodingInput, dx, i, decodersNextGrads)
        decoderStartGrads = [gs + g for gs, g in zip(decoderStartGrads, grads)]
    encoderEndGrads = [
        tf.zeros_like(m) for m in models["encoderEnd"].trainable_variables
    ]
    encoderEndNextGrads = [encoderGrads]  # (1,B,maxLen,dModel)
    for ii in range(encoderRecurrentCount):
        i = encoderRecurrentCount - ii - 1
        newEncoderEndNextGrads = []  # (numRecur,B,maxLen**2,dModel)
        for j in range(maxLen**ii):
            grads, nextGrad = encoderEndBack(encoderEndIns, encoderEndNextGrads, i, j)
            newEncoderEndNextGrads.append(nextGrad)
            encoderEndGrads = [gs + g for gs, g in zip(encoderEndGrads, grads)]
        encoderEndNextGrads = tf.transpose(
            tf.reshape(newEncoderEndNextGrads, (batchSize, -1, maxLen, dModel)),
            (1, 0, 2, 3),
        )
    encodersGrads = [
        [tf.zeros_like(m) for m in models["encoders"][i].trainable_variables]
        for i in range(layers)
    ]
    encodersNextGrads = []
    for i in range(encoderLength // maxLen):
        grads, nextGrad = encodersBack(
            encoderEndOut, encoderStartOuts, encodersStates, encoderEndNextGrads, i, j
        )
        for j in range(layers):
            encodersGrads[j] = [gs + g for gs, g in zip(encodersGrads[j], grads[j])]
        encodersNextGrads.append(nextGrad)
    encoderStartGrads = [
        tf.zeros_like(m) for m in models["encoderStart"].trainable_variables
    ]
    for i in range(encoderLength // maxLen):
        grads = encoderStartBack(positionalEncodingInput, encodersNextGrads, ex, i)
        encoderStartGrads = [gs + g for gs, g in zip(encoderStartGrads, grads)]
    optimizer.apply_gradients(
        zip(
            encoderStartGrads
            + sum(encodersGrads, [])
            + encoderEndGrads
            + decoderStartGrads
            + sum(decodersGrads, [])
            + decoderEndGrads,
            trainableVariables,
        )
    )
    return ret / (decoderLength // maxLen)


optimizer = tf.keras.optimizers.Adadelta(1.0)
optimizer.apply_gradients(
    zip([tf.zeros_like(m) for m in trainableVariables], trainableVariables)
)
with open("./weights/optimizer", "rb") as f:
    weights = pickle.load(f)
optimizer.set_weights(weights)
models["encoderStart"].load_weights("./weights/encoderStart")
for i in range(layers):
    models["encoders"][i].load_weights("./weights/encoder" + str(i))
models["encoderEnd"].load_weights("./weights/encoderEnd")
models["decoderStart"].load_weights("./weights/decoderStart")
for i in range(layers):
    models["decoders"][i].load_weights("./weights/decoder" + str(i))
models["decoderEnd"].load_weights("./weights/decoderEnd")


if toTrain:
    trainDatas = loader()
    step = 0
    while True:
        print("step:", step, ",loss:", train_step(optimizer, next(trainDatas)).numpy())
        step += 1
        if step % 10 == 0:
            models["encoderStart"].save_weights("./weights/encoderStart")
            for i in range(layers):
                models["encoders"][i].save_weights("./weights/encoder" + str(i))
            models["encoderEnd"].save_weights("./weights/encoderEnd")
            models["decoderStart"].save_weights("./weights/decoderStart")
            for i in range(layers):
                models["decoders"][i].save_weights("./weights/decoder" + str(i))
            models["decoderEnd"].save_weights("./weights/decoderEnd")
            weights = optimizer.get_weights()
            with open("./weights/optimizer", "wb") as f:
                pickle.dump(weights, f)
else:
    predict()
