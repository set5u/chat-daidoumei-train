import json
import tensorflow as tf
import numpy as np
import math
import random
import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model

policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)
toTrain = True


def save(model):
    weights = model.get_weights()
    ret = []
    for weight in weights:
        ret.append(json.dumps(weight.tolist()))
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
    return tf.constant(ret, "float16")


def load(weights: str):
    weights = weights.split("\n")
    ret = []
    for weight in weights:
        ret.append(np.array(json.loads(weight), "float16"))
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
    positionalEncodingInput = tf.keras.Input((maxLen, dModel))
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
        encoderLayerInput = tf.keras.Input((maxLen, dModel))
        encoderMaskInput = tf.keras.Input((maxLen))
        encoderMultiHeadAttention = tf.keras.layers.MultiHeadAttention(h, dModel)(
            encoderLayerInput, encoderLayerInput, attention_mask=encoderMaskInput
        )
        encoderAdd0 = tf.keras.layers.Add()(
            (encoderLayerInput, encoderMultiHeadAttention)
        )
        encoderNorm0 = tf.keras.layers.LayerNormalization()(encoderAdd0)
        encoderDropout0 = tf.keras.layers.Dropout(pDropout)(encoderNorm0)
        encoderDense0 = tf.keras.layers.Dense(dFF)(encoderDropout0)
        encoderDense1 = tf.keras.layers.Dense(dModel)(encoderDense0)
        encoderAdd1 = tf.keras.layers.Add()((encoderDense1, encoderDropout0))
        encoderNorm1 = tf.keras.layers.LayerNormalization()(encoderAdd1)
        encoderDropout1 = tf.keras.layers.Dropout(pDropout)(encoderNorm1)
        encoderStateInput = tf.keras.Input((maxLen, dModel))
        encoderAdd2 = tf.keras.layers.Add()((encoderStateInput, encoderDropout1))
        encoderNorm2 = tf.keras.layers.LayerNormalization()(encoderAdd2)
        encoderActivation = tf.keras.layers.Activation("tanh")(encoderNorm2)
        encoderDropout2 = tf.keras.layers.Dropout(pDropout)(encoderActivation)
        encoders.append(
            tf.keras.Model(
                (encoderLayerInput, encoderMaskInput, encoderStateInput),
                encoderDropout2,
            )
        )
    encoderEndInput = tf.keras.Input((maxLen**2, dModel))
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
        decoderLayerInput = tf.keras.Input((maxLen, dModel))
        decoderEncoderInput = tf.keras.Input((maxLen, dModel))
        decoderMaskInput = tf.keras.Input((maxLen))
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
        decoderDense0 = tf.keras.layers.Dense(dFF)(decoderDropout1)
        decoderDense1 = tf.keras.layers.Dense(dModel)(decoderDense0)
        decoderAdd2 = tf.keras.layers.Add()((decoderDense1, decoderDropout1))
        decoderNorm2 = tf.keras.layers.LayerNormalization()(decoderAdd2)
        decoderDropout2 = tf.keras.layers.Dropout(pDropout)(decoderNorm2)
        decoderStateInput = tf.keras.Input((maxLen, dModel))
        decoderAdd3 = tf.keras.layers.Add()((decoderStateInput, decoderDropout2))
        decoderNorm3 = tf.keras.layers.LayerNormalization()(decoderAdd3)
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
    decoderEndInput = tf.keras.Input((maxLen, dModel))
    decoderEndDense = tf.keras.layers.Dense(depthTarget)(decoderEndInput)
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

dModel = 16
dFF = 32
layers = 4
h = 4
models = useExtendedTransformer(
    dModel,
    dFF,
    0.1,
    h,
    maxLen,
    depth,
    depth,
    depth,
    layers,
)
tf.keras.utils.plot_model(models["encoderStart"], "encoderStart.png", show_shapes=True)
tf.keras.utils.plot_model(models["encoders"][0], "encoder.png", show_shapes=True)
tf.keras.utils.plot_model(models["encoderEnd"], "encoderEnd.png", show_shapes=True)
tf.keras.utils.plot_model(models["decoderStart"], "decoderStart.png", show_shapes=True)
tf.keras.utils.plot_model(models["decoders"][0], "decoder.png", show_shapes=True)
tf.keras.utils.plot_model(models["decoderEnd"], "decoderEnd.png", show_shapes=True)


def loader():
    while True:
        input = []
        output = []
        input2 = []
        for _ in range(16):
            startIndex = math.floor(random.random() * (len(tokens) - (16 + 16)))
            input.append(tokens[startIndex : startIndex + 16])
            endIndex = startIndex + 16
            out = []
            while len(out) != 16:
                if tokens[endIndex] != 3:
                    out.append(tokens[endIndex])
                endIndex += 1
            output.append(out)
            input2.append([0] + out[:-1])

        yield (
            np.array(input).reshape((16, -1, 8)),
            np.array(input2).reshape((16, -1, 8)),
        ), np.array(output).reshape((16, -1, 8))
