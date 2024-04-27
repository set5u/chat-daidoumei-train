import json
import tensorflow as tf
import numpy as np
import math

batchSize = 16


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
    return np.array(ret)


class RNNTiler(tf.keras.Model):
    def call(self, *inputs):
        return tf.tile(inputs[0], (1, inputs[1].shape[1], 1))

    def compute_output_shape(self, inputShape):
        return inputShape[0][0:1] + (None,) + inputShape[0][2:]


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
            tf.keras.ops.tril(tf.ones((maxLen, maxLen)), 0)
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
        input_shape = input_shape[0:1] + input_shape[1:2] * 2 + input_shape[2:]
        self.ff0.build(input_shape)
        input_shape = self.ff0.compute_output_shape(input_shape)
        input_shape = input_shape[0:1] + input_shape[1:2] * 2 + input_shape[2:]
        self.ff1.build(input_shape)

    def call(self, *inputs):
        mask = self.mask
        input = tf.expand_dims(inputs[0][:, tf.newaxis], (1, self.maxLen, 1))
        ret = input * mask
        ret = self.ff0(ret)
        ret = tf.expand_dims(ret[:, tf.newaxis], (1, self.maxLen, 1))
        ret = ret * mask
        ret = self.ff1(ret)
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape


class EncoderLayer(tf.keras.Model):
    def __init__(self, dModel, dFF, pDropout, h, maxLen, depthEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dModel = dModel
        self.dFF = dFF
        self.pDropout = pDropout
        self.h = h
        self.maxLen = maxLen
        self.depthEncoder = depthEncoder
        self.attn = tf.keras.layers.MultiHeadAttention(h, dModel // h)
        self.norm1 = AddNorm()
        self.dropout1 = tf.keras.layers.Dropout(pDropout)
        self.ff = FF(dModel, dFF, maxLen)
        self.norm2 = AddNorm()
        self.dropout2 = tf.keras.layers.Dropout(pDropout)

    def build(self, input_shape):
        input_shape = input_shape[0:2] + (input_shape[2] - 1,)
        self.attn.build(input_shape, input_shape)
        self.norm1.build(input_shape)
        self.dropout1.build(input_shape)
        self.ff.build(input_shape)
        self.norm2.build(input_shape)
        self.dropout2.build(input_shape)

    def call(self, *inputs):
        input = inputs[0][:, :, :-1]
        mask = inputs[0][:, :, -1]
        ret = input
        ret = self.attn(input, input, attention_mask=mask)
        ret = self.norm1(ret, input)
        input = self.dropout1(ret)
        ret = self.ff(input)
        ret = self.norm2(ret, input)
        ret = self.dropout2(ret)
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape[0:2] + (input_shape[2] - 1,)


class AttentionRNNCell(tf.keras.Model):
    def __init__(self, h, keyDim, maxLen, use_causal_mask=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn = tf.keras.layers.MultiHeadAttention(h, keyDim)
        self.norm1 = AddNorm()
        self.sattn = tf.keras.layers.MultiHeadAttention(h, keyDim)
        self.norm2 = AddNorm()
        self.h = h
        self.keyDim = keyDim
        self.maxLen = maxLen
        self.use_causal_mask = use_causal_mask
        self.state_size = h * keyDim * maxLen

    def build(self, input_shape):
        input_shape = (input_shape[0],) + (self.maxLen, self.h * self.keyDim)
        self.attn.build(input_shape, input_shape)
        self.norm1.build(input_shape)
        self.sattn.build(input_shape, input_shape)
        self.norm2.build(input_shape)

    def call(self, *inputs):
        input0 = tf.reshape(inputs[0], (-1, self.maxLen, self.h * self.keyDim))
        input1 = tf.reshape(inputs[1], (-1, self.maxLen, self.h * self.keyDim))
        ret = self.attn(input0, input1, use_causal_mask=self.use_causal_mask)
        ret = self.norm1(ret, input0)
        state = self.sattn(input0, input1, use_causal_mask=self.use_causal_mask)
        state = self.norm2(state, input0)
        return [
            tf.reshape(ret, (-1, self.maxLen * self.h * self.keyDim)),
            tf.reshape(state, (-1, self.maxLen * self.h * self.keyDim)),
        ]

    def compute_output_shape(self, input_shape):
        return input_shape


class MiddleLayer(tf.keras.Model):
    def __init__(self, h, keyDim, maxLen, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dModel = h * keyDim
        self.dModelLen = dModel * maxLen
        self.reshape0 = tf.keras.layers.Reshape(target_shape=(-1, maxLen * dModel))
        self.rnn = tf.keras.layers.RNN(
            AttentionRNNCell(h, dModel // h, maxLen),
            return_sequences=True,
            return_state=True,
        )
        self.reshape1 = tf.keras.layers.Reshape(target_shape=(-1, maxLen, dModel))

    def build(self, input_shape):
        computed = input_shape[0:2] + (self.dModelLen,)
        self.rnn.build(computed)

    def call(self, *inputs):
        input = inputs[0]
        initialState = inputs[1] if len(inputs) == 2 else None
        ret = self.reshape0(input)
        ret, state = self.rnn(ret, initial_state=initialState)
        ret = self.reshape1(ret)
        return ret, state

    def compute_output_shape(self, input_shapes):
        return input_shapes[0], input_shapes[0][:2] + (self.dModelLen,)


encoderRecurrent = None
decoderRecurrent = None


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
    encoderInput = tf.keras.Input(shape=[encoderRecurrent, maxLen])
    encoderOnes = tf.keras.layers.Dense(
        units=maxLen,
        kernel_initializer="zeros",
        bias_initializer="ones",
        trainable=False,
    )(encoderInput)
    encoderAttentionMask = tf.keras.layers.Minimum()([encoderInput, encoderOnes])
    encoderEmbedding = tf.keras.layers.TimeDistributed(
        layer=tf.keras.layers.Embedding(
            input_dim=depthEncoder,
            output_dim=dModel,
            mask_zero=True,
        ),
    )(encoderInput)
    encoderPositionalEncoding = encoderEmbedding + positionalEncoding(maxLen, dModel)
    encoderMiddleLayerStateInputs = []
    encoderMiddleLayerStateOutputs = []
    lastEncoderOutput = encoderPositionalEncoding
    lastEncoderStandaloneOutput = encoderPositionalEncoding
    encoderBypass = []
    encoderStandaloneBypass = []
    for i in range(layers):
        concattedInputLayer = tf.keras.layers.Concatenate(3)
        concattedInput = concattedInputLayer(
            [
                lastEncoderOutput,
                encoderAttentionMask[:, :, :, tf.newaxis],
            ]
        )
        concattedStandaloneInput = concattedInputLayer(
            [lastEncoderStandaloneOutput, encoderAttentionMask[:, :, :, tf.newaxis]]
        )
        encoderLayer = tf.keras.layers.TimeDistributed(
            EncoderLayer(dModel, dFF, pDropout, h, maxLen, depthEncoder)
        )
        encoder = encoderLayer(concattedInput)
        encoderStandalone = encoderLayer(concattedStandaloneInput)
        encoderMiddleLayer = MiddleLayer(h, dModel // h, maxLen)
        encoderMiddleRNN, _ = encoderMiddleLayer(encoder)
        encoderMiddleRNNInitialStateInput = tf.keras.layers.Input(
            shape=(maxLen * dModel,)
        )
        encoderMiddleLayerStateInputs.append(encoderMiddleRNNInitialStateInput)
        encoderStandaloneMiddleRNN, encoderStandaloneMiddleRNNState = (
            encoderMiddleLayer(encoderStandalone, encoderMiddleRNNInitialStateInput)
        )
        encoderMiddleLayerStateOutputs.append(encoderStandaloneMiddleRNNState)
        encoderBypass.append(lastEncoderOutput)
        encoderStandaloneBypass.append(lastEncoderStandaloneOutput)
        lastEncoderOutput = encoderMiddleRNN
        lastEncoderStandaloneOutput = encoderStandaloneMiddleRNN
        j = 1
        while (i + 1) % j == 0:
            layer = AddNorm()
            lastEncoderOutput = layer(encoderBypass[i - j + 1], lastEncoderOutput)
            lastEncoderStandaloneOutput = layer(
                encoderStandaloneBypass[i - j + 1], lastEncoderStandaloneOutput
            )
            j *= 2
    encoderReshape2 = tf.keras.layers.Reshape(
        target_shape=(encoderRecurrent, maxLen * dModel)
    )(lastEncoderOutput)
    encoderRNN, _ = tf.keras.layers.RNN(
        AttentionRNNCell(h, dModel // h, maxLen),
        return_state=True,
    )(encoderReshape2)
    encoderReshape3 = tf.keras.layers.Reshape(target_shape=(maxLen, dModel))(encoderRNN)
    tf.keras.Model(inputs=encoderInput, outputs=encoderReshape3).summary()


with open("./num2char.json") as f:
    num2char = json.loads("".join(f.readlines()))
with open("./char2num.json") as f:
    char2num = json.loads("".join(f.readlines()))
with open("./tokens.json") as f:
    tokens = json.loads("".join(f.readlines()))
depth = len(num2char)
maxLen = 8
toTrain = False
models = useExtendedTransformer(
    32,
    64,
    0.2,
    4,
    maxLen,
    depth,
    depth,
    depth,
    4,
)
