import json
import tensorflow as tf
import numpy as np
import math
import random

batchSize = 16
encoderRecurrentCount = None
decoderRecurrentCount = None


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


class RNNTiler(tf.keras.Model):
    def call(self, *inputs):
        return tf.tile(inputs[0], (1, inputs[1].shape[1], 1, 1))

    def compute_output_shape(self, inputShape):
        return inputShape[0][0:1] + (decoderRecurrentCount,) + inputShape[0][2:]


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
        mask = inputs[0][:, :, -1][:, tf.newaxis, :, tf.newaxis]
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


class DecoderLayer(tf.keras.Model):
    def __init__(self, dModel, dFF, pDropout, h, maxLen, depthDecoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dModel = dModel
        self.dFF = dFF
        self.pDropout = pDropout
        self.h = h
        self.maxLen = maxLen
        self.depthDecoder = depthDecoder
        self.attn0 = tf.keras.layers.MultiHeadAttention(h, dModel // h)
        self.norm1 = AddNorm()
        self.dropout1 = tf.keras.layers.Dropout(pDropout)
        self.attn1 = tf.keras.layers.MultiHeadAttention(h, dModel // h)
        self.norm2 = AddNorm()
        self.dropout2 = tf.keras.layers.Dropout(pDropout)
        self.ff = FF(dModel, dFF, maxLen, use_causal_mask=True)
        self.norm3 = AddNorm()
        self.dropout3 = tf.keras.layers.Dropout(pDropout)

    def build(self, input_shape):
        input_shape = input_shape[0:2] + ((input_shape[2] - 1) // 2,)
        self.attn0.build(input_shape, input_shape)
        self.norm1.build(input_shape)
        self.dropout1.build(input_shape)
        self.attn1.build(input_shape, input_shape)
        self.norm2.build(input_shape)
        self.dropout2.build(input_shape)
        self.ff.build(input_shape)
        self.norm3.build(input_shape)
        self.dropout3.build(input_shape)

    def call(self, *inputs):
        input = inputs[0][:, :, :-1]
        encoderOutput = input[:, :, input.shape[2] // 2 :]
        input = inputs[0][:, :, : input.shape[2] // 2]
        mask = inputs[0][:, :, -1][:, tf.newaxis, :, tf.newaxis]
        ret = input
        ret = self.attn0(ret, ret, attention_mask=mask, use_causal_mask=True)
        ret = self.norm1(ret, input)
        input = self.dropout1(ret)
        ret = self.attn1(input, encoderOutput)
        ret = self.norm2(ret, input)
        input = self.dropout2(ret)
        ret = self.ff(input)
        ret = self.norm3(ret, input)
        ret = self.dropout3(ret)
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape[0:2] + ((input_shape[2] - 1) // 2,)


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
    encoderInput = tf.keras.Input(shape=[encoderRecurrentCount, maxLen])
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
        target_shape=(encoderRecurrentCount, maxLen * dModel)
    )(lastEncoderOutput)
    encoderRNNLayer = tf.keras.layers.GRU(
        maxLen * dModel,
        return_state=True,
    )
    encoderRNN, _ = encoderRNNLayer(encoderReshape2)
    encoderReshape3 = tf.keras.layers.Reshape(target_shape=(maxLen, dModel))(encoderRNN)
    decoderInput = tf.keras.Input(shape=(decoderRecurrentCount, maxLen))
    decoderStandaloneRNNInput = tf.keras.Input(
        shape=(decoderRecurrentCount, maxLen, dModel)
    )
    decoderStandaloneInput = tf.keras.Input(
        shape=(decoderRecurrentCount, maxLen, dModel),
    )
    decoderStandaloneMaskInput = tf.keras.Input(
        shape=(decoderRecurrentCount, maxLen),
    )
    decoderOnes = tf.keras.layers.Dense(
        units=maxLen,
        kernel_initializer="zeros",
        bias_initializer="ones",
        trainable=False,
    )(decoderInput)
    decoderAttentionMask = tf.keras.layers.Minimum()((decoderInput, decoderOnes))
    decoderEmbeddingLayer = tf.keras.layers.TimeDistributed(
        layer=tf.keras.layers.Embedding(
            input_dim=depthDecoder,
            output_dim=dModel,
            mask_zero=True,
        ),
    )
    decoderEmbedding = decoderEmbeddingLayer(decoderInput)
    decoderPositionalEncoding = decoderEmbedding + positionalEncoding(maxLen, dModel)
    tiler = RNNTiler()(encoderReshape3[:, tf.newaxis], decoderInput)
    bridgeRNNLayer = MiddleLayer(h, dModel // h, maxLen)
    bridgeRNN, _ = bridgeRNNLayer(tiler)
    decoderMiddleLayerStateInputs = []
    decoderMiddleLayerStateOutputs = []
    lastDecoderOutput = decoderPositionalEncoding
    lastDecoderStandaloneOutput = decoderStandaloneInput
    decoderBypass = []
    decoderStandaloneBypass = []
    for i in range(layers):
        concattedInputLayer = tf.keras.layers.Concatenate(3)
        concattedInput = concattedInputLayer(
            [
                lastDecoderOutput,
                bridgeRNN,
                decoderAttentionMask[:, :, :, tf.newaxis],
            ]
        )
        concattedStandaloneInput = concattedInputLayer(
            [
                lastDecoderStandaloneOutput,
                decoderStandaloneRNNInput,
                decoderStandaloneMaskInput[:, :, :, tf.newaxis],
            ]
        )
        decoderLayer = tf.keras.layers.TimeDistributed(
            DecoderLayer(dModel, dFF, pDropout, h, maxLen, depthDecoder)
        )
        decoder = decoderLayer(concattedInput)
        decoderStandalone = decoderLayer(concattedStandaloneInput)
        decoderMiddleLayer = MiddleLayer(h, dModel // h, maxLen, use_causal_mask=True)
        decoderMiddleRNN, _ = decoderMiddleLayer(decoder)
        decoderMiddleRNNInitialStateInput = tf.keras.layers.Input(
            shape=(maxLen * dModel,)
        )
        decoderMiddleLayerStateInputs.append(decoderMiddleRNNInitialStateInput)
        decoderStandaloneMiddleRNN, decoderStandaloneMiddleRNNState = (
            decoderMiddleLayer(decoderStandalone, decoderMiddleRNNInitialStateInput)
        )
        decoderMiddleLayerStateOutputs.append(decoderStandaloneMiddleRNNState)
        decoderBypass.append(lastDecoderOutput)
        decoderStandaloneBypass.append(lastDecoderStandaloneOutput)
        lastDecoderOutput = decoderMiddleRNN
        lastDecoderStandaloneOutput = decoderStandaloneMiddleRNN
        j = 1
        while (i + 1) % j == 0:
            layer = AddNorm()
            lastDecoderOutput = layer(decoderBypass[i - j + 1], lastDecoderOutput)
            lastDecoderStandaloneOutput = layer(
                decoderStandaloneBypass[i - j + 1], lastDecoderStandaloneOutput
            )
            j *= 2
    decoderDenseLayer = tf.keras.layers.TimeDistributed(
        layer=tf.keras.layers.Dense(
            units=depthTarget,
            activation="softmax",
        ),
    )
    decoderDense = decoderDenseLayer(lastDecoderOutput)
    decoderStandaloneDense = decoderDenseLayer(lastDecoderStandaloneOutput)
    trainer = tf.keras.Model(
        inputs=(encoderInput, decoderInput),
        outputs=decoderDense,
    )
    optimizer = tf.keras.optimizers.Adadelta(1.0)
    trainer.compile(
        optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    encoder = tf.keras.Model(
        inputs=[encoderInput] + encoderMiddleLayerStateInputs,
        outputs=[lastEncoderOutput] + encoderMiddleLayerStateOutputs,
    )
    decoder = tf.keras.Model(
        inputs=[
            decoderStandaloneInput,
            decoderStandaloneMaskInput,
            decoderStandaloneRNNInput,
        ]
        + decoderMiddleLayerStateInputs,
        outputs=[decoderStandaloneDense] + decoderMiddleLayerStateOutputs,
    )
    return {
        "trainer": trainer,
        "encoder": encoder,
        "decoder": decoder,
        "encoderRNNLayer": encoderRNNLayer,
        "bridgeRNNLayer": bridgeRNNLayer,
        "decoderEmbeddingLayer": decoderEmbeddingLayer,
    }


with open("./num2char.json") as f:
    num2char = json.loads("".join(f.readlines()))
with open("./char2num.json") as f:
    char2num = json.loads("".join(f.readlines()))
with open("./tokens.json") as f:
    tokens = json.loads("".join(f.readlines()))
depth = len(num2char)
maxLen = 8
models = useExtendedTransformer(
    32,
    64,
    0.2,
    4,
    maxLen,
    depth,
    depth,
    depth,
    16,
)
models["trainer"].summary()
tf.keras.utils.plot_model(models["trainer"], "model.png", show_shapes=True)


def loader():
    pads = np.array([0] * maxLen)
    while True:
        encoderInput = []
        decoderInput = []
        decoderOutput = []
        for j in range(batchSize * 4):
            i = math.floor(random.random() * (len(tokens) - 8)) + 8
            ea = tokens[i - math.floor(random.random()) * 8 - 8 : i].copy()
            for k in range(len(ea)):
                ec = ea[k].copy()
                ea[k] = ec
                ec.append(2)
            flatten = sum(ea, [])
            flatten.extend(
                [0]
                * (math.floor(len(flatten) / maxLen) * maxLen + maxLen - len(flatten))
            )
            e = np.array(flatten).reshape([len(flatten) // maxLen, maxLen])

            da = tokens[i].copy()
            da.insert(0, 1)
            da.extend([0] * (math.floor(len(da) / maxLen) * maxLen + maxLen - len(da)))
            d = np.array(da).reshape([len(da) // maxLen, maxLen])

            de = tokens[i].copy()
            de.append(2)
            de.extend([0] * (math.floor(len(de) / maxLen) * maxLen + maxLen - len(de)))
            o = np.array(de).reshape([len(de) // maxLen, maxLen])
            if len(e) < 2:
                e = np.append(e, [pads], 0)

            if len(d) < 2:
                d = np.append(d, [pads], 0)

            if len(o) < 2:
                o = np.append(o, [pads], 0)
            encoderInput.append(e)
            decoderInput.append(d)
            decoderOutput.append(o)
        encoderInputMax = 0
        for e in encoderInput:
            encoderInputMax = max(len(e), encoderInputMax)
        for b, e in enumerate(encoderInput):
            for a in range(encoderInputMax - len(e)):
                encoderInput[b] = np.append(encoderInput[b], [pads], 0)
        decoderInputMax = 0
        for e in decoderInput:
            decoderInputMax = max(len(e), decoderInputMax)

        for b, e in enumerate(decoderInput):
            for a in range(decoderInputMax - len(e)):
                decoderInput[b] = np.append(decoderInput[b], [pads], 0)

        decoderOutputMax = 0
        for e in decoderOutput:
            decoderOutputMax = max(len(e), decoderOutputMax)

        for b, e in enumerate(decoderOutput):
            for a in range(decoderOutputMax - len(e)):
                decoderOutput[b] = np.append(decoderOutput[b], [pads], 0)
        yield (
            (np.array(encoderInput), np.array(decoderInput)),
            np.array(decoderOutput),
        )


class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, _):
        toSave = save(models["trainer"])
        with open(
            "./weights/weight-" + str(1) + ".jsonl",
            "w",
        ) as f:
            f.write(toSave)


def train():
    trainDatas = loader()
    epoch = 0
    while True:
        print("epoch " + str(epoch))
        data = next(trainDatas)
        models["trainer"].fit(
            data[0],
            data[1],
            batch_size=batchSize,
            steps_per_epoch=4,
            epochs=1,
            callbacks=[Callback()] if epoch % 50 == 1 else [],
        )
        epoch += 1


def predict():
    batchSize = 1
    prompt = "20240509"
    encoderInput = [4]
    constantPositionalEncoding = positionalEncoding(maxLen, 32)[
        tf.newaxis, tf.newaxis, :, :
    ]
    for c in prompt:
        encoderInput.append(char2num[c])
    encoderInput.append(2)
    encoderState = [tf.zeros((batchSize, maxLen * 32))] * 16
    encoderRNNState = tf.zeros((batchSize, maxLen * 32))
    while True:
        while len(encoderInput) > 8:
            tempEncoderInput = tf.reshape(
                (
                    encoderInput
                    + [0]
                    * (
                        (len(encoderInput) // maxLen) * maxLen
                        + maxLen
                        - len(encoderInput)
                    )
                )[0:8],
                (batchSize, 1, 8),
            )
            encoderInput = encoderInput[8:]
            encoderOutput, *encoderState = models["encoder"](
                [tempEncoderInput] + encoderState
            )
            for i, encoderStateI in enumerate(encoderState):
                encoderState[i] = tf.reshape(encoderStateI, (1, maxLen * 32))
            _, encoderRNNState = models["encoderRNNLayer"](
                tf.reshape(encoderOutput, (batchSize, -1, maxLen * 32)),
                initial_state=encoderRNNState,
            )
            encoderRNNState = tf.reshape(encoderRNNState, (batchSize, -1))
        tempEncoderInput = tf.reshape(
            (
                encoderInput
                + [0]
                * ((len(encoderInput) // maxLen) * maxLen + maxLen - len(encoderInput))
            )[0:8],
            (batchSize, 1, 8),
        )
        encoderInput = encoderInput[8:]
        encoderOutput, *_ = models["encoder"]([tempEncoderInput] + encoderState)
        encoderRNNOutput, _ = models["encoderRNNLayer"](
            tf.reshape(encoderOutput, (batchSize, -1, maxLen * 32)),
            initial_state=encoderRNNState,
        )
        encoderRNNOutput = tf.reshape(encoderRNNOutput, (batchSize, 1, maxLen, 32))
        decoderInput = [1]
        decoderOutputTokens = []
        bridgeRNNState = tf.zeros((batchSize, maxLen * 32))
        decoderState = [tf.zeros((batchSize, maxLen * 32))] * 16
        eos = False
        while True:
            if eos:
                break
            bridgeRNNOutput, bridgeRNNState = models["bridgeRNNLayer"](
                encoderRNNOutput, bridgeRNNState
            )
            bridgeRNNState = tf.reshape(bridgeRNNState, (1, maxLen * 32))
            for k in range(8):
                tempDecoderInput = tf.reshape(
                    (
                        decoderInput
                        + [0]
                        * (
                            (len(decoderInput) // maxLen) * maxLen
                            + maxLen
                            - len(decoderInput)
                        )
                    )[0:8],
                    (batchSize, 1, 8),
                )
                decoderMask = tf.minimum(tempDecoderInput, 1)
                tempDecoderInput = (
                    models["decoderEmbeddingLayer"](tempDecoderInput)
                    + constantPositionalEncoding
                )
                decoderOutput, *newDecoderState = models["decoder"](
                    [
                        tempDecoderInput,
                        decoderMask,
                        bridgeRNNOutput,
                    ]
                    + decoderState
                )
                resultSorted = tf.argsort(decoderOutput[0][0], 1)[k]
                results = []
                sum = 0
                for l in range(5):
                    c = decoderOutput[0][0][k][resultSorted[~l]].numpy()
                    sum += c
                    results.append(c)
                r = random.random() * sum
                t = 0
                m = 0
                while t < r:
                    t += results[m]
                    m += 1
                result = resultSorted[~m + 1]
                decoderInput.append(result)
                decoderOutputTokens.append(result)
                print(num2char[result], end="", flush=True)
                if result == 2:
                    eos = True
                    break
            decoderInput = decoderInput[8:]
            for i, decoderStateI in enumerate(newDecoderState):
                decoderState[i] = tf.reshape(decoderStateI, (1, maxLen * 32))
        encoderInput.extend(decoderOutputTokens)
        print()


with open("./weights/weight-1.jsonl") as f:
    weights = load("".join(f.readlines()))
models["trainer"].set_weights(weights)
# toSave = save(models["trainer"])
# with open("./weights/weight-" + str(2) + ".jsonl", "w") as f:
#     f.write(toSave)
toTrain = False
if toTrain:
    train()
else:
    predict()
