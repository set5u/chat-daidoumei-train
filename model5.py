import json
import tensorflow as tf
import numpy as np
import math
import random


toTrain = False
if toTrain:
    batchSize = 1
    encoderRecurrentCount = 8
    decoderRecurrentCount = 2
else:
    batchSize = 1
    encoderRecurrentCount = 1
    decoderRecurrentCount = 1


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
        self.attn.build(input_shape)
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
        self.attn0.build(input_shape)
        self.norm1.build(input_shape)
        self.dropout1.build(input_shape)
        self.attn1.build(input_shape)
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


class MiddleLayerCell(tf.keras.Model):
    def __init__(self, h, keyDim, maxLen, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.h = h
        self.keyDim = keyDim
        self.maxLen = maxLen
        self.state_size = h * keyDim * maxLen
        self.addNorm = AddNorm()

    def build(self, _):
        self.addNorm.build((None, self.maxLen, self.keyDim * self.h))

    def call(self, *inputs):
        ret = tf.reshape(
            inputs[0],
            (inputs[0].shape[0], self.maxLen, self.keyDim * self.h),
        )
        state = tf.reshape(
            inputs[1][0],
            (inputs[0].shape[0], self.maxLen, self.keyDim * self.h),
        )
        ret = self.addNorm(ret, state)
        ret = tf.reshape(ret, (inputs[0].shape[0], -1))
        return ret, ret


class MiddleLayer(tf.keras.layers.RNN):
    def __init__(self, h, keyDim, maxLen, *args, **kwargs):
        super().__init__(
            MiddleLayerCell(h, keyDim, maxLen), return_state=True, *args, **kwargs
        )
        self.maxLen = maxLen
        self.keyDim = keyDim
        self.h = h

    def call(self, inputs, initial_state=None):
        input = inputs[0] if isinstance(inputs, list) else inputs
        state = (
            initial_state
            if initial_state is not None
            else self.get_initial_state(input)
        )
        ret = tf.reshape(input, (batchSize, -1, self.maxLen * self.keyDim * self.h))
        ret, state = super().call(ret, initial_state=state)
        ret = tf.reshape(ret, (batchSize, -1, self.maxLen, self.keyDim * self.h))
        return (ret, state) if self.return_state else ret


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
    # encoderBypass = []
    # encoderStandaloneBypass = []
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
        # encoderBypass.append(lastEncoderOutput)
        # encoderStandaloneBypass.append(lastEncoderStandaloneOutput)
        lastEncoderOutput = encoderMiddleRNN
        lastEncoderStandaloneOutput = encoderStandaloneMiddleRNN
        # j = 1
        # while (i + 1) % j == 0:
        #     layer = AddNorm()
        #     lastEncoderOutput = layer(encoderBypass[i - j + 1], lastEncoderOutput)
        #     lastEncoderStandaloneOutput = layer(
        #         encoderStandaloneBypass[i - j + 1], lastEncoderStandaloneOutput
        #     )
        #     j *= 2
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
    # decoderBypass = []
    # decoderStandaloneBypass = []
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
        decoderMiddleLayer = MiddleLayer(h, dModel // h, maxLen)
        decoderMiddleRNN, _ = decoderMiddleLayer(decoder)
        decoderMiddleRNNInitialStateInput = tf.keras.layers.Input(
            shape=(maxLen * dModel,)
        )
        decoderMiddleLayerStateInputs.append(decoderMiddleRNNInitialStateInput)
        decoderStandaloneMiddleRNN, decoderStandaloneMiddleRNNState = (
            decoderMiddleLayer(decoderStandalone, decoderMiddleRNNInitialStateInput)
        )
        decoderMiddleLayerStateOutputs.append(decoderStandaloneMiddleRNNState)
        # decoderBypass.append(lastDecoderOutput)
        # decoderStandaloneBypass.append(lastDecoderStandaloneOutput)
        lastDecoderOutput = decoderMiddleRNN
        lastDecoderStandaloneOutput = decoderStandaloneMiddleRNN
        # j = 1
        # while (i + 1) % j == 0:
        #     layer = AddNorm()
        #     lastDecoderOutput = layer(decoderBypass[i - j + 1], lastDecoderOutput)
        #     lastDecoderStandaloneOutput = layer(
        #         decoderStandaloneBypass[i - j + 1], lastDecoderStandaloneOutput
        #     )
        #     j *= 2
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
    # optimizer = tf.keras.optimizers.Adadelta(1.0)
    # trainer.compile(
    #     optimizer,
    #     loss="sparse_categorical_crossentropy",
    #     metrics=["accuracy"],
    # )
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


with open("./num2word.json", "r", -1, "utf-8") as f:
    num2word = json.loads("".join(f.readlines()))
with open("./word2num.json", "r", -1, "utf-8") as f:
    word2num = json.loads("".join(f.readlines()))
with open("./wordTokens.json", "r", -1, "utf-8") as f:
    tokens = json.loads("".join(f.readlines()))
depth = len(num2word)
maxLen = 8
# tf.keras.utils.plot_model(models["trainer"], "model.png", show_shapes=True)

stepsPerEpoch = 256
dModel = 256
dFF = 128
layers = 16

def loader():
    while True:
        input = []
        output = []
        input2 = []
        for _ in range(batchSize * stepsPerEpoch):
            startIndex = math.floor(random.random() * (len(tokens) - (64 + 16)))
            input.append(tokens[startIndex : startIndex + 64])
            endIndex = startIndex + 64
            out = []
            while len(out) != 16:
                if tokens[endIndex] != 3:
                    out.append(tokens[endIndex])
                endIndex += 1
            output.append(out)
            input2.append([0] + out[:-1])

        yield (
            np.array(input).reshape((batchSize * stepsPerEpoch, -1, 8)),
            np.array(input2).reshape((batchSize * stepsPerEpoch, -1, 8)),
        ), np.array(output).reshape((batchSize * stepsPerEpoch, -1, 8))


def predict():
    encoderInput = []
    constantPositionalEncoding = positionalEncoding(maxLen, dModel)[
        tf.newaxis, tf.newaxis, :, :
    ]
    encoderState = [tf.zeros((batchSize, maxLen * dModel))] * layers
    encoderRNNState = tf.zeros((batchSize, maxLen * dModel))
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
                encoderState[i] = tf.reshape(encoderStateI, (1, maxLen * dModel))
            _, encoderRNNState = models["encoderRNNLayer"](
                tf.reshape(encoderOutput, (batchSize, -1, maxLen * dModel)),
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
            tf.reshape(encoderOutput, (batchSize, -1, maxLen * dModel)),
            initial_state=encoderRNNState,
        )
        encoderRNNOutput = tf.reshape(encoderRNNOutput, (batchSize, 1, maxLen, dModel))
        decoderInput = [1]
        decoderOutputTokens = []
        bridgeRNNState = tf.zeros((batchSize, maxLen * dModel))
        decoderState = [tf.zeros((batchSize, maxLen * dModel))] * layers
        bos = False
        while True:
            if bos:
                break
            bridgeRNNOutput, bridgeRNNState = models["bridgeRNNLayer"](
                encoderRNNOutput, bridgeRNNState
            )
            bridgeRNNState = tf.reshape(bridgeRNNState, (1, maxLen * dModel))
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
                print(num2word[result], end="", flush=True)
                if result == 1:
                    bos = True
                    break
            decoderInput = decoderInput[8:]
            for i, decoderStateI in enumerate(newDecoderState):
                decoderState[i] = tf.reshape(decoderStateI, (1, maxLen * dModel))
        encoderInput.extend(decoderOutputTokens)
        print()


class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, _):
        models["trainer"].save_weights("./weights/weights")


models = useExtendedTransformer(
    dModel,
    dFF,
    0.1,
    4,
    maxLen,
    depth,
    depth,
    depth,
    layers,
)
models["trainer"].summary()


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
            steps_per_epoch=stepsPerEpoch,
            epochs=1,
            callbacks=[Callback()] if epoch % 10 == 9 else [],
        )
        epoch += 1


# models["trainer"].load_weights("./weights/weights")

if toTrain:
    train()
else:
    predict()
