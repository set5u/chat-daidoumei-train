import json
import tensorflow as tf
import numpy as np
import math
import random
import pandas as pd
from functools import reduce

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
    encoderInput = tf.keras.Input(shape=[None, maxLen])
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
            # mask_zero=True,
        ),
    )(encoderInput)
    encoderConstantPositionalEncoding = tf.keras.layers.TimeDistributed(
        layer=PositionalEncoding(maxLen, dModel),
    )(encoderEmbedding)
    encoderPositionalEncoding = tf.keras.layers.Add()(
        [encoderEmbedding, encoderConstantPositionalEncoding]
    )
    encoderMiddleLayerStateInputs = []
    encoderMiddleLayerStateOutputs = []
    lastEncoderOutput = encoderPositionalEncoding
    lastEncoderStandaloneOutput = encoderPositionalEncoding
    encoderBypass = []
    encoderStandaloneBypass = []
    for i in range(layers):
        encoderMultiHeadAttentionLayer = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.MultiHeadAttention(
                num_heads=h,
                key_dim=dModel // h,
            ),
        )
        encoderMultiHeadAttention = encoderMultiHeadAttentionLayer(
            lastEncoderOutput,
            lastEncoderOutput,
            attention_mask=encoderAttentionMask,
        )
        encoderStandaloneMultiHeadAttention = encoderMultiHeadAttentionLayer(
            lastEncoderStandaloneOutput,
            lastEncoderStandaloneOutput,
            attention_mask=encoderAttentionMask,
        )
        encoderDropoutLayer0 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.Dropout(rate=pDropout),
        )
        encoderDropout0 = encoderDropoutLayer0(encoderMultiHeadAttention)
        encoderStandaloneDropout0 = encoderDropoutLayer0(
            encoderStandaloneMultiHeadAttention
        )
        encoderAddLayer0 = tf.keras.layers.Add()
        encoderAdd0 = encoderAddLayer0([encoderDropout0, lastEncoderOutput])
        encoderStandaloneAdd0 = encoderAddLayer0(
            [encoderStandaloneDropout0, lastEncoderStandaloneOutput]
        )
        encoderNormLayer0 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.LayerNormalization(),
        )
        encoderNorm0 = encoderNormLayer0(encoderAdd0)
        encoderStandaloneNorm0 = encoderNormLayer0(encoderStandaloneAdd0)
        encoderFFLayer0 = tf.keras.layers.EinsumDense(
            "abcd,de->abde", (None, dModel, dFF), activation="relu"
        )
        encoderFF0 = encoderFFLayer0(encoderNorm0)
        encoderStandaloneFF0 = encoderFFLayer0(encoderStandaloneNorm0)
        encoderFFLayer1 = tf.keras.layers.EinsumDense(
            "abde,dc->abcd", (None, maxLen, dModel), activation="linear"
        )
        encoderFF1 = encoderFFLayer1(encoderFF0)
        encoderStandaloneFF1 = encoderFFLayer1(encoderStandaloneFF0)
        encoderDropoutLayer1 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.Dropout(rate=pDropout),
        )
        encoderDropout1 = encoderDropoutLayer1(encoderFF1)
        encoderStandaloneDropout1 = encoderDropoutLayer1(encoderStandaloneFF1)
        encoderAddLayer1 = tf.keras.layers.Add()
        encoderAdd1 = encoderAddLayer1([encoderNorm0, encoderDropout1])
        encoderStandaloneAdd1 = encoderAddLayer1(
            [encoderStandaloneNorm0, encoderStandaloneDropout1]
        )
        encoderNormLayer1 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.LayerNormalization(),
        )
        encoderNorm1 = encoderNormLayer1(encoderAdd1)
        encoderStandaloneNorm1 = encoderNormLayer1(encoderStandaloneAdd1)
        encoderMiddleReshapeLayer0 = tf.keras.layers.Reshape(
            target_shape=(None, maxLen * dModel)
        )
        encoderMiddleReshape0 = encoderMiddleReshapeLayer0(encoderNorm1)
        encoderStandaloneMiddleReshape0 = encoderMiddleReshapeLayer0(
            encoderStandaloneNorm1
        )
        encoderMiddleRNNLayer = tf.keras.layers.GRU(
            maxLen * dModel, return_sequences=True, return_state=True
        )
        encoderMiddleRNNInitialStateInput = tf.keras.layers.Input(
            shape=(maxLen * dModel,)
        )
        encoderMiddleLayerStateInputs.append(encoderMiddleRNNInitialStateInput)
        encoderMiddleRNN, _ = encoderMiddleRNNLayer(encoderMiddleReshape0)
        encoderStandaloneMiddleRNN, encoderStandaloneMiddleRNNState = (
            encoderMiddleRNNLayer(
                encoderStandaloneMiddleReshape0,
                initial_state=encoderMiddleRNNInitialStateInput,
            )
        )
        encoderMiddleLayerStateOutputs.append(encoderStandaloneMiddleRNNState)
        encoderMiddleReshapeLayer1 = tf.keras.layers.Reshape(
            target_shape=(None, maxLen, dModel)
        )
        encoderMiddleReshape1 = encoderMiddleReshapeLayer1(encoderMiddleRNN)
        encoderStandaloneMiddleReshape1 = encoderMiddleReshapeLayer1(
            encoderStandaloneMiddleRNN
        )
        encoderAddLayer2 = tf.keras.layers.Add()
        encoderAdd2 = encoderAddLayer2([encoderMiddleReshape1, encoderNorm1])
        encoderStandaloneAdd2 = encoderAddLayer2(
            [encoderStandaloneMiddleReshape1, encoderStandaloneNorm1]
        )
        encoderNormLayer2 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.LayerNormalization(),
        )
        encoderNorm2 = encoderNormLayer2(encoderAdd2)
        encoderStandaloneNorm2 = encoderNormLayer2(encoderStandaloneAdd2)
        encoderBypass.append(lastEncoderOutput)
        encoderStandaloneBypass.append(lastEncoderStandaloneOutput)
        lastEncoderOutput = encoderNorm2
        lastEncoderStandaloneOutput = encoderStandaloneNorm2
        j = 1
        while (i + 1) % j == 0:
            layer = tf.keras.layers.Add()
            lastEncoderOutput = layer([encoderBypass[i - j + 1], lastEncoderOutput])
            lastEncoderStandaloneOutput = layer(
                [encoderStandaloneBypass[i - j + 1], lastEncoderStandaloneOutput]
            )
            j *= 2
    encoderReshape0 = tf.keras.layers.Reshape(target_shape=(None, maxLen * dModel))(
        lastEncoderOutput
    )
    encoderRNNLayer = tf.keras.layers.GRU(maxLen * dModel, return_state=True)
    encoderRNN, _ = encoderRNNLayer(encoderReshape0)
    encoderReshape1 = tf.keras.layers.Reshape(target_shape=(maxLen, dModel))(encoderRNN)
    decoderInput = tf.keras.Input(shape=(None, maxLen))
    decoderStandaloneInput = tf.keras.Input(shape=(None, maxLen, dModel))
    decoderStandaloneRNNInput = tf.keras.Input(
        shape=(None, maxLen, dModel),
    )
    decoderStandaloneMaskInput = tf.keras.Input(
        shape=(None, maxLen),
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
            # mask_zero=True,
        ),
    )
    decoderEmbedding = decoderEmbeddingLayer(decoderInput)
    decoderConstantPositionalEncoding = tf.keras.layers.TimeDistributed(
        layer=PositionalEncoding(maxLen, dModel),
    )(decoderEmbedding)
    decoderPositionalEncoding = tf.keras.layers.Add()(
        [decoderEmbedding, decoderConstantPositionalEncoding]
    )
    # decoderReshape0 = tf.keras.layers.Reshape(target_shape=(None, maxLen * dModel))(
    #     decoderPositionalEncoding
    # )
    # decoderRNNLayer = tf.keras.layers.GRU(
    #     maxLen * dModel, return_sequences=True, return_state=True
    # )
    # decoderRNN, _ = decoderRNNLayer(decoderReshape0)
    # decoderReshape1 = tf.keras.layers.Reshape(target_shape=(None, maxLen, dModel))(
    #     decoderRNN
    # )
    decoderReshape2Layer = tf.keras.layers.Reshape(target_shape=(1, maxLen * dModel))
    decoderReshape2 = decoderReshape2Layer(encoderReshape1)
    tilerLayer = RNNTiler()
    tiler = tilerLayer(decoderReshape2, decoderInput)
    bridgeRNNLayer = tf.keras.layers.GRU(
        maxLen * dModel, return_sequences=True, return_state=True
    )
    bridgeRNN, _ = bridgeRNNLayer(tiler)
    decoderReshape3Layer = tf.keras.layers.Reshape(target_shape=(None, maxLen, dModel))
    decoderReshape3 = decoderReshape3Layer(bridgeRNN)
    decoderMiddleLayerStateInputs = []
    decoderMiddleLayerStateOutputs = []
    lastDecoderOutput = decoderPositionalEncoding
    lastDecoderStandaloneOutput = decoderStandaloneRNNInput
    decoderBypass = []
    decoderStandaloneBypass = []
    for i in range(layers):
        decoderMaskedMultiHeadAttentionLayer = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.MultiHeadAttention(
                num_heads=h, key_dim=dModel // h, use_causal_mask=i == 0
            ),
        )
        decoderMaskedMultiHeadAttention = decoderMaskedMultiHeadAttentionLayer(
            lastDecoderOutput,
            lastDecoderOutput,
            attention_mask=decoderAttentionMask,
            use_causal_mask=True,
        )
        decoderStandaloneMaskedMultiHeadAttention = (
            decoderMaskedMultiHeadAttentionLayer(
                lastDecoderStandaloneOutput,
                lastDecoderStandaloneOutput,
                attention_mask=decoderStandaloneMaskInput,
                use_causal_mask=True,
            )
        )
        decoderDropoutLayer0 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.Dropout(rate=pDropout),
        )
        decoderDropout0 = decoderDropoutLayer0(decoderMaskedMultiHeadAttention)
        decoderStandaloneDropout0 = decoderDropoutLayer0(
            decoderStandaloneMaskedMultiHeadAttention
        )
        decoderAddLayer0 = tf.keras.layers.Add()
        decoderAdd0 = decoderAddLayer0(
            [
                decoderDropout0,
                lastDecoderOutput,
            ]
        )
        decoderStandaloneAdd0 = decoderAddLayer0(
            [
                decoderStandaloneDropout0,
                lastDecoderStandaloneOutput,
            ]
        )
        decoderNormLayer0 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.LayerNormalization(),
        )
        decoderNorm0 = decoderNormLayer0(decoderAdd0)
        decoderStandaloneNorm0 = decoderNormLayer0(decoderStandaloneAdd0)
        decoderMultiHeadAttentionLayer = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.MultiHeadAttention(
                num_heads=h,
                key_dim=dModel // h,
            ),
        )
        decoderMultiHeadAttention = decoderMultiHeadAttentionLayer(
            decoderNorm0,
            decoderReshape3,
            attention_mask=decoderAttentionMask,
        )
        decoderStandaloneMultiHeadAttention = decoderMultiHeadAttentionLayer(
            decoderStandaloneNorm0,
            decoderStandaloneInput,
            attention_mask=decoderStandaloneMaskInput,
        )
        decoderDropoutLayer1 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.Dropout(rate=pDropout),
        )
        decoderDropout1 = decoderDropoutLayer1(decoderMultiHeadAttention)
        decoderStandaloneDropout1 = decoderDropoutLayer1(
            decoderStandaloneMultiHeadAttention
        )
        decoderAddLayer1 = tf.keras.layers.Add()
        decoderAdd1 = decoderAddLayer1((decoderDropout1, decoderAdd0))
        decoderStandaloneAdd1 = decoderAddLayer1(
            [
                decoderStandaloneDropout1,
                decoderStandaloneAdd0,
            ]
        )
        decoderNormLayer1 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.LayerNormalization(),
        )
        decoderNorm1 = decoderNormLayer1(decoderAdd1)
        decoderStandaloneNorm1 = decoderNormLayer1(decoderStandaloneAdd1)
        decoderFFLayer0 = tf.keras.layers.EinsumDense(
            "abcd,de->abde", (None, dModel, dFF), activation="relu"
        )
        decoderFF0 = decoderFFLayer0(decoderNorm1)
        decoderStandaloneFF0 = decoderFFLayer0(decoderStandaloneNorm1)
        decoderFFLayer1 = tf.keras.layers.EinsumDense(
            "abde,dc->abcd",
            (None, maxLen, dModel),
            activation="linear",
        )
        decoderFF1 = decoderFFLayer1(decoderFF0)
        decoderStandaloneFF1 = decoderFFLayer1(decoderStandaloneFF0)
        decoderDropoutLayer2 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.Dropout(rate=pDropout),
        )
        decoderDropout2 = decoderDropoutLayer2(decoderFF1)
        decoderStandaloneDropout2 = decoderDropoutLayer2(decoderStandaloneFF1)
        decoderAddLayer2 = tf.keras.layers.Add()
        decoderAdd2 = decoderAddLayer2((decoderDropout2, decoderAdd1))
        decoderStandaloneAdd2 = decoderAddLayer2(
            (
                decoderStandaloneDropout2,
                decoderStandaloneAdd1,
            )
        )
        decoderNormLayer2 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.LayerNormalization(),
        )
        decoderNorm2 = decoderNormLayer2(decoderAdd2)
        decoderStandaloneNorm2 = decoderNormLayer2(decoderStandaloneAdd2)
        decoderMiddleReshapeLayer0 = tf.keras.layers.Reshape(
            target_shape=(None, maxLen * dModel)
        )
        decoderMiddleReshape0 = decoderMiddleReshapeLayer0(decoderNorm2)
        decoderStandaloneMiddleReshape0 = decoderMiddleReshapeLayer0(
            decoderStandaloneNorm2
        )
        decoderMiddleRNNLayer = tf.keras.layers.GRU(
            maxLen * dModel, return_sequences=True, return_state=True
        )
        decoderMiddleRNNInitialStateInput = tf.keras.layers.Input(
            shape=(maxLen * dModel,)
        )
        decoderMiddleLayerStateInputs.append(decoderMiddleRNNInitialStateInput)
        decoderMiddleRNN, _ = decoderMiddleRNNLayer(decoderMiddleReshape0)
        decoderStandaloneMiddleRNN, decoderStandaloneMiddleRNNState = (
            decoderMiddleRNNLayer(
                decoderStandaloneMiddleReshape0,
                initial_state=decoderMiddleRNNInitialStateInput,
            )
        )
        decoderMiddleLayerStateOutputs.append(decoderStandaloneMiddleRNNState)
        decoderMiddleReshapeLayer1 = tf.keras.layers.Reshape(
            target_shape=(None, maxLen, dModel)
        )
        decoderMiddleReshape1 = decoderMiddleReshapeLayer1(decoderMiddleRNN)
        decoderStandaloneMiddleReshape1 = decoderMiddleReshapeLayer1(
            decoderStandaloneMiddleRNN
        )
        decoderAddLayer3 = tf.keras.layers.Add()
        decoderAdd3 = decoderAddLayer3([decoderMiddleReshape1, decoderNorm2])
        decoderStandaloneAdd3 = decoderAddLayer3(
            [decoderStandaloneMiddleReshape1, decoderStandaloneNorm2]
        )
        decoderNormLayer3 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.LayerNormalization(),
        )
        decoderNorm3 = decoderNormLayer3(decoderAdd3)
        decoderStandaloneNorm3 = decoderNormLayer3(decoderStandaloneAdd3)
        lastDecoderOutput = decoderNorm3
        lastDecoderStandaloneOutput = decoderStandaloneNorm3
        decoderBypass.append(lastDecoderOutput)
        decoderStandaloneBypass.append(lastDecoderStandaloneOutput)
        lastDecoderOutput = decoderNorm2
        lastDecoderStandaloneOutput = decoderStandaloneNorm2
        j = 1
        while (i + 1) % j == 0:
            layer = tf.keras.layers.Add()
            lastDecoderOutput = layer([decoderBypass[i - j + 1], lastDecoderOutput])
            lastDecoderStandaloneOutput = layer(
                [decoderStandaloneBypass[i - j + 1], lastDecoderStandaloneOutput]
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
    optimizer = tf.keras.optimizers.Adadelta()
    trainer.compile(
        optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True,
    )
    encoder = tf.keras.Model(
        inputs=[encoderInput] + encoderMiddleLayerStateInputs,
        outputs=[lastEncoderOutput] + encoderMiddleLayerStateOutputs,
    )
    decoder = tf.keras.Model(
        inputs=[
            decoderStandaloneRNNInput,
            decoderStandaloneMaskInput,
            decoderStandaloneInput,
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
models["trainer"].summary()
tf.keras.utils.plot_model(models["trainer"], "model.png", show_shapes=True)


def loader():
    pads = np.array([0] * maxLen)
    while True:
        encoderInput = []
        decoderInput = []
        decoderOutput = []
        for j in range(batchSize):
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


epochOffset = 1


class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, _):
        predict()
        toSave = save(models["trainer"])
        with open("./weights/weight-" + str(epoch + epochOffset) + ".jsonl", "w") as f:
            f.write(toSave)


def train():
    history = models["trainer"].fit(
        tf.data.Dataset.from_generator(
            loader, output_types=(("float32", "float32"), "float32")
        ),
        epochs=150,
        steps_per_epoch=32,
        validation_data=tf.data.Dataset.from_generator(
            loader, output_types=(("float32", "float32"), "float32")
        ),
        validation_steps=4,
        callbacks=[Callback()],
    )
    pd.DataFrame(history.history)[["loss", "val_loss"]].plot()


def predict():
    batchSize = 1
    prompt = "数学やった?"
    encoderInput = [3, 51, 51, 454, 703, 5]
    for c in prompt:
        encoderInput.append(char2num[c])

    encoderInput.extend(
        [0]
        * (math.floor(len(encoderInput) / maxLen) * maxLen + maxLen - len(encoderInput))
    )
    encoderInput = np.array(encoderInput).reshape(
        [1, len(encoderInput) // maxLen, maxLen]
    )
    if len(encoderInput[0]) < 2:
        encoderInput = np.append(encoderInput, [[[0] * maxLen]], 0)
    encoderInput = tf.tile(encoderInput, [batchSize, 1, 1])
    encoderOutput = models["encoder"](
        (
            encoderInput,
            tf.zeros(
                (
                    batchSize,
                    maxLen * 32,
                )
            ),
            tf.zeros(
                (
                    batchSize,
                    maxLen * 32,
                )
            ),
            tf.zeros(
                (
                    batchSize,
                    maxLen * 32,
                )
            ),
            tf.zeros(
                (
                    batchSize,
                    maxLen * 32,
                )
            ),
        )
    )
    encoderRNNOutput = tf.reshape(
        models["encoderRNNLayer"](
            tf.reshape(encoderOutput[0], [batchSize, -1, maxLen * 32])
        )[0],
        [batchSize, maxLen, -1],
    )
    outputs = [1]
    for i in range(32):
        decoderInput = outputs.copy()
        decoderInput.extend(
            [0]
            * (
                math.floor(len(decoderInput) / maxLen) * maxLen
                + maxLen
                - len(decoderInput)
            )
        )

        decoderInput = np.array(decoderInput).reshape(
            [len(decoderInput) // maxLen, maxLen]
        )
        if len(decoderInput) < 2:
            decoderInput = np.append(decoderInput, [[0] * maxLen], 0)
        decoderInput = decoderInput[tf.newaxis, :, :]
        decoderInput = np.tile(decoderInput, [batchSize, 1, 1])
        decoderPositionalEncodingOutput = models["decoderEmbeddingLayer"](
            decoderInput
        ) + np.tile(
            positionalEncoding(maxLen, 32)[tf.newaxis, tf.newaxis, :, :],
            (batchSize, len(decoderInput[0]), 1, 1),
        )
        bridgeRNNOutput = tf.reshape(
            models["bridgeRNNLayer"](
                tf.tile(
                    tf.reshape(encoderRNNOutput, (batchSize, 1, maxLen * 32)),
                    (1, decoderInput.shape[1], 1),
                )
            )[0],
            (batchSize, -1, maxLen, 32),
        )
        decoderLayerOutput = models["decoder"](
            (
                decoderPositionalEncodingOutput,
                tf.minimum(decoderInput, tf.ones_like(decoderInput)),
                bridgeRNNOutput,
                tf.zeros(
                    (
                        batchSize,
                        maxLen * 32,
                    )
                ),
                tf.zeros(
                    (
                        batchSize,
                        maxLen * 32,
                    )
                ),
                tf.zeros(
                    (
                        batchSize,
                        maxLen * 32,
                    )
                ),
                tf.zeros(
                    (
                        batchSize,
                        maxLen * 32,
                    )
                ),
            )
        )
        decoderOutput = tf.argmax(decoderLayerOutput[0], 3)
        decoderArgmax = tf.reshape(decoderOutput[0], (-1,))
        print(num2char[decoderArgmax[i]], end="")
        outputs.append(decoderArgmax[i].numpy())


with open("./weights/weight-2.jsonl") as f:
    weights = load("".join(f.readlines()))
models["trainer"].set_weights(weights)
# toSave = save(models["trainer"])
# with open("./weights/weight-" + str(2) + ".jsonl", "w") as f:
#     f.write(toSave)
if toTrain:
    train()
else:
    predict()
