import json
import tensorflow as tf
import numpy as np
import math
import random
import pandas as pd
from functools import reduce

batchSize = 32


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
        return (
            input
            if tf.is_symbolic_tensor(input)
            else tf.tile(
                positionalEncoding(self.length, self.depth)[np.newaxis, :, :],
                (batchSize, 1, 1),
            )
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


class CustomizedMultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    def __init__(self, use_causal_mask=False, *args, **kwargs):
        if len(args) != 0:
            raise
        super().__init__(*args, **kwargs)
        self.use_causal_mask = use_causal_mask

    def build(self, input_shape):
        shape = (
            input_shape[0]
            if isinstance(input_shape[0], list)
            else input_shape[0:1] + input_shape[2:]
        )
        super().build(shape, value_shape=shape)
        self.built = True

    def call(self, inputs, **kwargs):
        if len(inputs.shape) == 4:
            buffer = tf.unstack(tf.transpose(inputs, [1, 0, 2, 3]))
            inputs = buffer[0]
            value = buffer[1]
            _a = tf.unstack(buffer[2], None, 2)[0] if len(buffer) == 3 else None
            attentionMask = (
                None
                if _a is not None
                else np.tile(
                    np.tile(_a[:, :, np.newaxis], [1, 1, buffer[2].shape[1]])[
                        :, np.newaxis, :, :
                    ],
                    (1, self.numHeads, 1, 1),
                )
            )
            return super().call(
                inputs,
                value=value,
                attention_mask=attentionMask,
                use_causal_mask=self.use_causal_mask,
            )
        return super().call(inputs, **kwargs, use_causal_mask=self.use_causal_mask)

    def compute_output_shape(self, inputShape):
        shape = inputShape[0:1] + inputShape[2:]
        if self.output_shape is not None:
            return shape[0:-1].concat(self.output_shape)
        return shape


class RNNMultiHeadAttentionCell(tf.keras.Model):
    def __init__(self, numHeads, keyDim, maxLen, use_causal_mask=False):
        super().__init__()
        self.maxLen = maxLen
        self.state_size = keyDim * numHeads * maxLen
        self.attention = CustomizedMultiHeadAttention(
            use_causal_mask=use_causal_mask, num_heads=numHeads, key_dim=keyDim
        )

    def call(self, *inputs):
        ret = tf.reshape(
            self.attention.call(
                tf.reshape(inputs[0], [batchSize, self.maxLen, -1]),
                value=tf.reshape(inputs[1], [batchSize, self.maxLen, -1]),
            ),
            [batchSize, -1],
        )
        return [ret, ret]

    def build(self, inputShape):
        super().build(inputShape)
        self.attention.build(
            [
                [inputShape[0], self.maxLen, inputShape[1] // self.maxLen],
                [inputShape[0], self.maxLen, inputShape[1] // self.maxLen],
            ]
        )

    def computeOutputShape(self, inputShape):
        return inputShape


class RNNMultiHeadAttention(tf.keras.layers.RNN):
    def __init__(self, length, depth, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = length
        self.depth = depth

    def get_initial_state(self, batch_size):
        return [
            tf.tile(
                tf.reshape(
                    positionalEncoding(self.length, self.depth),
                    [-1, 1, self.length * self.depth],
                ),
                (1, batch_size, 1),
            ),
        ]


class MultiHeadAttentionConcatter(tf.keras.Model):

    def call(self, *inputs):
        query = inputs[0]
        kv = inputs[1]
        mask = inputs[2] if len(inputs) == 3 else None
        dModel = query.shape[len(query.shape) - 1]
        query = (
            tf.reshape(query, (batchSize,) + query.shape[1:2] + (1,) + query.shape[2:])
            if query is not None
            else query
        )
        if reduce(
            lambda x, y: x * batchSize if y is None else x * y, kv.shape, 1
        ) != reduce(lambda x, y: x * batchSize if y is None else x * y, query.shape, 1):
            kv = tf.tile(kv[:, tf.newaxis, :, :], (1, query.shape[1], 1, 1))
        kv = tf.reshape(kv, query.shape) if kv is not None else query
        mask = (
            tf.reshape(
                tf.tile(mask[:, :, :, tf.newaxis], (1, 1, 1, dModel)), query.shape
            )
            if mask is not None
            else tf.ones(query.shape)
        )
        return tf.concat([query, kv, mask], 2)

    def compute_output_shape(self, input_shape):
        input = input_shape[0] if isinstance(input_shape[0], list) else input_shape
        return input[0][0:2] + (3,) + input[0][2:]


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
            mask_zero=True,
        ),
    )(encoderInput)
    encoderConstantPositionalEncoding = tf.keras.layers.TimeDistributed(
        layer=PositionalEncoding(maxLen, dModel),
    )(encoderEmbedding)
    encoderPositionalEncoding = tf.keras.layers.Add()(
        [encoderEmbedding, encoderConstantPositionalEncoding]
    )
    lastEncoderOutput = encoderPositionalEncoding
    for i in range(layers):
        encoderMultiHeadAttentionConcatter = MultiHeadAttentionConcatter()(
            lastEncoderOutput,
            lastEncoderOutput,
            encoderAttentionMask,
        )
        encoderMultiHeadAttention = tf.keras.layers.TimeDistributed(
            layer=CustomizedMultiHeadAttention(
                num_heads=h,
                key_dim=dModel // h,
            ),
        )(encoderMultiHeadAttentionConcatter)
        encoderDropout0 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.Dropout(rate=pDropout),
        )(encoderMultiHeadAttention)
        encoderAdd0 = tf.keras.layers.Add()([encoderDropout0, lastEncoderOutput])
        encoderNorm0 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.LayerNormalization(),
        )(encoderAdd0)
        encoderFF0 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.Dense(units=dFF, activation="relu"),
        )(encoderNorm0)
        encoderFF1 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.Dense(units=dModel, activation="linear"),
        )(encoderFF0)
        encoderDropout1 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.Dropout(rate=pDropout),
        )(encoderFF1)
        encoderAdd1 = tf.keras.layers.Add()([encoderNorm0, encoderDropout1])
        encoderNorm1 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.LayerNormalization(),
        )(encoderAdd1)
        lastEncoderOutput = encoderNorm1
    encoderReshape0 = tf.keras.layers.Reshape(target_shape=(None, maxLen * dModel))(
        lastEncoderOutput
    )
    encoderRNNLayer = RNNMultiHeadAttention(
        cell=RNNMultiHeadAttentionCell(h, dModel // h, maxLen),
        length=maxLen,
        depth=dModel,
    )
    encoderRNN = encoderRNNLayer(encoderReshape0)
    encoderReshape1 = tf.keras.layers.Reshape(target_shape=(maxLen, dModel))(encoderRNN)
    decoderInput = tf.keras.Input(shape=(None, maxLen))
    decoderStandaloneInput = tf.keras.Input(shape=(maxLen, dModel))
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
            mask_zero=True,
        ),
    )
    decoderEmbedding = decoderEmbeddingLayer(decoderInput)
    decoderConstantPositionalEncoding = tf.keras.layers.TimeDistributed(
        layer=PositionalEncoding(maxLen, dModel),
    )(decoderEmbedding)
    decoderPositionalEncoding = tf.keras.layers.Add()(
        [decoderEmbedding, decoderConstantPositionalEncoding]
    )
    decoderReshape0 = tf.keras.layers.Reshape(target_shape=(None, maxLen * dModel))(
        decoderPositionalEncoding
    )
    decoderRNNLayer = RNNMultiHeadAttention(
        cell=RNNMultiHeadAttentionCell(h, dModel // h, maxLen, use_causal_mask=True),
        length=maxLen,
        depth=dModel,
        return_sequences=True,
    )
    decoderRNN = decoderRNNLayer(decoderReshape0)
    decoderReshape1 = tf.keras.layers.Reshape(target_shape=(None, maxLen, dModel))(
        decoderRNN
    )
    lastDecoderOutput = decoderReshape1
    lastDecoderStandaloneOutput = decoderStandaloneRNNInput
    for i in range(layers):
        decoderMaskedMultiHeadAttentionConcatterLayer = MultiHeadAttentionConcatter()
        decoderMaskedMultiHeadAttentionConcatter = (
            decoderMaskedMultiHeadAttentionConcatterLayer(
                lastDecoderOutput,
                lastDecoderOutput,
            )
        )
        decoderStandaloneMaskedMultiHeadAttentionConcatter = (
            decoderMaskedMultiHeadAttentionConcatterLayer(
                lastDecoderStandaloneOutput,
                lastDecoderStandaloneOutput,
            )
        )
        decoderMaskedMultiHeadAttentionLayer = tf.keras.layers.TimeDistributed(
            layer=CustomizedMultiHeadAttention(
                num_heads=h,
                key_dim=dModel // h,
                use_causal_mask=True,
            ),
        )
        decoderMaskedMultiHeadAttention = decoderMaskedMultiHeadAttentionLayer(
            decoderMaskedMultiHeadAttentionConcatter,
        )
        decoderStandaloneMaskedMultiHeadAttention = (
            decoderMaskedMultiHeadAttentionLayer(
                decoderStandaloneMaskedMultiHeadAttentionConcatter,
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
        decoderMultiHeadAttentionConcatterLayer = MultiHeadAttentionConcatter()
        decoderMultiHeadAttentionConcatter = decoderMultiHeadAttentionConcatterLayer(
            decoderNorm0,
            encoderReshape1,
            decoderAttentionMask,
        )

        decoderStandaloneMultiHeadAttentionConcatter = (
            decoderMultiHeadAttentionConcatterLayer(
                decoderStandaloneNorm0,
                decoderStandaloneInput,
                decoderStandaloneMaskInput,
            )
        )
        decoderMultiHeadAttentionLayer = tf.keras.layers.TimeDistributed(
            layer=CustomizedMultiHeadAttention(
                num_heads=h,
                key_dim=dModel // h,
            ),
        )
        decoderMultiHeadAttention = decoderMultiHeadAttentionLayer(
            decoderMultiHeadAttentionConcatter
        )
        decoderStandaloneMultiHeadAttention = decoderMultiHeadAttentionLayer(
            decoderStandaloneMultiHeadAttentionConcatter
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
        decoderFFLayer0 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.Dense(units=dFF, activation="relu"),
        )
        decoderFF0 = decoderFFLayer0(decoderNorm1)
        decoderStandaloneFF0 = decoderFFLayer0(decoderStandaloneNorm1)
        decoderFFLayer1 = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.Dense(
                units=dModel,
                activation="linear",
            ),
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
        lastDecoderOutput = decoderNorm2
        lastDecoderStandaloneOutput = decoderStandaloneNorm2

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
    optimizer = tf.keras.optimizers.Adagrad()
    trainer.compile(
        optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True,
    )
    encoder = tf.keras.Model(
        inputs=encoderInput,
        outputs=lastEncoderOutput,
    )
    decoder = tf.keras.Model(
        inputs=(
            decoderStandaloneRNNInput,
            decoderStandaloneMaskInput,
            decoderStandaloneInput,
        ),
        outputs=decoderStandaloneDense,
    )
    return {
        "trainer": trainer,
        "encoder": encoder,
        "decoder": decoder,
        "encoderRNNLayer": encoderRNNLayer,
        "decoderRNNLayer": decoderRNNLayer,
        "decoderEmbeddingLayer": decoderEmbeddingLayer,
    }


with open("./num2char.json") as f:
    num2char = json.loads("".join(f.readlines()))
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
    4,
)
models["trainer"].summary()
# tf.keras.utils.plot_model(models["trainer"], "model.png", show_shapes=True)


def loader():
    zeros = np.array([0] * maxLen)
    while True:
        encoderInput = []
        decoderInput = []
        decoderOutput = []
        for j in range(batchSize):
            i = math.floor(random.random() * (len(tokens) - 8)) + 8
            ea = tokens[i - 8 : i - math.floor(random.random() * 8)]
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
                e = np.append(e, [zeros], 0)

            if len(d) < 2:
                d = np.append(d, [zeros], 0)

            if len(o) < 2:
                o = np.append(o, [zeros], 0)
            encoderInput.append(e)
            decoderInput.append(d)
            decoderOutput.append(o)
        encoderInputMax = 0
        for e in encoderInput:
            encoderInputMax = max(len(e), encoderInputMax)
        for b, e in enumerate(encoderInput):
            for a in range(encoderInputMax - len(e)):
                encoderInput[b] = np.append(encoderInput[b], [zeros], 0)
        decoderInputMax = 0
        for e in decoderInput:
            decoderInputMax = max(len(e), decoderInputMax)

        for b, e in enumerate(decoderInput):
            for a in range(decoderInputMax - len(e)):
                decoderInput[b] = np.append(decoderInput[b], [zeros], 0)

        decoderOutputMax = 0
        for e in decoderOutput:
            decoderOutputMax = max(len(e), decoderOutputMax)

        for b, e in enumerate(decoderOutput):
            for a in range(decoderOutputMax - len(e)):
                decoderOutput[b] = np.append(decoderOutput[b], [zeros], 0)
        yield (
            (np.array(encoderInput), np.array(decoderInput)),
            np.array(decoderOutput),
        )


epochOffset = 113


class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, _):
        toSave = save(models["trainer"])
        with open("./weights/weight-" + str(epoch + epochOffset) + ".json", "w") as f:
            f.write(toSave)


def train():
    history = models["trainer"].fit(
        tf.data.Dataset.from_generator(
            loader, output_types=(("float32", "float32"), "float32")
        ),
        epochs=128,
        steps_per_epoch=32,
        validation_data=tf.data.Dataset.from_generator(
            loader, output_types=(("float32", "float32"), "float32")
        ),
        validation_steps=4,
        callbacks=[Callback()],
    )
    pd.DataFrame(history.history)[["loss", "val_loss"]].plot()


with open("./weights/weight-113.json") as f:
    weights = load("".join(f.readlines()))
models["trainer"].set_weights(weights)
train()
