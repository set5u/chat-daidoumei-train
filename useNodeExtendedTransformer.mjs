import { NodeMultiHeadAttention } from "./NodeCustomizedMultiHeadAttention.mjs";
import * as tf from "@tensorflow/tfjs-node-gpu";
import { NodePositionalEncoding } from "./useNodePositionalEncoding.mjs";
class RNNMultiHeadAttentionCell extends tf.layers.RNNCell {
  constructor(numHeads, keyDim, maxLen) {
    super();
    this.maxLen = maxLen;
    this.stateSize = keyDim * numHeads * maxLen;
    this.attention = new NodeMultiHeadAttention({ numHeads, keyDim });
  }
  call(inputs, kwargs) {
    super.call(inputs, kwargs);
    var batchSize = inputs[0].shape[0];
    var ret = tf.tidy(() =>
      this.attention
        .call(
          inputs[0].reshape([batchSize, this.maxLen, -1]),
          Object.assign(Object.assign({}, kwargs), {
            value: inputs[1].reshape([batchSize, this.maxLen, -1]),
          })
        )
        .reshape([batchSize, -1])
    );
    return [ret, ret];
  }
  build(inputShape) {
    super.build(inputShape);
    this.attention.build([
      [inputShape[0], this.maxLen, inputShape[1] / this.maxLen],
      [inputShape[0], this.maxLen, inputShape[1] / this.maxLen],
    ]);
  }
  get trainableWeights() {
    return this.attention.trainableWeights;
  }
  getWeights() {
    return this.attention.getWeights();
  }
  setWeights(weights) {
    this.attention.setWeights(weights);
  }
}
RNNMultiHeadAttentionCell.className = "RNNMultiheadAttentionCell";
class RNNMultiHeadAttention extends tf.layers.RNN {
  getInitialState(inputs) {
    return [tf.ones([inputs.shape[0], inputs.shape[2]])];
  }
}
class MultiHeadAttentionConcatter extends tf.layers.Layer {
  call(inputs) {
    return tf.tidy(() => {
      let query = inputs[0];
      let kv = inputs[1];
      let mask = inputs[2];
      var dModel = query.shape[query.shape.length - 1];
      query =
        query &&
        query.reshape([...query.shape.slice(0, 2), 1, ...query.shape.slice(2)]);
      if (kv.size !== query.size) {
        kv = kv.expandDims(1).tile([1, query.shape[1], 1, 1]);
      }
      kv = kv ? kv.reshape(query.shape) : query;
      mask = mask
        ? mask.expandDims(3).tile([1, 1, 1, dModel]).reshape(query.shape)
        : tf.ones(query.shape);
      return tf.concat([query, kv, mask], 2);
    });
  }
  computeOutputShape(inputShape) {
    var input = Array.isArray(inputShape[0]) ? inputShape[0] : inputShape;
    return [...input.slice(0, 2), 3, ...input.slice(2)];
  }
}
MultiHeadAttentionConcatter.className = "MultiHeadAttentionConcatter";
export default ({
  dModel,
  dFF,
  pDropout,
  h,
  maxLen,
  depthEncoder,
  depthDecoder,
  depthTarget,
  layers,
}) => {
  var encoderInput = tf.input({ shape: [null, maxLen] });
  var encoderOnes = tf.layers
    .dense({
      units: maxLen,
      kernelInitializer: "zeros",
      biasInitializer: "ones",
      trainable: false,
    })
    .apply(encoderInput);
  var encoderAttentionMask = tf.layers
    .minimum()
    .apply([encoderInput, encoderOnes]);
  var encoderEmbedding = tf.layers
    .timeDistributed({
      layer: tf.layers.embedding({
        inputDim: depthEncoder,
        outputDim: dModel,
        maskZero: true,
      }),
    })
    .apply(encoderInput);
  var encoderConstantPositionalEncoding = tf.layers
    .timeDistributed({
      layer: new NodePositionalEncoding(maxLen, dModel),
    })
    .apply(encoderEmbedding);
  var encoderPositionalEncoding = tf.layers
    .add()
    .apply([encoderEmbedding, encoderConstantPositionalEncoding]);
  let lastEncoderOutput = encoderPositionalEncoding;
  for (let i = 0; i < layers; i++) {
    var encoderMultiHeadAttentionConcatter =
      new MultiHeadAttentionConcatter().apply([
        lastEncoderOutput,
        lastEncoderOutput,
        encoderAttentionMask,
      ]);
    var encoderMultiHeadAttention = tf.layers
      .timeDistributed({
        layer: new NodeMultiHeadAttention({
          numHeads: h,
          keyDim: dModel / h,
        }),
      })
      .apply(encoderMultiHeadAttentionConcatter);
    var encoderDropout0 = tf.layers
      .timeDistributed({
        layer: tf.layers.dropout({ rate: pDropout }),
      })
      .apply(encoderMultiHeadAttention);
    var encoderAdd0 = tf.layers
      .add()
      .apply([encoderDropout0, lastEncoderOutput]);
    var encoderNorm0 = tf.layers
      .timeDistributed({
        layer: tf.layers.layerNormalization(),
      })
      .apply(encoderAdd0);
    var encoderFF0 = tf.layers
      .timeDistributed({
        layer: tf.layers.dense({ units: dFF, activation: "relu" }),
      })
      .apply(encoderNorm0);
    var encoderFF1 = tf.layers
      .timeDistributed({
        layer: tf.layers.dense({ units: dModel, activation: "linear" }),
      })
      .apply(encoderFF0);
    var encoderDropout1 = tf.layers
      .timeDistributed({
        layer: tf.layers.dropout({ rate: pDropout }),
      })
      .apply(encoderFF1);
    var encoderAdd1 = tf.layers.add().apply([encoderNorm0, encoderDropout1]);
    var encoderNorm1 = tf.layers
      .timeDistributed({
        layer: tf.layers.layerNormalization(),
      })
      .apply(encoderAdd1);
    lastEncoderOutput = encoderNorm1;
  }
  var encoderReshape0 = tf.layers
    .reshape({ targetShape: [null, maxLen * dModel] })
    .apply(lastEncoderOutput);
  var encoderRNNLayer = new RNNMultiHeadAttention({
    cell: new RNNMultiHeadAttentionCell(h, dModel / h, maxLen),
  });
  var encoderRNN = encoderRNNLayer.apply(encoderReshape0);
  var encoderReshape1 = tf.layers
    .reshape({ targetShape: [maxLen, dModel] })
    .apply(encoderRNN);
  var decoderInput = tf.input({ shape: [null, maxLen] });
  var decoderStandaloneInput = tf.input({ shape: [maxLen, dModel] });
  var decoderStandaloneRNNInput = tf.input({
    shape: [null, maxLen, dModel],
  });
  var decoderStandaloneMaskInput = tf.input({
    shape: [null, maxLen],
  });
  var decoderOnes = tf.layers
    .dense({
      units: maxLen,
      kernelInitializer: "zeros",
      biasInitializer: "ones",
      trainable: false,
    })
    .apply(decoderInput);
  var decoderAttentionMask = tf.layers
    .minimum()
    .apply([decoderInput, decoderOnes]);
  var decoderEmbeddingLayer = tf.layers.timeDistributed({
    layer: tf.layers.embedding({
      inputDim: depthDecoder,
      outputDim: dModel,
      maskZero: true,
    }),
  });
  var decoderEmbedding = decoderEmbeddingLayer.apply(decoderInput);
  var decoderConstantPositionalEncoding = tf.layers
    .timeDistributed({
      layer: new NodePositionalEncoding(maxLen, dModel),
    })
    .apply(decoderEmbedding);
  var decoderPositionalEncoding = tf.layers
    .add()
    .apply([decoderEmbedding, decoderConstantPositionalEncoding]);
  var decoderReshape0 = tf.layers
    .reshape({ targetShape: [null, maxLen * dModel] })
    .apply(decoderPositionalEncoding);
  var decoderRNNLayer = new RNNMultiHeadAttention({
    cell: new RNNMultiHeadAttentionCell(h, dModel / h, maxLen),
    returnSequences: true,
  });
  var decoderRNN = decoderRNNLayer.apply(decoderReshape0);
  var decoderReshape1 = tf.layers
    .reshape({ targetShape: [null, maxLen, dModel] })
    .apply(decoderRNN);
  let lastDecoderOutput = decoderReshape1;
  let lastDecoderStandaloneOutput = decoderStandaloneRNNInput;
  for (let i = 0; i < layers; i++) {
    var decoderMaskedMultiHeadAttentionConcatterLayer =
      new MultiHeadAttentionConcatter();
    var decoderMaskedMultiHeadAttentionConcatter =
      decoderMaskedMultiHeadAttentionConcatterLayer.apply([
        lastDecoderOutput,
        lastDecoderOutput,
      ]);
    var decoderStandaloneMaskedMultiHeadAttentionConcatter =
      decoderMaskedMultiHeadAttentionConcatterLayer.apply([
        lastDecoderStandaloneOutput,
        lastDecoderStandaloneOutput,
      ]);
    var decoderMaskedMultiHeadAttentionLayer = tf.layers.timeDistributed({
      layer: new NodeMultiHeadAttention({
        numHeads: h,
        keyDim: dModel / h,
      }),
    });
    var decoderMaskedMultiHeadAttention =
      decoderMaskedMultiHeadAttentionLayer.apply(
        decoderMaskedMultiHeadAttentionConcatter,
        {
          useCausalMask: true,
        }
      );
    var decoderStandaloneMaskedMultiHeadAttention =
      decoderMaskedMultiHeadAttentionLayer.apply(
        decoderStandaloneMaskedMultiHeadAttentionConcatter,
        {
          useCausalMask: true,
        }
      );
    var decoderDropoutLayer0 = tf.layers.timeDistributed({
      layer: tf.layers.dropout({ rate: pDropout }),
    });
    var decoderDropout0 = decoderDropoutLayer0.apply(
      decoderMaskedMultiHeadAttention
    );
    var decoderStandaloneDropout0 = decoderDropoutLayer0.apply(
      decoderStandaloneMaskedMultiHeadAttention
    );
    var decoderAddLayer0 = tf.layers.add();
    var decoderAdd0 = decoderAddLayer0.apply([
      decoderDropout0,
      lastDecoderOutput,
    ]);
    var decoderStandaloneAdd0 = decoderAddLayer0.apply([
      decoderStandaloneDropout0,
      lastDecoderStandaloneOutput,
    ]);
    var decoderNormLayer0 = tf.layers.timeDistributed({
      layer: tf.layers.layerNormalization(),
    });
    var decoderNorm0 = decoderNormLayer0.apply(decoderAdd0);
    var decoderStandaloneNorm0 = decoderNormLayer0.apply(decoderStandaloneAdd0);
    var decoderMultiHeadAttentionConcatterLayer =
      new MultiHeadAttentionConcatter();
    var decoderMultiHeadAttentionConcatter =
      decoderMultiHeadAttentionConcatterLayer.apply([
        decoderNorm0,
        encoderReshape1,
        decoderAttentionMask,
      ]);
    var decoderStandaloneMultiHeadAttentionConcatter =
      decoderMultiHeadAttentionConcatterLayer.apply([
        decoderStandaloneNorm0,
        decoderStandaloneInput,
        decoderStandaloneMaskInput,
      ]);
    var decoderMultiHeadAttentionLayer = tf.layers.timeDistributed({
      layer: new NodeMultiHeadAttention({
        numHeads: h,
        keyDim: dModel / h,
      }),
    });
    var decoderMultiHeadAttention = decoderMultiHeadAttentionLayer.apply(
      decoderMultiHeadAttentionConcatter
    );
    var decoderStandaloneMultiHeadAttention =
      decoderMultiHeadAttentionLayer.apply(
        decoderStandaloneMultiHeadAttentionConcatter
      );
    var decoderDropoutLayer1 = tf.layers.timeDistributed({
      layer: tf.layers.dropout({ rate: pDropout }),
    });
    var decoderDropout1 = decoderDropoutLayer1.apply(decoderMultiHeadAttention);
    var decoderStandaloneDropout1 = decoderDropoutLayer1.apply(
      decoderStandaloneMultiHeadAttention
    );
    var decoderAddLayer1 = tf.layers.add();
    var decoderAdd1 = decoderAddLayer1.apply([decoderDropout1, decoderAdd0]);
    var decoderStandaloneAdd1 = decoderAddLayer1.apply([
      decoderStandaloneDropout1,
      decoderStandaloneAdd0,
    ]);
    var decoderNormLayer1 = tf.layers.timeDistributed({
      layer: tf.layers.layerNormalization(),
    });
    var decoderNorm1 = decoderNormLayer1.apply(decoderAdd1);
    var decoderStandaloneNorm1 = decoderNormLayer1.apply(decoderStandaloneAdd1);
    var decoderFFLayer0 = tf.layers.timeDistributed({
      layer: tf.layers.dense({ units: dFF, activation: "relu" }),
    });
    var decoderFF0 = decoderFFLayer0.apply(decoderNorm1);
    var decoderStandaloneFF0 = decoderFFLayer0.apply(decoderStandaloneNorm1);
    var decoderFFLayer1 = tf.layers.timeDistributed({
      layer: tf.layers.dense({
        units: dModel,
        activation: "linear",
      }),
    });
    var decoderFF1 = decoderFFLayer1.apply(decoderFF0);
    var decoderStandaloneFF1 = decoderFFLayer1.apply(decoderStandaloneFF0);
    var decoderDropoutLayer2 = tf.layers.timeDistributed({
      layer: tf.layers.dropout({ rate: pDropout }),
    });
    var decoderDropout2 = decoderDropoutLayer2.apply(decoderFF1);
    var decoderStandaloneDropout2 =
      decoderDropoutLayer2.apply(decoderStandaloneFF1);
    var decoderAddLayer2 = tf.layers.add();
    var decoderAdd2 = decoderAddLayer2.apply([decoderDropout2, decoderAdd1]);
    var decoderStandaloneAdd2 = decoderAddLayer2.apply([
      decoderStandaloneDropout2,
      decoderStandaloneAdd1,
    ]);
    var decoderNormLayer2 = tf.layers.timeDistributed({
      layer: tf.layers.layerNormalization(),
    });
    var decoderNorm2 = decoderNormLayer2.apply(decoderAdd2);
    var decoderStandaloneNorm2 = decoderNormLayer2.apply(decoderStandaloneAdd2);
    lastDecoderOutput = decoderNorm2;
    lastDecoderStandaloneOutput = decoderStandaloneNorm2;
  }
  var decoderDenseLayer = tf.layers.timeDistributed({
    layer: tf.layers.dense({
      units: depthTarget,
      activation: "softmax",
    }),
  });
  var decoderDense = decoderDenseLayer.apply(lastDecoderOutput);
  var decoderStandaloneDense = decoderDenseLayer.apply(
    lastDecoderStandaloneOutput
  );
  var trainer = tf.model({
    inputs: [encoderInput, decoderInput],
    outputs: decoderDense,
  });
  var optimizer = new tf.AdagradOptimizer(0.001);
  trainer.compile({
    optimizer,
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"],
  });
  var encoder = tf.model({
    inputs: encoderInput,
    outputs: lastEncoderOutput,
  });
  var decoder = tf.model({
    inputs: [
      decoderStandaloneRNNInput,
      decoderStandaloneMaskInput,
      decoderStandaloneInput,
    ],
    outputs: decoderStandaloneDense,
  });
  return {
    trainer,
    encoder,
    decoder,
    encoderRNNLayer,
    decoderRNNLayer,
    decoderEmbeddingLayer,
    a: encoderInput,
    b: encoderReshape1,
  };
};
