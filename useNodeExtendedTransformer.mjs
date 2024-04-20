import { NodeMultiHeadAttention } from "./NodeCustomizedMultiHeadAttention.mjs";
import * as tf from "@tensorflow/tfjs-node-gpu";
import {
  NodePositionalEncoding,
  positionalEncoding,
} from "./useNodePositionalEncoding.mjs";
class RNNMultiHeadAttentionCell extends tf.layers.RNNCell {
  constructor(numHeads, keyDim, maxLen) {
    super();
    this.maxLen = maxLen;
    this.stateSize = keyDim * numHeads * maxLen;
    this.attention = new NodeMultiHeadAttention({ numHeads, keyDim });
  }
  call(inputs, kwargs) {
    super.call(inputs, kwargs);
    const batchSize = inputs[0].shape[0];
    const ret = tf.tidy(() =>
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
  constructor(args) {
    super(args);
    this.length = args.length;
    this.depth = args.depth;
  }
  getInitialState(inputs) {
    return tf.tidy(() => [
      positionalEncoding(this.length, this.depth)
        .reshape([-1, 1, inputs.shape[2]])
        .tile([1, inputs.shape[0], 1]),
    ]);
  }
}
class MultiHeadAttentionConcatter extends tf.layers.Layer {
  call(inputs) {
    return tf.tidy(() => {
      let query = inputs[0];
      let kv = inputs[1];
      let mask = inputs[2];
      const dModel = query.shape[query.shape.length - 1];
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
    const input = Array.isArray(inputShape[0]) ? inputShape[0] : inputShape;
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
  const encoderInput = tf.input({ shape: [null, maxLen] });
  const encoderOnes = tf.layers
    .dense({
      units: maxLen,
      kernelInitializer: "zeros",
      biasInitializer: "ones",
      trainable: false,
    })
    .apply(encoderInput);
  const encoderAttentionMask = tf.layers
    .minimum()
    .apply([encoderInput, encoderOnes]);
  const encoderEmbedding = tf.layers
    .timeDistributed({
      layer: tf.layers.embedding({
        inputDim: depthEncoder,
        outputDim: dModel,
        maskZero: true,
      }),
    })
    .apply(encoderInput);
  const encoderConstantPositionalEncoding = tf.layers
    .timeDistributed({
      layer: new NodePositionalEncoding(maxLen, dModel),
    })
    .apply(encoderEmbedding);
  const encoderPositionalEncoding = tf.layers
    .add()
    .apply([encoderEmbedding, encoderConstantPositionalEncoding]);
  let lastEncoderOutput = encoderPositionalEncoding;
  for (let i = 0; i < layers; i++) {
    const encoderMultiHeadAttentionConcatter =
      new MultiHeadAttentionConcatter().apply([
        lastEncoderOutput,
        lastEncoderOutput,
        encoderAttentionMask,
      ]);
    const encoderMultiHeadAttention = tf.layers
      .timeDistributed({
        layer: new NodeMultiHeadAttention({
          numHeads: h,
          keyDim: dModel / h,
        }),
      })
      .apply(encoderMultiHeadAttentionConcatter);
    const encoderDropout0 = tf.layers
      .timeDistributed({
        layer: tf.layers.dropout({ rate: pDropout }),
      })
      .apply(encoderMultiHeadAttention);
    const encoderAdd0 = tf.layers
      .add()
      .apply([encoderDropout0, lastEncoderOutput]);
    const encoderNorm0 = tf.layers
      .timeDistributed({
        layer: tf.layers.layerNormalization(),
      })
      .apply(encoderAdd0);
    const encoderFF0 = tf.layers
      .timeDistributed({
        layer: tf.layers.dense({ units: dFF, activation: "relu" }),
      })
      .apply(encoderNorm0);
    const encoderFF1 = tf.layers
      .timeDistributed({
        layer: tf.layers.dense({ units: dModel, activation: "linear" }),
      })
      .apply(encoderFF0);
    const encoderDropout1 = tf.layers
      .timeDistributed({
        layer: tf.layers.dropout({ rate: pDropout }),
      })
      .apply(encoderFF1);
    const encoderAdd1 = tf.layers.add().apply([encoderNorm0, encoderDropout1]);
    const encoderNorm1 = tf.layers
      .timeDistributed({
        layer: tf.layers.layerNormalization(),
      })
      .apply(encoderAdd1);
    lastEncoderOutput = encoderNorm1;
  }
  const encoderReshape0 = tf.layers
    .reshape({ targetShape: [null, maxLen * dModel] })
    .apply(lastEncoderOutput);
  const encoderRNNLayer = new RNNMultiHeadAttention({
    cell: new RNNMultiHeadAttentionCell(h, dModel / h, maxLen),
    length: maxLen,
    depth: dModel,
  });
  const encoderRNN = encoderRNNLayer.apply(encoderReshape0);
  const encoderReshape1 = tf.layers
    .reshape({ targetShape: [maxLen, dModel] })
    .apply(encoderRNN);
  const decoderInput = tf.input({ shape: [null, maxLen] });
  const decoderStandaloneInput = tf.input({ shape: [maxLen, dModel] });
  const decoderStandaloneRNNInput = tf.input({
    shape: [null, maxLen, dModel],
  });
  const decoderStandaloneMaskInput = tf.input({
    shape: [null, maxLen],
  });
  const decoderOnes = tf.layers
    .dense({
      units: maxLen,
      kernelInitializer: "zeros",
      biasInitializer: "ones",
      trainable: false,
    })
    .apply(decoderInput);
  const decoderAttentionMask = tf.layers
    .minimum()
    .apply([decoderInput, decoderOnes]);
  const decoderEmbeddingLayer = tf.layers.timeDistributed({
    layer: tf.layers.embedding({
      inputDim: depthDecoder,
      outputDim: dModel,
      maskZero: true,
    }),
  });
  const decoderEmbedding = decoderEmbeddingLayer.apply(decoderInput);
  const decoderConstantPositionalEncoding = tf.layers
    .timeDistributed({
      layer: new NodePositionalEncoding(maxLen, dModel),
    })
    .apply(decoderEmbedding);
  const decoderPositionalEncoding = tf.layers
    .add()
    .apply([decoderEmbedding, decoderConstantPositionalEncoding]);
  const decoderReshape0 = tf.layers
    .reshape({ targetShape: [null, maxLen * dModel] })
    .apply(decoderPositionalEncoding);
  const decoderRNNLayer = new RNNMultiHeadAttention({
    cell: new RNNMultiHeadAttentionCell(h, dModel / h, maxLen),
    length: maxLen,
    depth: dModel,
    returnSequences: true,
  });
  const decoderRNN = decoderRNNLayer.apply(decoderReshape0);
  const decoderReshape1 = tf.layers
    .reshape({ targetShape: [null, maxLen, dModel] })
    .apply(decoderRNN);
  let lastDecoderOutput = decoderReshape1;
  let lastDecoderStandaloneOutput = decoderStandaloneRNNInput;
  for (let i = 0; i < layers; i++) {
    const decoderMaskedMultiHeadAttentionConcatterLayer =
      new MultiHeadAttentionConcatter();
    const decoderMaskedMultiHeadAttentionConcatter =
      decoderMaskedMultiHeadAttentionConcatterLayer.apply([
        lastDecoderOutput,
        lastDecoderOutput,
      ]);
    const decoderStandaloneMaskedMultiHeadAttentionConcatter =
      decoderMaskedMultiHeadAttentionConcatterLayer.apply([
        lastDecoderStandaloneOutput,
        lastDecoderStandaloneOutput,
      ]);
    const decoderMaskedMultiHeadAttentionLayer = tf.layers.timeDistributed({
      layer: new NodeMultiHeadAttention({
        numHeads: h,
        keyDim: dModel / h,
      }),
    });
    const decoderMaskedMultiHeadAttention =
      decoderMaskedMultiHeadAttentionLayer.apply(
        decoderMaskedMultiHeadAttentionConcatter,
        {
          useCausalMask: true,
        }
      );
    const decoderStandaloneMaskedMultiHeadAttention =
      decoderMaskedMultiHeadAttentionLayer.apply(
        decoderStandaloneMaskedMultiHeadAttentionConcatter,
        {
          useCausalMask: true,
        }
      );
    const decoderDropoutLayer0 = tf.layers.timeDistributed({
      layer: tf.layers.dropout({ rate: pDropout }),
    });
    const decoderDropout0 = decoderDropoutLayer0.apply(
      decoderMaskedMultiHeadAttention
    );
    const decoderStandaloneDropout0 = decoderDropoutLayer0.apply(
      decoderStandaloneMaskedMultiHeadAttention
    );
    const decoderAddLayer0 = tf.layers.add();
    const decoderAdd0 = decoderAddLayer0.apply([
      decoderDropout0,
      lastDecoderOutput,
    ]);
    const decoderStandaloneAdd0 = decoderAddLayer0.apply([
      decoderStandaloneDropout0,
      lastDecoderStandaloneOutput,
    ]);
    const decoderNormLayer0 = tf.layers.timeDistributed({
      layer: tf.layers.layerNormalization(),
    });
    const decoderNorm0 = decoderNormLayer0.apply(decoderAdd0);
    const decoderStandaloneNorm0 = decoderNormLayer0.apply(
      decoderStandaloneAdd0
    );
    const decoderMultiHeadAttentionConcatterLayer =
      new MultiHeadAttentionConcatter();
    const decoderMultiHeadAttentionConcatter =
      decoderMultiHeadAttentionConcatterLayer.apply([
        decoderNorm0,
        encoderReshape1,
        decoderAttentionMask,
      ]);
    const decoderStandaloneMultiHeadAttentionConcatter =
      decoderMultiHeadAttentionConcatterLayer.apply([
        decoderStandaloneNorm0,
        decoderStandaloneInput,
        decoderStandaloneMaskInput,
      ]);
    const decoderMultiHeadAttentionLayer = tf.layers.timeDistributed({
      layer: new NodeMultiHeadAttention({
        numHeads: h,
        keyDim: dModel / h,
      }),
    });
    const decoderMultiHeadAttention = decoderMultiHeadAttentionLayer.apply(
      decoderMultiHeadAttentionConcatter
    );
    const decoderStandaloneMultiHeadAttention =
      decoderMultiHeadAttentionLayer.apply(
        decoderStandaloneMultiHeadAttentionConcatter
      );
    const decoderDropoutLayer1 = tf.layers.timeDistributed({
      layer: tf.layers.dropout({ rate: pDropout }),
    });
    const decoderDropout1 = decoderDropoutLayer1.apply(
      decoderMultiHeadAttention
    );
    const decoderStandaloneDropout1 = decoderDropoutLayer1.apply(
      decoderStandaloneMultiHeadAttention
    );
    const decoderAddLayer1 = tf.layers.add();
    const decoderAdd1 = decoderAddLayer1.apply([decoderDropout1, decoderAdd0]);
    const decoderStandaloneAdd1 = decoderAddLayer1.apply([
      decoderStandaloneDropout1,
      decoderStandaloneAdd0,
    ]);
    const decoderNormLayer1 = tf.layers.timeDistributed({
      layer: tf.layers.layerNormalization(),
    });
    const decoderNorm1 = decoderNormLayer1.apply(decoderAdd1);
    const decoderStandaloneNorm1 = decoderNormLayer1.apply(
      decoderStandaloneAdd1
    );
    const decoderFFLayer0 = tf.layers.timeDistributed({
      layer: tf.layers.dense({ units: dFF, activation: "relu" }),
    });
    const decoderFF0 = decoderFFLayer0.apply(decoderNorm1);
    const decoderStandaloneFF0 = decoderFFLayer0.apply(decoderStandaloneNorm1);
    const decoderFFLayer1 = tf.layers.timeDistributed({
      layer: tf.layers.dense({
        units: dModel,
        activation: "linear",
      }),
    });
    const decoderFF1 = decoderFFLayer1.apply(decoderFF0);
    const decoderStandaloneFF1 = decoderFFLayer1.apply(decoderStandaloneFF0);
    const decoderDropoutLayer2 = tf.layers.timeDistributed({
      layer: tf.layers.dropout({ rate: pDropout }),
    });
    const decoderDropout2 = decoderDropoutLayer2.apply(decoderFF1);
    const decoderStandaloneDropout2 =
      decoderDropoutLayer2.apply(decoderStandaloneFF1);
    const decoderAddLayer2 = tf.layers.add();
    const decoderAdd2 = decoderAddLayer2.apply([decoderDropout2, decoderAdd1]);
    const decoderStandaloneAdd2 = decoderAddLayer2.apply([
      decoderStandaloneDropout2,
      decoderStandaloneAdd1,
    ]);
    const decoderNormLayer2 = tf.layers.timeDistributed({
      layer: tf.layers.layerNormalization(),
    });
    const decoderNorm2 = decoderNormLayer2.apply(decoderAdd2);
    const decoderStandaloneNorm2 = decoderNormLayer2.apply(
      decoderStandaloneAdd2
    );
    lastDecoderOutput = decoderNorm2;
    lastDecoderStandaloneOutput = decoderStandaloneNorm2;
  }
  const decoderDenseLayer = tf.layers.timeDistributed({
    layer: tf.layers.dense({
      units: depthTarget,
      activation: "softmax",
    }),
  });
  const decoderDense = decoderDenseLayer.apply(lastDecoderOutput);
  const decoderStandaloneDense = decoderDenseLayer.apply(
    lastDecoderStandaloneOutput
  );
  const trainer = tf.model({
    inputs: [encoderInput, decoderInput],
    outputs: decoderDense,
  });
  const optimizer = new tf.AdagradOptimizer(0.001);
  trainer.compile({
    optimizer,
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"],
  });
  const encoder = tf.model({
    inputs: encoderInput,
    outputs: lastEncoderOutput,
  });
  const decoder = tf.model({
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
