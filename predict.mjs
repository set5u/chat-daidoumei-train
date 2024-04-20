import * as tf from "@tensorflow/tfjs-node-gpu";
import fs from "fs";

import num2char from "./num2char.json"; // with { type: "json" };
// import tokens from "./tokens.json"; // with { type: "json" };
import char2num from "./char2num.json"; // with { type: "json" };
import useNodeExtendedTransformer from "./useNodeExtendedTransformer.mjs";
import { arrayBuffer2Weights } from "./initTensorflow.mjs";
import { positionalEncoding } from "./useNodePositionalEncoding.mjs";

const depth = num2char.length;
const maxLen = 8;
const models = useNodeExtendedTransformer({
  dModel: 16,
  dFF: 32,
  pDropout: 0,
  h: 4,
  maxLen,
  depthEncoder: depth,
  depthDecoder: depth,
  depthTarget: depth,
  layers: 4,
});

const load = async (data) => {
  const weights = arrayBuffer2Weights(data);
  for (const layer of models.trainer.layers) {
    if (layer.trainable && layer.trainableWeights.length) {
      layer.setWeights(weights.splice(0, layer.trainableWeights.length));
    }
  }
};
const print = (t) => {
  t.print();
  return t;
};

const predict = async () => {
  Error.stackTraceLimit = Infinity;
  models.encoder.summary();
  const prompt = "数学やった？";
  const encoderInput = [3, 51, 51, 454, 703, 5];
  for (let i = 0; ; i++) {
    const reg = new RegExp(`^.{${i}}(.{0,${1}})`, "u");
    const m = prompt.match(reg)[1];
    if (!m) {
      break;
    }
    encoderInput.push(char2num[m]);
  }
  encoderInput.push(
    ...Array(
      Math.floor(encoderInput.length / maxLen) * maxLen +
        maxLen -
        encoderInput.length
    ).fill(0)
  );
  const e = tf.tidy(() => {
    return tf
      .tensor(encoderInput)
      .reshape([encoderInput.length / maxLen, maxLen])
      .arraySync();
  });

  if (e.length < 2) {
    e.push(zeros);
  }
  const test = tf
    .model({ inputs: models.a, outputs: models.b })
    .apply(tf.tensor([e]), { training: true });
  test.print();
  const encoderOutput = tf.tidy(() => {
    return models.encoder.apply(tf.tensor([e]), { training: true });
  });
  encoderOutput.print();
  const encoderRNNOutput = tf.tidy(() => {
    return models.encoderRNNLayer
      .apply(encoderOutput.reshape([1, -1, maxLen * 16]))
      .reshape([maxLen, 16]);
  });
  encoderRNNOutput.print();
  encoderOutput.dispose();
  const output = [1];
  for (let i = 0; i < 1; i++) {
    tf.tidy(() => {
      const decoderInputArray = Array(
        Math.floor(output.length / maxLen) * maxLen + maxLen
      ).fill(0);
      decoderInputArray[0] = 1;
      output.forEach((v, i) => (decoderInputArray[i + 1] = v));
      const decoderInput = tf
        .tensor(decoderInputArray)
        .reshape([1, -1, maxLen]);
      const embeddingOut = models.decoderEmbeddingLayer.apply(
        decoderInput.reshape([1, -1, maxLen])
      );
      const rnnOut = models.decoderRNNLayer
        .apply(
          embeddingOut
            .add(positionalEncoding(maxLen, 16).expandDims(0))
            .reshape([1, -1, maxLen * 16])
        )
        .reshape([1, -1, maxLen, 16]);
      const decoderOut = models.decoder.apply(
        [
          rnnOut,
          tf.maximum(decoderInput.onesLike(), decoderInput),
          encoderRNNOutput,
        ],
        { training: true }
      );
      // .argMax(2);
      return decoderOut;
    }).print();
  }
};
// fs.readFile("./weights/weights-106.bin", null, (err, data) => {
//   err && console.error(err);
// load(data.buffer);
// const weights = models.trainer.getWeights();
// for (const layer of models.trainer.layers) {
//   if (layer.trainable && layer.trainableWeights.length) {
//     console.log(layer.name, ":::");
//     console.log(
//       weights
//         .splice(0, layer.trainableWeights.length)
//         .map((v) => v.toString())
//     );
//   }
// }
predict();
// });
