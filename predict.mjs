import * as tf from "@tensorflow/tfjs-node-gpu";
import fs from "fs";

import num2char from "./num2char.json"; // with { type: "json" };
// import tokens from "./tokens.json"; // with { type: "json" };
import char2num from "./char2num.json"; // with { type: "json" };
import useNodeExtendedTransformer from "./useNodeExtendedTransformer.mjs";
import { arrayBuffer2Weights } from "./initTensorflow.mjs";

const depth = num2char.length;
const maxLen = 8;
const models = useNodeExtendedTransformer({
  dModel: 16,
  dFF: 32,
  pDropout: 0.2,
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
  const output = [];
  if (e.length < 2) {
    e.push(zeros);
  }
  models.encoder.predict(tf.tensor([e]), { batchSize: 1 }).print();

  for (let i = 0; i < 16; i++) {
    tf.tidy(() => {});
  }
};
fs.readFile("./weights/weights-106.bin", null, (err, data) => {
  err && console.error(err);
  load(data.buffer);
  predict();
});
