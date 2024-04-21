import * as tf from "@tensorflow/tfjs-node-gpu";
import fs from "fs";
/*

Important note: Copy napi-v9/tensorflow.dll into napi-v8/tensorflow.dll

*/
import num2char from "./num2char.json"; // with { type: "json" };
import tokens from "./tokens.json"; // with { type: "json" };
import useNodeExtendedTransformer from "./useNodeExtendedTransformer.mjs";
import { weights2ArrayBuffer, arrayBuffer2Weights } from "./initTensorflow.mjs";

tf.setBackend("cpu");

tf.registerGradient({
  kernelName: tf.Einsum,
  inputsToSave: ["0", "1"],
  gradFunc(dy, saved, attrs) {
    const equation = attrs["equation"];
    const splittedEquation0 = equation.split(",");
    const splittedEquation1 = splittedEquation0[1].split("->");
    return {
      0: () =>
        tf.einsum(
          `${splittedEquation1[1]},${splittedEquation1[0]}->${splittedEquation0[0]}`,
          tf.onesLike(dy),
          saved[1]
        ),
      1: () =>
        tf.einsum(
          `${splittedEquation1[1]},${splittedEquation0[0]}->${splittedEquation1[0]}`,
          tf.onesLike(dy),
          saved[0]
        ),
    };
  },
});
const depth = num2char.length;
const maxLen = 8;
const models = useNodeExtendedTransformer({
  dModel: 128,
  dFF: 256,
  pDropout: 0.1,
  h: 4,
  maxLen,
  depthEncoder: depth,
  depthDecoder: depth,
  depthTarget: depth,
  layers: 4,
});
models.trainer.summary();
const batchSize = 16;
const loader = function* () {
  const zeros = Array(maxLen).fill(0);
  while (true) {
    const encoderInput = [];
    const decoderInput = [];
    const decoderOutput = [];
    for (let j = 0; j < batchSize; j++) {
      const i = Math.floor(Math.random() * (tokens.length - 8)) + 8;
      const ea = tokens.slice(i - 8, i - Math.floor(Math.random() * 8) - 1);
      for (let k = 0; k < ea.length; k++) {
        const ec = (ea[k] = ea[k].slice());
        ec[ec.length] = 2;
      }
      const flatten = ea.flat();
      flatten.push(
        ...Array(
          Math.floor(flatten.length / maxLen) * maxLen + maxLen - flatten.length
        ).fill(0)
      );
      const e = tf.tidy(() => {
        return tf
          .tensor(flatten)
          .reshape([flatten.length / maxLen, maxLen])
          .arraySync();
      });
      encoderInput.push(e);
      const da = tokens[i].slice();
      da.unshift(1);
      da.push(
        ...Array(
          Math.floor(da.length / maxLen) * maxLen + maxLen - da.length
        ).fill(0)
      );
      const d = tf.tidy(() => {
        return tf
          .tensor(da)
          .reshape([da.length / maxLen, maxLen])
          .arraySync();
      });
      decoderInput.push(d);
      const de = tokens[i].slice();
      de.push(2);
      de.push(
        ...Array(
          Math.floor(de.length / maxLen) * maxLen + maxLen - de.length
        ).fill(0)
      );
      const o = tf.tidy(() => {
        return tf
          .tensor(de)
          .reshape([de.length / maxLen, maxLen])
          .arraySync();
      });
      decoderOutput.push(o);
      if (e.length < 2) {
        e.push(zeros);
      }
      if (d.length < 2) {
        d.push(zeros);
      }
      if (o.length < 2) {
        o.push(zeros);
      }
    }
    let encoderInputMax = 0;
    for (const e of encoderInput) {
      encoderInputMax = Math.max(e.length, encoderInputMax);
    }
    for (const e of encoderInput) {
      e.push(...Array(encoderInputMax - e.length).fill(zeros));
    }
    let decoderInputMax = 0;
    for (const e of decoderInput) {
      decoderInputMax = Math.max(e.length, decoderInputMax);
    }
    for (const e of decoderInput) {
      e.push(...Array(decoderInputMax - e.length).fill(zeros));
    }
    let decoderOutputMax = 0;
    for (const e of decoderOutput) {
      decoderOutputMax = Math.max(e.length, decoderOutputMax);
    }
    for (const e of decoderOutput) {
      e.push(...Array(decoderOutputMax - e.length).fill(zeros));
    }
    yield {
      xs: [tf.tensor(encoderInput), tf.tensor(decoderInput)],
      ys: tf.tensor(decoderOutput),
    };
  }
};
const train = async () => {
  await models.trainer.fitDataset(tf.data.generator(loader), {
    batchesPerEpoch: 32,
    epochs: 512,
    callbacks: {
      async onEpochEnd() {
        save(await weights2ArrayBuffer(models.trainer));
      },
    },
  });
  save(await weights2ArrayBuffer(models.trainer));
};

const load = async (data) => {
  const weights = arrayBuffer2Weights(data);
  for (const layer of models.trainer.layers) {
    if (layer.trainable && layer.trainableWeights.length) {
      layer.setWeights(weights.splice(0, layer.trainableWeights.length));
    }
  }
};
let saveOffset = 1;
const save = (buffer) => {
  const path = "./weights/weights-" + saveOffset++ + ".bin";
  console.log(path);
  fs.writeFile(path, Buffer.from(buffer), (err) => {
    err && console.error(err);
  });
};
// fs.readFile("./weights/weights-107.bin", null, (err, data) => {
//   err && console.error(err);
//   load(data.buffer);
train();
// });
