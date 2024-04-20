import * as tf from "@tensorflow/tfjs-node-gpu";

export const weights2JSONL = async (model) => {
  const out = [];
  for (const weight of model.getWeights(true)) {
    out.push(JSON.stringify(await weight.array()));
  }
  return out.join("\n");
};
export const jsonl2Weights = (weights) => {
  const input = weights.split("\n");
  const tensors = Array(input.length);
  for (let i = 0; i < tensors.length; i++) {
    tensors[i] = tf.tensor(JSON.parse(input[i]));
  }
  return tensors;
};
export const weights2ArrayBuffer = async (model) => {
  let outArray = [];
  for (const weight of model.getWeights(true)) {
    const array = await weight.data();
    outArray.push(weight.shape.length);
    outArray.push(...weight.shape);
    outArray.push(array.length);
    outArray = outArray.concat(Array.from(array));
  }
  return new Float32Array(outArray).buffer;
};
export const weights2Blob = async (model) => {
  return new Blob([await weights2ArrayBuffer(model)]);
};
export const arrayBuffer2Weights = (buffer) => {
  const inArray = new Float32Array(buffer);
  const outArray = [];
  for (let i = 0; i < inArray.length; ) {
    const shapeLength = inArray[i++];
    const shape = Array.from(inArray.slice(i, i + shapeLength));
    i += shapeLength;
    const dataLength = inArray[i++];
    const data = inArray.slice(i, i + dataLength);
    i += dataLength;
    const tensor = tf.tensor(data, shape);
    outArray.push(tensor);
  }
  return outArray;
};
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
