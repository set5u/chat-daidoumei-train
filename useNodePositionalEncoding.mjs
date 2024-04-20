import * as tf from "@tensorflow/tfjs-node-gpu";
import { Initializer } from "@tensorflow/tfjs-layers/dist/initializers";
class PositionalEncodingInitializer extends Initializer {
  constructor() {
    super();
  }
  apply(shape) {
    return positionalEncoding(shape[0], shape[1]);
  }
}

export class NodePositionalEncoding extends tf.layers.Layer {
  weight;
  constructor(length, depth) {
    super({ trainable: false });
    this.length = length;
    this.depth = depth;
  }
  build() {
    this.weight = this.addWeight(
      "PositionalEncoding",
      [this.length, this.depth],
      "float32",
      new PositionalEncodingInitializer(),
      undefined,
      false
    );
  }
  call(input) {
    return tf.tidy(() =>
      this.weight.read().expandDims(0).tile([input.shape[0], 1, 1])
    );
  }
}
NodePositionalEncoding.className = "PositionalEncoding";
export const positionalEncoding = (length, depth) => {
  const ret = Array(length);
  for (let i = 0; i < length; i++) {
    const r = [];
    ret[i] = r;
    for (let j = 0; j < depth; j++) {
      r[j] =
        j % 2
          ? Math.sin(i / 10000 ** (j / depth))
          : Math.cos(i / 10000 ** (j / depth));
    }
  }
  return tf.tensor(ret);
};
