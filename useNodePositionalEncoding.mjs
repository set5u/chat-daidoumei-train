import * as tf from "@tensorflow/tfjs-node-gpu";
export class NodePositionalEncoding extends tf.layers.Layer {
  constructor(length, depth) {
    super({trainable: false});
    this.length = length;
    this.depth = depth;
  }
  call(input) {
    return tf.tidy(() =>
      positionalEncoding(this.length, this.depth)
        .expandDims(0)
        .tile([input.shape[0], 1, 1])
    );
  }
}
NodePositionalEncoding.className = "PositionalEncoding";
const positionalEncoding = (length, depth) => {
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
