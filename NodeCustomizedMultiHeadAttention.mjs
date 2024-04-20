import { MultiHeadAttention } from "@tensorflow/tfjs-layers/dist/layers/nlp/multihead_attention.js";
import * as tf from "@tensorflow/tfjs-node-gpu";
class CustomizedMultiHeadAttention extends MultiHeadAttention {
  build(inputShape) {
    const shape =
      inputShape[0] === null || typeof inputShape[0] === "number"
        ? [
            [inputShape[0], ...inputShape.slice(2)],
            [inputShape[0], ...inputShape.slice(2)],
          ]
        : inputShape;
    this.buildFromSignature(shape[0], shape[1], shape[2]);
    this._queryDense.build(shape[0]);
    this._keyDense.build(shape[1]);
    this._valueDense.build(shape[2] || shape[1]);
    this._outputDense.build([
      shape[0][0],
      shape[0][1],
      this.numHeads,
      shape[0][2] / this.numHeads,
    ]);
    this.built = true;
  }
  call(inputs, kwargs) {
    if (inputs.rank === 4) {
      const result = tf.tidy(() => {
        var _a;
        const buffer = inputs.transpose([1, 0, 2, 3]).unstack();
        inputs = buffer[0];
        kwargs.value = buffer[1];
        // @ts-ignore
        kwargs.attentionMask =
          (_a = buffer[2].unstack(2)[0]) === null || _a === void 0
            ? void 0
            : _a
                .expandDims(2)
                .tile([1, 1, buffer[2].shape[1]])
                .expandDims(1)
                .tile([1, this.numHeads, 1, 1]);
        return [inputs, kwargs.value, kwargs.attentionMask];
      });
      return super.call(
        result[0],
        Object.assign(Object.assign({}, kwargs), {
          value: result[1],
          attentionMask: result[2],
        })
      );
    }
    return super.call(inputs, kwargs);
  }
  get trainableWeights() {
    return [
      ...this.queryDense.trainableWeights,
      ...this.keyDense.trainableWeights,
      ...this.valueDense.trainableWeights,
      ...this.outputDense.trainableWeights,
    ];
  }
  computeOutputShape(inputShape) {
    const shape =
      inputShape[0] === null || typeof inputShape[0] === "number"
        ? [
            [inputShape[0], ...inputShape.slice(2)],
            [inputShape[0], ...inputShape.slice(2)],
          ]
        : inputShape;
    const [queryShape] = shape;
    // @ts-ignore
    if (this._outputShape) {
      // @ts-ignore
      return queryShape.slice(0, -1).concat(this._outputShape);
    }
    return queryShape;
  }
  getWeights() {
    return [
      ...this._queryDense.getWeights(),
      ...this._keyDense.getWeights(),
      ...this._valueDense.getWeights(),
      ...this._outputDense.getWeights(),
    ];
  }
  setWeights(weights) {
    this._queryDense.setWeights(weights.slice(0, 2));
    this._keyDense.setWeights(weights.slice(2, 4));
    this._valueDense.setWeights(weights.slice(4, 6));
    this._outputDense.setWeights(weights.slice(6, 8));
  }
}
export { CustomizedMultiHeadAttention as NodeMultiHeadAttention };
