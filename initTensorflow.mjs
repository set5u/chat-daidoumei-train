import * as tf from "@tensorflow/tfjs-node-gpu";
// import { multiplyConfig } from "@tensorflow/tfjs-backend-wasm/dist/kernels/Multiply";
// import { reshapeConfig } from "@tensorflow/tfjs-backend-wasm/dist/kernels/Reshape";
// import { sumConfig } from "@tensorflow/tfjs-backend-wasm/dist/kernels/Sum";
// import { transposeConfig } from "@tensorflow/tfjs-backend-wasm/dist/kernels/Transpose";
// import { expandDimsConfig } from "@tensorflow/tfjs-backend-wasm/dist/kernels/ExpandDims";
// import { equalConfig } from "@tensorflow/tfjs-backend-wasm/dist/kernels/Equal";
// import { castConfig } from "@tensorflow/tfjs-backend-wasm/dist/kernels/Cast";
// import { packConfig } from "@tensorflow/tfjs-backend-wasm/dist/kernels/Pack";
// import "@tensorflow/tfjs-backend-wasm";
// import wasmPath0 from "@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm?url";
// import wasmPath1 from "@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm?url";
// import wasmPath2 from "@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm?url";
// import { BackendWasm, setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
// setWasmPaths({
//   "tfjs-backend-wasm.wasm": wasmPath0,
//   "tfjs-backend-wasm-simd.wasm": wasmPath1,
//   "tfjs-backend-wasm-threaded-simd.wasm": wasmPath2,
// });
// await tf.setBackend("wasm");
// const multiply = multiplyConfig.kernelFunc;
// const reshape = reshapeConfig.kernelFunc;
// const sum = sumConfig.kernelFunc;
// const transpose = transposeConfig.kernelFunc;
// const expandDims = expandDimsConfig.kernelFunc;
// const equal = equalConfig.kernelFunc;
// const cast = castConfig.kernelFunc;
// const pack = packConfig.kernelFunc;
// await tf.setBackend("cpu");
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
// function einsum(args: {
//   inputs: tf.EinsumInputs;
//   backend: BackendWasm;
//   attrs: tf.EinsumAttrs;
// }): tf.TensorInfo {
//   const { inputs, backend, attrs } = args;
//   const { equation } = attrs;
//   const tensors = inputs as tf.Tensor[];
//   const { allDims, summedDims, idDims } = tf.backend_util.decodeEinsumEquation(
//     equation,
//     tensors.length,
//   );
//   tf.backend_util.checkEinsumDimSizes(allDims.length, idDims, tensors);
//   const { path, steps } = tf.backend_util.getEinsumComputePath(
//     summedDims,
//     idDims,
//   );
//   const nSteps = steps.length;
//   let out: tf.TensorInfo | null = null;
//   let numDimsRemaining = allDims.length;
//   const tensorsToDispose: tf.TensorInfo[] = [];
//   for (let i = 0; i < nSteps; ++i) {
//     for (const idTerm of steps[i]!) {
//       const { permutationIndices: perm, expandDims: dimsToExpand } =
//         tf.backend_util.getEinsumPermutation(numDimsRemaining, idDims[idTerm]!);
//       let x: tf.TensorInfo;
//       if (tf.backend_util.isIdentityPermutation(perm)) {
//         x = tensors[idTerm]!;
//       } else {
//         x = transpose({
//           inputs: { x: tensors[idTerm] },
//           backend,
//           attrs: { perm },
//         }) as tf.TensorInfo;
//         tensorsToDispose.push(x);
//       }
//       const targetShape: number[] = x.shape.slice();
//       for (let k = 0; k < dimsToExpand.length; ++k) {
//         targetShape.splice(dimsToExpand[k]!, 0, 1);
//       }
//       if (!tf.util.arraysEqual(x.shape, targetShape)) {
//         x = reshape({
//           inputs: { x },
//           backend,
//           attrs: { shape: targetShape },
//         }) as tf.TensorInfo;
//         tensorsToDispose.push(x);
//       }
//       if (out === null) {
//         out = x;
//       } else {
//         // tslint:disable-next-line: no-unnecessary-type-assertion
//         out = multiply({ inputs: { a: x, b: out }, backend }) as tf.TensorInfo;
//         tensorsToDispose.push(out);
//       }
//     }
//     if (i < nSteps - 1) {
//       if (path[i]! >= 0) {
//         out = sum({
//           inputs: { x: out! },
//           backend,
//           attrs: {
//             axis: path[i]! - (allDims.length - numDimsRemaining),
//             keepDims: false,
//           },
//         }) as tf.TensorInfo;
//         tensorsToDispose.push(out!);
//       }
//       numDimsRemaining--;
//     }
//   }
//   // Clean up intermediate tensors.
//   for (const tensorInfo of tensorsToDispose) {
//     if (tensorInfo === out) {
//       continue;
//     }
//     backend.disposeData(tensorInfo);
//   }
//   return out!;
// }
// tf.registerKernel({
//   kernelName: tf.Einsum,
//   backendName: "wasm",
//   kernelFunc: einsum as unknown as tf.KernelFunc,
// });
// function makeTensorInfo(
//   backend: BackendWasm,
//   shape: number[],
//   dtype: tf.DataType,
//   values?: tf.backend_util.BackendValues | string[],
// ): tf.TensorInfo {
//   let outId;
//   if (
//     dtype === "string" &&
//     values != null &&
//     values.length > 0 &&
//     tf.util.isString(values[0]!)
//   ) {
//     const encodedValues = (values as unknown as string[]).map((d) =>
//       tf.util.encodeString(d),
//     );
//     outId = backend.write(encodedValues, shape, dtype);
//   } else {
//     outId = backend.write(values as tf.TypedArray, shape, dtype);
//   }
//   return { dataId: outId, shape, dtype };
// }
// function unsortedSegmentSum(args: {
//   inputs: tf.UnsortedSegmentSumInputs;
//   backend: BackendWasm;
//   attrs: tf.UnsortedSegmentSumAttrs;
// }): tf.TensorInfo {
//   const { inputs, backend, attrs } = args;
//   const { x, segmentIds } = inputs;
//   const { numSegments } = attrs;
//   //   assertNotComplex(x, "unsortedSegmentSum");
//   const xRank = x!.shape.length;
//   const segmentIdsRank = segmentIds!.shape.length;
//   const res = [];
//   const intermediates: tf.TensorInfo[] = [];
//   // Reshape the segment id's so that they can be broadcast with
//   // x. The new shape should be [segmentIds.shape, 1, ..., 1]
//   const numIters = xRank - segmentIdsRank;
//   let $segmentIds = segmentIds;
//   for (let i = 0; i < numIters; ++i) {
//     const expanded = expandDims({
//       inputs: { input: $segmentIds },
//       backend,
//       attrs: { dim: i + 1 },
//     });
//     $segmentIds = expanded as tf.TensorInfo;
//     intermediates.push(expanded as tf.TensorInfo);
//   }
//   for (let i = 0; i < numSegments; ++i) {
//     const scalarValue = tf.util.createScalarValue(
//       i as unknown as "int32",
//       "int32",
//     );
//     const segmentId = makeTensorInfo(backend, [], "int32", scalarValue);
//     const mask = equal({
//       inputs: { a: segmentId, b: $segmentIds },
//       backend,
//     }) as tf.TensorInfo;
//     const maskCasted = cast({
//       inputs: { x: mask },
//       backend,
//       attrs: { dtype: "float32" },
//     });
//     const mul = multiply({
//       inputs: { a: maskCasted as tf.TensorInfo, b: x },
//       backend,
//     }) as tf.TensorInfo;
//     const sumTensorInfo = sum({
//       inputs: { x: mul },
//       backend,
//       attrs: { axis: 0, keepDims: false },
//     });
//     res.push(sumTensorInfo);
//     intermediates.push(segmentId);
//     intermediates.push(mask);
//     intermediates.push(maskCasted as tf.TensorInfo);
//     intermediates.push(mul);
//     intermediates.push(sumTensorInfo as tf.TensorInfo);
//   }
//   const result = pack({
//     inputs: res as unknown as tf.NamedTensorInfoMap,
//     backend,
//     attrs: { axis: 0 },
//   });
//   intermediates.forEach((t) => backend.disposeData(t));
//   return result as tf.TensorInfo;
// }
// tf.registerKernel({
//   kernelName: tf.UnsortedSegmentSum,
//   backendName: "wasm",
//   kernelFunc: unsortedSegmentSum as unknown as tf.KernelFunc,
// });
