import json from "@rollup/plugin-json";
import commonjs from "@rollup/plugin-commonjs";
import nodeResolve from "@rollup/plugin-node-resolve";
import terser from "@rollup/plugin-terser";
import babel from "@rollup/plugin-babel";

export default {
  input: ["index.mjs", "predict.mjs"],
  output: {
    dir: "dist",
    format: "cjs",
  },
  plugins: [
    json(),
    nodeResolve({ transformMixedEsModules: true }),
    commonjs({ ignoreDynamicRequires: true }),
    babel({
      presets: [
        [
          "@babel/preset-env",
          {
            include: ["@babel/plugin-transform-classes"],
            useBuiltIns: false,
          },
        ],
      ],
      plugins: [
        "@babel/plugin-syntax-import-attributes",
        "@babel/plugin-transform-runtime",
      ],
      babelHelpers: "runtime",
    }),
    // terser()
  ],
};
