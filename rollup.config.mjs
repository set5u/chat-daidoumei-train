import json from "@rollup/plugin-json";
import commonjs from "@rollup/plugin-commonjs";
import nodeResolve from "@rollup/plugin-node-resolve";
import terser from "@rollup/plugin-terser";
import typescript from "@rollup/plugin-typescript";
import babel from "@rollup/plugin-babel";

export default {
  input: ["tokenize.ts", "decode.ts"],
  output: {
    dir: "dist",
    format: "cjs",
  },
  plugins: [
    json(),
    typescript(),
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
    terser(),
  ],
};
