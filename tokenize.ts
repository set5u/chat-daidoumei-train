import TinySegmenter from "./segmenter";
import num2char from "./num2char.json";
import char2num from "./char2num.json";
import tokens from "./tokens.json";
import fs from "fs";
const segmenter = new TinySegmenter();
const words: { [key: string]: number } = {};
const lines: string[][] = [];
const flatten = [];
for (const nums of tokens) {
  if (nums[0] === 4 || nums.includes(char2num["["])) {
    continue;
  }
  nums.splice(0, 6);
  const strs = segmenter.segment(nums.map((v) => num2char[v]).join(""));
  flatten.push(1, ...[...nums]);
  const result = [strs[0]];
  for (const seg of strs.slice(1)) {
    const last = result.at(-1) || "";
    if (/[\ud800-\udbff]$/.test(last) && /^[\udc00-\udfff]/.test(seg)) {
      result.splice(-1, 1, last + seg);
    } else {
      result.push(seg);
    }
  }
  result[0] && lines.push(result);
  for (const str of result) {
    words[str] !== undefined ? words[str]++ : (words[str] = 1);
  }
}
for (const word in words) {
  if (words[word] < 8) {
    delete words[word];
  }
}
fs.writeFile("./words.json", JSON.stringify(words), (e) => {
  if (e) {
    throw e;
  }
});
fs.writeFile("./flatten.json", JSON.stringify(flatten), (e) => {
  if (e) {
    throw e;
  }
});
const num2word = ["<pad>", "<bos>", "<suk>", "<euk>"];
for (const word in words) {
  num2word.push(word);
}
fs.writeFile("./num2word.json", JSON.stringify(num2word), (e) => {
  if (e) {
    throw e;
  }
});
const word2num = {};
for (let i = 0; i < num2word.length; i++) {
  word2num[num2word[i]] = i;
}
fs.writeFile("./word2num.json", JSON.stringify(word2num), (e) => {
  if (e) {
    throw e;
  }
});
const wordTokens: number[] = [];
for (const line of lines) {
  wordTokens.push(word2num["<bos>"]);
  for (let word of line) {
    const wordI = word2num[word];
    if (wordI) {
      wordTokens.push(wordI);
    } else {
      let end = word.length;
      while (true) {
        if (!end) {
          const zero = word2num[Array.from(word)[0]];
          if (zero) {
            wordTokens.push(zero);
          } else {
            wordTokens.push(word2num["<suk>"]);
            for (let code of word.codePointAt(0).toString(16).toUpperCase()) {
              if (!word2num[code]) {
                throw "invalid code: " + code;
              }
              wordTokens.push(word2num[code]);
            }
            wordTokens.push(word2num["<euk>"]);
          }
          word = Array.from(word).slice(1).join("");
          end = word.length;
          if (word.length) {
            continue;
          } else {
            break;
          }
        }
        const splitted = word.substring(0, end);
        const wordI = word2num[splitted];
        if (wordI) {
          wordTokens.push(wordI);
          word = word.substring(end);
          end = word.length;
          if (!end) {
            break;
          }
        } else {
          end--;
        }
      }
    }
  }
}
fs.writeFile("./wordTokens.json", JSON.stringify(wordTokens), (e) => {
  if (e) {
    throw e;
  }
});
