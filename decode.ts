import num2word from "./num2word.json";
import word2num from "./word2num.json";
import wordTokens from "./wordTokens.json";
let buf = "";
for (const w of wordTokens) {
  if (buf) {
    if (w == word2num["<euk>"]) {
      const num = parseInt(buf, 16);
      process.stdout.write("{" + String.fromCharCode(num) + "}");
      buf = "";
    } else {
      buf += num2word[w];
    }
  } else {
    if (w == word2num["<suk>"]) {
      buf += "0";
    } else {
      w == word2num["<bos>"]
        ? console.log(num2word[w])
        : process.stdout.write(num2word[w]);
    }
  }
}
