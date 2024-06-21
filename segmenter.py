import json

with open("./num2char.json", "r", -1, "utf-8") as f:
    num2char = json.loads("".join(f.readlines()))
with open("./tokens.json", "r", -1, "utf-8") as f:
    tokens = json.loads("".join(f.readlines()))
import MeCab
import ipadic

mecab = MeCab.Tagger(ipadic.MECAB_ARGS + " -Owakati")
out = ""
for nums in tokens:
    if len(nums) != 0 and nums[0] == 4:
        continue
    nums = nums[6:]
    text = "".join(num2char[num] for num in nums)
    if "[" in text:
        continue
    out += mecab.parse(text)
with open("./corpus.txt", "w", -1, "utf-8") as f:
    f.write(out)
