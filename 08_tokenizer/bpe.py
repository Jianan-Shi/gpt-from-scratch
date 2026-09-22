"""Byte-pair encoding from scratch (Let's build the GPT Tokenizer).

分词器是**独立于模型**的一段预处理：它有自己的训练集、自己的训练过程，和神经网络
之间只通过整数序列打交道。09 章用的 tiktoken gpt2（50257）就是这样训练出来的。

算法本身很短：
1. 文本按 UTF-8 编码成字节，于是初始词表就是 0..255，任何文本都能表示，没有 <unk>；
2. 统计相邻字节对的出现次数，把最高频的那一对合并成一个新 id（256、257、…）；
3. 重复 vocab_size - 256 次。

两个版本：
- BasicTokenizer   直接在整段文本上合并。
- RegexTokenizer   先用正则把文本切成块（单词、数字、标点、空白各成一块），
                   只在块内部合并。GPT-2 就是这么做的，为的是不让 "dog." 里的
                   "g" 和 "." 合成一个 token，也不让长数字被切得毫无规律。

`experiments_bpe.py` 里量了这件事带来的差别，以及 tokenization 是怎么造成
"数不清字母、算术差、行尾空格敏感、中文更费 token"这些现象的。
"""
import regex as re

# GPT-4 的切分模式（cl100k_base）。GPT-2 的版本不处理大小写混排和三位以上数字。
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def get_stats(ids, counts=None):
    """统计相邻 id 对出现的次数。counts 传进来可以跨多个块累加。"""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """把 ids 里所有出现的 pair 替换成 idx。"""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BasicTokenizer:
    """最朴素的版本：整段文本上反复合并最高频的相邻对。"""

    def __init__(self):
        self.merges = {}   # (int, int) -> int，插入顺序就是合并顺序
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))

        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break # 已经合并到只剩一个 token
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1}: {pair} -> {idx} ({self.vocab[idx]}) 出现 {stats[pair]} 次")
        return self

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # 按训练时的顺序合并：早合并的优先，否则 encode 和 train 不一致
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        # errors="replace"：单个 token 可能是半个字符，逐 token 解码必然会遇到
        return text_bytes.decode("utf-8", errors="replace")


class RegexTokenizer(BasicTokenizer):
    """先按正则切块，只在块内合并。special token 走单独的通道。"""

    def __init__(self, pattern=GPT4_SPLIT_PATTERN):
        super().__init__()
        self.compiled_pattern = re.compile(pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        chunks = [list(ch.encode("utf-8")) for ch in re.findall(self.compiled_pattern, text)]

        for i in range(vocab_size - 256):
            stats = {}
            for chunk in chunks:
                get_stats(chunk, stats) # 跨块的相邻对不统计，也就永远不会被合并
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            chunks = [merge(chunk, pair, idx) for chunk in chunks]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1}: {pair} -> {idx} ({self.vocab[idx]}) 出现 {stats[pair]} 次")
        return self

    def register_special_tokens(self, special_tokens):
        """例如 {"<|endoftext|>": 100257}。它们不参与合并，直接映射到固定 id。"""
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        return self

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text):
        """不处理 special token 的版本。"""
        ids = []
        for chunk in re.findall(self.compiled_pattern, text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """allowed_special: "all" 认特殊 token，"none" 当普通文本，
        "none_raise"（默认）在文本里出现特殊 token 时直接报错——
        用户输入里混进 <|endoftext|> 是一类真实的注入风险。
        """
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special in ("none", "none_raise"):
            special = {}
            if allowed_special == "none_raise":
                assert all(tok not in text for tok in self.special_tokens)
        else:
            raise ValueError(f"allowed_special={allowed_special} 不认识")

        if not special:
            return self.encode_ordinary(text)

        pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        ids = []
        for part in re.split(pattern, text):
            if part in special:
                ids.append(special[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids):
        parts = []
        for idx in ids:
            if idx in self.vocab:
                parts.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                parts.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"不认识的 token id: {idx}")
        return b"".join(parts).decode("utf-8", errors="replace")


def compression_ratio(tokenizer, text):
    """每个 token 平均代表多少字节。越大越省，1.0 等于没压缩。"""
    encode = getattr(tokenizer, "encode_ordinary", tokenizer.encode)
    return len(text.encode("utf-8")) / len(encode(text))
