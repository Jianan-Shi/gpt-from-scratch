import re as stdlib_re

import pytest

from bpe import (BasicTokenizer, RegexTokenizer, compression_ratio, get_stats,
                 merge)
from nnzh.shakespeare import load_text

TEXT = load_text()[:50_000]
UNICODE = "First Citizen: 你好 🙂 café naïve 12345! <tab>\there"

BASIC = BasicTokenizer().train(TEXT, 400)
REGEX = RegexTokenizer().train(TEXT, 400)


def test_get_stats_counts_adjacent_pairs():
    assert get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_get_stats_accumulates_across_chunks():
    """RegexTokenizer 靠这个跨块累加统计，同时不产生跨块的相邻对。"""
    counts = get_stats([1, 2])
    get_stats([1, 2, 5], counts)

    assert counts[(1, 2)] == 2
    assert (2, 1) not in counts, "两块之间不相邻"


def test_merge_replaces_every_occurrence():
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    assert merge([1, 1, 1], (1, 1), 9) == [9, 1], "重叠时从左往右，不能重复消耗"
    assert merge([1, 2], (3, 4), 9) == [1, 2], "没出现就原样返回"


def test_vocab_grows_by_one_per_merge():
    assert len(BASIC.merges) == 400 - 256
    assert len(BASIC.vocab) == 400
    assert all(idx in BASIC.vocab for idx in BASIC.merges.values())


def test_roundtrip_on_arbitrary_unicode():
    """字节级起步，所以没有 <unk>——任何文本都能编码回原样。"""
    for tok in (BASIC, REGEX):
        assert tok.decode(tok.encode(UNICODE)) == UNICODE
        assert tok.decode(tok.encode("")) == ""
        assert tok.decode(tok.encode(TEXT[:2000])) == TEXT[:2000]


def test_every_byte_is_representable():
    for tok in (BASIC, REGEX):
        assert tok.decode(list(range(256))) == bytes(range(256)).decode("utf-8", errors="replace")


def test_regex_never_merges_across_category_boundaries():
    """本章正则切分存在的理由：不让 "dog." 的 g 和 . 合成一个 token。

    不切分的话，同一个词会因为后面跟的标点不同而落到完全不同的 token 上，
    模型得把同一个词学好几遍。
    """
    crossing = stdlib_re.compile(rb"[A-Za-z][ .,!?;:\n]|[ .,!?;:\n][A-Za-z]")

    def straddling(tok):
        return [v for k, v in tok.vocab.items()
                if k >= 256 and crossing.search(v) and not v.startswith(b" ")]

    assert straddling(REGEX) == []
    assert len(straddling(BASIC)) > 10, "对照：不切分时这类 token 大量出现"


def test_encode_follows_training_merge_order():
    """encode 必须按训练时的顺序合并，否则和训练出来的词表不自洽。"""
    ids = REGEX.encode(TEXT[:5000])
    assert REGEX.decode(ids) == TEXT[:5000]

    tok = RegexTokenizer().train("aaabdaaabac", 259)
    # 先合并 aa(256)，再 256+a(257)，再 257+b(258)
    assert tok.encode("aaabdaaabac") == [258, 100, 258, 97, 99]


def test_compression_improves_with_vocab_size():
    ratios = [compression_ratio(RegexTokenizer().train(TEXT, v), TEXT)
              for v in (300, 500, 1000)]

    assert all(a < b for a, b in zip(ratios, ratios[1:])), ratios
    assert ratios[0] > 1.0


def test_special_tokens_bypass_the_merges():
    tok = RegexTokenizer().train(TEXT, 300).register_special_tokens({"<|endoftext|>": 100_257})
    text = "hello<|endoftext|>world"

    ids = tok.encode(text, allowed_special="all")

    assert 100_257 in ids
    assert tok.decode(ids) == text
    assert tok.encode("hello", allowed_special="all") == tok.encode_ordinary("hello")


def test_special_tokens_in_user_text_raise_by_default():
    """用户输入里混进 <|endoftext|> 是一类真实的注入风险，默认不能静默接受。"""
    tok = RegexTokenizer().train(TEXT, 300).register_special_tokens({"<|endoftext|>": 100_257})

    with pytest.raises(AssertionError):
        tok.encode("hello <|endoftext|> world")

    ids = tok.encode("hello <|endoftext|> world", allowed_special="none")
    assert 100_257 not in ids, "当普通文本处理时，它只是一串普通字符"


def test_decode_rejects_unknown_ids():
    with pytest.raises(ValueError):
        REGEX.decode([999_999])
