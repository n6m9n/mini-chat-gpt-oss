"""
Tokenizer for mini-gpt-oss : the exact o200k_harmony BPE used by gpt-oss
(reproduced from VizuaraAILabs/nano-gpt-oss/architecture/tokenizer.py).

It extends tiktoken's o200k_base with the harmony special tokens, giving a
vocab of 201,088. <|endoftext|> (id 199999) is used to separate documents.
"""

import functools

import tiktoken

EOT = 199999          # <|endoftext|>
VOCAB_SIZE = 201088


@functools.lru_cache(maxsize=1)
def get_tokenizer() -> "tiktoken.Encoding":
    o200k_base = tiktoken.get_encoding("o200k_base")
    return tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=o200k_base._pat_str,
        mergeable_ranks=o200k_base._mergeable_ranks,
        special_tokens={
            **o200k_base._special_tokens,
            "<|startoftext|>": 199998,
            "<|endoftext|>": 199999,
            "<|reserved_200000|>": 200000,
            "<|reserved_200001|>": 200001,
            "<|return|>": 200002,
            "<|constrain|>": 200003,
            "<|reserved_200004|>": 200004,
            "<|channel|>": 200005,
            "<|start|>": 200006,
            "<|end|>": 200007,
            "<|message|>": 200008,
            "<|reserved_200009|>": 200009,
            "<|reserved_200010|>": 200010,
            "<|reserved_200011|>": 200011,
            "<|call|>": 200012,
        }
        | {f"<|reserved_{i}|>": i for i in range(200013, 201088)},
    )


def encode(text: str) -> list[int]:
    return get_tokenizer().encode(text)


def decode(tokens: list[int]) -> str:
    return get_tokenizer().decode(tokens)


if __name__ == "__main__":
    enc = get_tokenizer()
    print("vocab size:", enc.n_vocab)
    ids = enc.encode("Once upon a time, there was a little robot.")
    print("ids:", ids)
    print("roundtrip:", enc.decode(ids))
