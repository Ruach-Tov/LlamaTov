# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
#!/usr/bin/env python3
"""qwen_bpe.py — minimal GPT-2-style byte-level BPE tokenizer from the GGUF's embedded vocab+merges.
Lets US control tokenization (no Ollama chat template), so the divergence test compares clean raw
continuations. GPT-2 BPE: bytes -> unicode chars -> greedy merge by merge-rank."""
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import llamatov_run as R

def bytes_to_unicode():
    # standard GPT-2 byte<->unicode map
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256+n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

class QwenBPE:
    def __init__(self, blob):
        md, ts, do = R.parse_gguf(blob)
        self.tokens = md["tokenizer.ggml.tokens"]
        merges = md["tokenizer.ggml.merges"]
        self.tok2id = {t: i for i, t in enumerate(self.tokens)}
        self.merge_rank = {tuple(m.split(" ")): i for i, m in enumerate(merges)}
        self.b2u = bytes_to_unicode()

    def _bpe(self, word_chars):
        word = list(word_chars)
        while len(word) > 1:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            ranked = [(self.merge_rank.get(p, 1<<30), i) for i, p in enumerate(pairs)]
            rank, idx = min(ranked)
            if rank == (1<<30): break
            word = word[:idx] + [word[idx]+word[idx+1]] + word[idx+2:]
        return word

    def encode(self, text):
        # byte-level: map each UTF-8 byte to its unicode char, then BPE per whitespace-split chunk
        # Qwen/GPT2 uses a regex pre-tokenizer; approximate by splitting on spaces keeping the leading space marker.
        ids = []
        # GPT-2 prepends nothing; spaces become 'Ġ'. Emulate by byte-encoding the whole string.
        unicode_str = "".join(self.b2u[b] for b in text.encode("utf-8"))
        # split into 'words' at the space-marker boundary (Ġ = \u0120)
        chunks = []
        cur = ""
        for ch in unicode_str:
            if ch == "\u0120" and cur:
                chunks.append(cur); cur = ch
            else:
                cur += ch
        if cur: chunks.append(cur)
        for chunk in chunks:
            for piece in self._bpe(chunk):
                if piece in self.tok2id: ids.append(self.tok2id[piece])
                else:
                    for c in piece:
                        if c in self.tok2id: ids.append(self.tok2id[c])
        return ids

    def decode(self, ids):
        u2b = {v: k for k, v in self.b2u.items()}
        s = "".join(self.tokens[i] for i in ids)
        return bytes(u2b.get(c, ord(c)) for c in s).decode("utf-8", errors="replace")

if __name__ == "__main__":
    B = "models/qwen2.5-0.5b.gguf"
    t = QwenBPE(B)
    for s in ["The capital of France is", "Once upon a time"]:
        ids = t.encode(s)
        print(f"{s!r} -> {ids}  -> roundtrip {t.decode(ids)!r}")
