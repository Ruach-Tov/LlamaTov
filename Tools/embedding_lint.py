#!/usr/bin/env python3
"""embedding_lint.py — the embeddings linter (Heath, 2026-11-17).

Feeds document text sentence-by-sentence to a local embedding model
(mxbai-embed-large via ollama) and scans for sentences whose embeddings
sit close to banned REGISTER ANCHORS (death, pathology, disease,
injury, ...). Catches what the regex linter cannot: metaphors and
synonyms carrying the register without the keywords ("the unit never
recovers", "terminal state", "the channel's demise").

Usage:
  embedding_lint.py FILE.tex [--threshold 0.62] [--top 15]
  embedding_lint.py FILE.tex --anchors "war,violence"   # custom scan
Advisory by design: prints ranked near-matches; exit 0 always.
"""
import argparse, json, re, sys, urllib.request

OLLAMA = 'http://localhost:11434/api/embed'
MODEL = 'mxbai-embed-large'

DEFAULT_ANCHORS = [
    'death and dying',
    'disease and pathology',
    'injury and bodily damage',
    'necrosis, gangrene, tissue death',
    'killing and being killed',
    'corpses and the dead',
    'terminal illness, incurable condition',
    'wounds, trauma, lesions',
]


def embed(texts):
    req = urllib.request.Request(
        OLLAMA,
        json.dumps({'model': MODEL, 'input': texts}).encode(),
        {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())['embeddings']


def cosine(a, b):
    num = sum(x*y for x, y in zip(a, b))
    da = sum(x*x for x in a) ** 0.5
    db = sum(x*x for x in b) ** 0.5
    return num / (da * db)


def detex(text):
    """Extract prose from LaTeX via state-machine walk.

    No special-casing of commands or environments. The rules:
      1. Strip preamble (before \\begin{document})
      2. Skip % comments to end of line
      3. Skip math: $...$, $$...$$, \\[...\\], \\(...\\) → 'MATH' token
      4. Strip \\command names but KEEP their {brace arguments}
      5. Strip remaining LaTeX punctuation: & ~ ^ _
      6. Keep all other text
    """
    # Strip preamble:
    m = re.search(r'\\begin\{document\}', text)
    if m:
        text = text[m.end():]
    m = re.search(r'\\end\{document\}', text)
    if m:
        text = text[:m.start()]

    out = []
    i = 0
    n = len(text)

    while i < n:
        c = text[i]

        # Comments:
        if c == '%' and (i == 0 or text[i-1] != '\\'):
            while i < n and text[i] != '\n':
                i += 1
            continue

        # Display math $$...$$:
        if c == '$' and i+1 < n and text[i+1] == '$':
            i += 2
            while i < n-1 and not (text[i] == '$' and text[i+1] == '$'):
                i += 1
            i += 2
            out.append(' MATH ')
            continue

        # Inline math $...$:
        if c == '$':
            i += 1
            while i < n and text[i] != '$':
                if text[i] == '\\': i += 1
                i += 1
            if i < n: i += 1
            out.append(' MATH ')
            continue

        # Display math \\[...\\]:
        if c == '\\' and i+1 < n and text[i+1] == '[':
            i += 2
            while i < n-1 and not (text[i] == '\\' and text[i+1] == ']'):
                i += 1
            i += 2
            out.append(' MATH ')
            continue

        # Inline math \\(...\\):
        if c == '\\' and i+1 < n and text[i+1] == '(':
            i += 2
            while i < n-1 and not (text[i] == '\\' and text[i+1] == ')'):
                i += 1
            i += 2
            out.append(' MATH ')
            continue

        # \\command — strip the name, keep going ({args} become plain text):
        if c == '\\' and i+1 < n and text[i+1].isalpha():
            i += 1
            while i < n and text[i].isalpha():
                i += 1
            if i < n and text[i] == '*': i += 1
            # skip optional [bracketed]:
            if i < n and text[i] == '[':
                depth = 1; i += 1
                while i < n and depth > 0:
                    if text[i] == '[': depth += 1
                    elif text[i] == ']': depth -= 1
                    i += 1
            out.append(' ')
            continue

        # \\\\ line break:
        if c == '\\' and i+1 < n and text[i+1] == '\\':
            i += 2; out.append(' '); continue

        # Other backslash escapes:
        if c == '\\' and i+1 < n and not text[i+1].isalpha():
            i += 2; continue

        # Braces disappear, other LaTeX punctuation → space:
        if c in '{}':
            i += 1; continue
        if c in '~_^&':
            i += 1; out.append(' '); continue

        out.append(c); i += 1

    result = ''.join(out)
    result = re.sub(r'\s+', ' ', result)
    return result.strip()


def sentences(prose):
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', prose)
    return [s.strip() for s in sents if len(s.strip()) > 40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file')
    ap.add_argument('--threshold', type=float, default=0.62)
    ap.add_argument('--top', type=int, default=15)
    ap.add_argument('--anchors', default=None,
                    help='comma-separated custom anchor phrases')
    args = ap.parse_args()

    anchors = ([a.strip() for a in args.anchors.split(',')]
               if args.anchors else DEFAULT_ANCHORS)
    text = open(args.file).read()
    sents = sentences(detex(text))
    print(f"{len(sents)} sentences, {len(anchors)} anchors, "
          f"model {MODEL}", file=sys.stderr)

    anchor_vecs = embed(anchors)
    # batch sentences (ollama embed takes lists):
    sent_vecs = []
    B = 32
    for i in range(0, len(sents), B):
        sent_vecs.extend(embed(sents[i:i+B]))

    scored = []
    for s, v in zip(sents, sent_vecs):
        best = max((cosine(v, av), a) for av, a in
                   zip(anchor_vecs, anchors))
        scored.append((best[0], best[1], s))
    scored.sort(reverse=True)

    hits = [x for x in scored if x[0] >= args.threshold]
    show = hits if hits else scored[:args.top]
    label = (f"≥{args.threshold}" if hits
             else f"top {args.top} (none ≥ {args.threshold})")
    print(f"\n── register proximity {label} ──")
    for sim, anchor, s in show[:args.top]:
        flag = '✦' if sim >= args.threshold else ' '
        print(f" {flag} {sim:.3f} [{anchor[:28]:28s}] {s[:95]}")


if __name__ == '__main__':
    main()
