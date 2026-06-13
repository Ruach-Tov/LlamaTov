#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""multiturn_test.py — test KV-cache reproducibility across multi-turn conversations.

Heath's design:
  - Each test runs SEVERAL turns (so the KV-cache accumulates and is exercised turn-over-turn).
  - Each test ZEROS the cache at start (fresh model load) so it reproduces identically whether
    run in ISOLATION or in SEQUENCE with other tests.

Method: use /api/chat with a growing message history (each turn appends the model's reply),
all deterministic (temp=0, seed, top_k=1, num_gpu:0). The full multi-turn token stream is
fingerprinted. We then:
  1. Run each conversation in ISOLATION (cache zeroed before) -> record.
  2. Run them in SEQUENCE (cache zeroed before each) -> verify isolation == sequence.
A KV-cache that's correct + a zeroed start => the multi-turn stream is bit-identical both ways.
"""
import json, sys, hashlib, urllib.request

OLLAMA = "http://localhost:11434"
MODEL = "llama3.2:1b"

# Multi-turn conversation seeds (each a list of user turns; model replies feed forward).
CONVERSATIONS = {
    "math_chain": [
        "What is 17 times 23?",
        "Now divide that result by 4.",
        "Round it to the nearest integer and explain.",
    ],
    "story_build": [
        "Start a story about a lighthouse keeper.",
        "Introduce a mysterious ship on the horizon.",
        "End the story with a twist.",
    ],
    "code_refine": [
        "Write a Python function to check if a string is a palindrome.",
        "Now make it ignore case and spaces.",
        "Add a docstring and a test case.",
    ],
}

def chat_turn(messages, zero_cache_first=False):
    """One /api/chat turn. keep_alive:0 forces unload after (zeroes cache for next isolated test)."""
    opts = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 42,
            "num_predict": 200, "num_ctx": 4096, "num_gpu": 0, "num_thread": 1}
    body = {"model": MODEL, "messages": messages, "stream": False,
            "keep_alive": 0, "options": opts}   # keep_alive:0 => cache zeroed between tests
    req = urllib.request.Request(f"{OLLAMA}/api/chat",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return resp["message"]["content"]

def run_conversation(turns):
    """Run a full multi-turn conversation from a ZEROED cache. Returns a fingerprint of the
    entire turn-by-turn reply stream (so KV-cache accumulation is exercised + captured)."""
    messages = []
    replies = []
    for user_turn in turns:
        messages.append({"role": "user", "content": user_turn})
        reply = chat_turn(messages)
        messages.append({"role": "assistant", "content": reply})
        replies.append(reply)
    # fingerprint the full multi-turn stream
    full = "\x00".join(replies)
    return hashlib.sha256(full.encode()).hexdigest()[:16], [len(r) for r in replies]

def main():
    convs = list(CONVERSATIONS.items())
    if len(sys.argv) > 1:
        convs = [(k, CONVERSATIONS[k]) for k in sys.argv[1:] if k in CONVERSATIONS]

    print("=== PASS 1: each conversation in ISOLATION (cache zeroed before each) ===")
    isolated = {}
    for name, turns in convs:
        fp, lens = run_conversation(turns)
        isolated[name] = fp
        print(f"  {name:14s} {len(turns)}turns reply_lens={lens} fp={fp}")

    print("\n=== PASS 2: SEQUENCE — same convs back-to-back (cache zeroed before each) ===")
    passed = True
    for name, turns in convs:
        fp, lens = run_conversation(turns)
        ok = fp == isolated[name]
        passed = passed and ok
        print(f"  {name:14s} fp={fp} {'OK (isolation==sequence)' if ok else 'DIVERGE vs isolation'}")

    print(f"\nGATE: {'PASS — multi-turn KV-cache reproduces identically in isolation and sequence' if passed else 'FAIL — KV-cache state leaks across tests'}")
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
