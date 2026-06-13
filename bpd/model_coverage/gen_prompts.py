#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""gen_prompts.py — generate ~100 example prompts ranging short->long, varied categories.
Writes prompts.jsonl: {"id", "prompt", "category", "len_class"}.
Categories exercise different token distributions to stress the bit-exact decode."""
import json, os

prompts = []
def add(cat, length, text):
    prompts.append({"id": f"{cat}_{length}_{len([p for p in prompts if p['category']==cat]):02d}",
                    "prompt": text, "category": cat, "len_class": length})

# ── SHORT (factual, terse) ──
short_factual = [
    "What is the capital of France?",
    "Name three primary colors.",
    "What is 17 times 23?",
    "Define photosynthesis in one sentence.",
    "What year did World War II end?",
    "List the planets in order from the sun.",
    "What is the chemical symbol for gold?",
    "Translate 'hello' to Spanish.",
    "What is the speed of light?",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water in Celsius?",
    "Name the largest ocean on Earth.",
]
for t in short_factual: add("factual", "short", t)

# ── SHORT (code) ──
short_code = [
    "Write a Python function to reverse a string.",
    "Write a one-line bash command to count files in a directory.",
    "What does the SQL JOIN keyword do?",
    "Write a regex to match an email address.",
    "Explain what a hash map is.",
    "Write a function to check if a number is prime.",
]
for t in short_code: add("code", "short", t)

# ── MEDIUM (explanatory) ──
medium = [
    "Explain how a neural network learns through backpropagation.",
    "Describe the differences between TCP and UDP protocols.",
    "Write a short story about a robot who discovers music.",
    "Explain the theory of relativity to a high school student.",
    "Compare and contrast functional and object-oriented programming.",
    "Describe the water cycle and its importance to life on Earth.",
    "Explain how vaccines work to build immunity.",
    "Write a haiku about autumn, then explain its structure.",
    "Describe the process of cellular respiration step by step.",
    "Explain the concept of supply and demand in economics.",
    "Write a recipe for a simple vegetable soup.",
    "Describe how a computer's CPU executes instructions.",
    "Explain the causes of the French Revolution.",
    "Describe the lifecycle of a star from birth to death.",
    "Explain how blockchain technology achieves consensus.",
    "Write instructions for changing a flat tire.",
    "Describe the major branches of philosophy.",
    "Explain how the human immune system fights infection.",
    "Compare the architectures of CPUs and GPUs.",
    "Describe the greenhouse effect and climate change.",
]
for t in medium: add("explain", "medium", t)

# ── MEDIUM (reasoning) ──
reasoning = [
    "If a train travels 60 mph for 2.5 hours, how far does it go? Show your reasoning.",
    "A farmer has chickens and cows totaling 30 animals and 74 legs. How many of each?",
    "Explain the Monty Hall problem and why switching doors helps.",
    "If all roses are flowers and some flowers fade quickly, can we conclude some roses fade quickly?",
    "Solve: a bat and ball cost $1.10 together, the bat costs $1 more than the ball. How much is the ball?",
    "Walk through the steps to sort the list [5,2,8,1,9] using merge sort.",
    "If today is Wednesday, what day will it be in 100 days? Explain.",
    "Explain why the sky appears blue during the day.",
]
for t in reasoning: add("reasoning", "medium", t)

# ── LONG (multi-part, extended generation) ──
long_prompts = [
    "Write a detailed essay on the history and impact of the printing press on European society, covering its invention, spread, and effects on literacy, religion, and science.",
    "Explain in depth how the TCP/IP networking stack works, from the physical layer up to the application layer, describing each layer's responsibilities and key protocols.",
    "Write a comprehensive tutorial on building a REST API in Python, including routing, request handling, database integration, authentication, and error handling.",
    "Describe the complete process of how a modern compiler transforms source code into executable machine code, covering lexing, parsing, semantic analysis, optimization, and code generation.",
    "Write a long-form analysis of the causes, key events, and lasting consequences of the Industrial Revolution in Britain.",
    "Explain the fundamentals of quantum mechanics, including wave-particle duality, the uncertainty principle, superposition, and entanglement, with intuitive examples.",
    "Write a detailed guide to training a machine learning model from scratch: data collection, preprocessing, feature engineering, model selection, training, evaluation, and deployment.",
    "Compose a thorough explanation of how the human brain processes visual information, from the retina through the visual cortex to higher-order interpretation.",
    "Write an extended story about an explorer who finds an ancient library containing all the world's lost knowledge, and the choices they must make.",
    "Explain the complete lifecycle of a software project using agile methodology, covering planning, sprints, standups, reviews, retrospectives, and continuous deployment.",
    "Describe in detail how modern GPUs achieve parallelism, covering SIMD execution, warps/wavefronts, memory hierarchy, and why they excel at matrix operations.",
    "Write a comprehensive overview of the major programming paradigms — imperative, functional, object-oriented, logic, and concurrent — with examples and trade-offs.",
]
for t in long_prompts: add("longform", "long", t)

# ── EDGE (unusual tokens, repetition, numbers, symbols) ──
edge = [
    "Count from 1 to 50.",
    "Repeat the word 'echo' exactly 20 times.",
    "List the first 30 prime numbers.",
    "Write out the alphabet, then write it backwards.",
    "Generate a sequence: 2, 4, 8, 16, ... continue for 15 terms.",
    "Write the numbers 1 through 20 in both digits and English words.",
    "Output this JSON structure for a user with name, age, email, and address fields.",
    "Write a multiplication table from 1 to 12.",
]
for t in edge: add("edge", "varied", t)

# ── MULTILINGUAL ──
multi = [
    "Write a paragraph in French about your favorite season.",
    "Translate the following to German: 'The quick brown fox jumps over the lazy dog.'",
    "Write a short poem in Spanish about the ocean.",
    "Explain the concept of 'umami' and its origins in Japanese cuisine.",
]
for t in multi: add("multilingual", "medium", t)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.jsonl")
with open(out, "w") as f:
    for p in prompts:
        f.write(json.dumps(p) + "\n")
print(f"wrote {len(prompts)} prompts -> {out}")
# summary by category and length
from collections import Counter
print("by category:", dict(Counter(p["category"] for p in prompts)))
print("by length:  ", dict(Counter(p["len_class"] for p in prompts)))
