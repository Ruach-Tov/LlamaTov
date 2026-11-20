#!/usr/bin/env python3
"""claim_strength_lint.py — epistemic claim strength scanner (Mavchin, 2026-11-18).

Companion to embedding_lint.py (register scanner). Uses the same architecture
(sentence → embed → cosine vs anchors) but scans for EPISTEMIC STRENGTH:
sentences whose meaning is close to strong causal/conclusive claims.

The goal: flag sentences that claim more than "not consistent with" before
a peer reviewer does. The tool caught our §9's "rules out screen perturbation"
in hindsight — now it catches it in advance.

THREE TIERS of anchors:

  STRONG CLAIMS (threshold ≥ 0.60):
    "proves", "establishes", "rules out", "demonstrates conclusively"
    → Sentences near these anchors are making strong causal claims
      that need exhaustive evidence.

  HEDGED CLAIMS (reference — these are GOOD):
    "is consistent with", "suggests", "is not inconsistent with"
    → For comparison: well-hedged sentences should score higher
      on these anchors than on strong-claim anchors.

  EXHAUSTIVE NEGATION (threshold ≥ 0.55):
    "eliminates all alternatives", "the only explanation"
    → Claims that something is the ONLY possibility.
      Valid only when alternatives are exhaustively enumerated.

Usage:
  claim_strength_lint.py FILE.tex [--threshold 0.55] [--top 15]
  claim_strength_lint.py FILE.tex --check strong    # only strong claims
  claim_strength_lint.py FILE.tex --check exhaustive # only exhaustive
Advisory by design: prints ranked near-matches; exit 0 always.
"""
import argparse, json, re, sys, urllib.request

OLLAMA = 'http://localhost:11434/api/embed'
MODEL = 'mxbai-embed-large'

# Strong causal/conclusive claims:
STRONG_ANCHORS = [
    'this proves that the hypothesis is correct',
    'this rules out the alternative explanation entirely',
    'this demonstrates conclusively that the effect is real',
    'this establishes beyond doubt that the cause is identified',
    'we have shown definitively that this is the mechanism',
    'this eliminates the possibility of any other explanation',
    'the evidence is conclusive and admits no alternative',
]

# Exhaustive negation (claiming to be the ONLY explanation):
EXHAUSTIVE_ANCHORS = [
    'the only possible explanation is',
    'no other mechanism can account for this observation',
    'all other alternatives have been eliminated',
    'this is necessarily the cause because nothing else fits',
]

# Hedged claims (reference — well-calibrated epistemics):
HEDGED_ANCHORS = [
    'the results are consistent with this interpretation',
    'this suggests but does not prove the hypothesis',
    'the data are not inconsistent with this model',
    'one possible explanation is',
    'further investigation is needed to confirm',
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
    return num / (da * db + 1e-10)


def detex(text):
    """Extract prose from LaTeX via state-machine walk. Same as embedding_lint.py."""
    m = re.search(r'\\begin\{document\}', text)
    if m: text = text[m.end():]
    m = re.search(r'\\end\{document\}', text)
    if m: text = text[:m.start()]
    out = []; i = 0; n = len(text)
    while i < n:
        c = text[i]
        if c == '%' and (i == 0 or text[i-1] != '\\'):
            while i < n and text[i] != '\n': i += 1
            continue
        if c == '$' and i+1 < n and text[i+1] == '$':
            i += 2
            while i < n-1 and not (text[i] == '$' and text[i+1] == '$'): i += 1
            i += 2; out.append(' MATH '); continue
        if c == '$':
            i += 1
            while i < n and text[i] != '$':
                if text[i] == '\\': i += 1
                i += 1
            if i < n: i += 1
            out.append(' MATH '); continue
        if c == '\\' and i+1 < n and text[i+1] == '[':
            i += 2
            while i < n-1 and not (text[i] == '\\' and text[i+1] == ']'): i += 1
            i += 2; out.append(' MATH '); continue
        if c == '\\' and i+1 < n and text[i+1] == '(':
            i += 2
            while i < n-1 and not (text[i] == '\\' and text[i+1] == ')'): i += 1
            i += 2; out.append(' MATH '); continue
        if c == '\\' and i+1 < n and text[i+1].isalpha():
            i += 1
            while i < n and text[i].isalpha(): i += 1
            if i < n and text[i] == '*': i += 1
            if i < n and text[i] == '[':
                depth = 1; i += 1
                while i < n and depth > 0:
                    if text[i] == '[': depth += 1
                    elif text[i] == ']': depth -= 1
                    i += 1
            out.append(' '); continue
        if c == '\\' and i+1 < n and text[i+1] == '\\':
            i += 2; out.append(' '); continue
        if c == '\\' and i+1 < n and not text[i+1].isalpha():
            i += 2; continue
        if c in '{}': i += 1; continue
        if c in '~_^&': i += 1; out.append(' '); continue
        out.append(c); i += 1
    result = re.sub(r'\s+', ' ', ''.join(out))
    return result.strip()


def sentences(prose):
    raw = re.split(r'(?<=[.!?])\s+', prose.strip())
    return [s.strip() for s in raw if len(s.strip()) > 30]


def main():
    ap = argparse.ArgumentParser(
        description='Epistemic claim strength scanner')
    ap.add_argument('file')
    ap.add_argument('--threshold', type=float, default=0.55)
    ap.add_argument('--top', type=int, default=15)
    ap.add_argument('--check', choices=['strong', 'exhaustive', 'all'],
                    default='all', help='which anchor set to scan')
    args = ap.parse_args()

    # Build anchor sets:
    anchor_groups = {}
    if args.check in ('strong', 'all'):
        anchor_groups['STRONG CLAIM'] = STRONG_ANCHORS
    if args.check in ('exhaustive', 'all'):
        anchor_groups['EXHAUSTIVE'] = EXHAUSTIVE_ANCHORS

    text = open(args.file).read()
    sents = sentences(detex(text))

    # Also embed hedged anchors for the comparison:
    all_anchors = []
    anchor_labels = []
    for group_name, anchors in anchor_groups.items():
        for a in anchors:
            all_anchors.append(a)
            anchor_labels.append(group_name)

    hedged_anchors = HEDGED_ANCHORS[:]
    hedged_start = len(all_anchors)
    all_anchors.extend(hedged_anchors)
    for a in hedged_anchors:
        anchor_labels.append('HEDGED (reference)')

    print(f"{len(sents)} sentences, "
          f"{len(all_anchors) - len(hedged_anchors)} claim anchors + "
          f"{len(hedged_anchors)} hedged reference anchors, "
          f"model {MODEL}", file=sys.stderr)

    anchor_vecs = embed(all_anchors)
    claim_vecs = anchor_vecs[:hedged_start]
    hedge_vecs = anchor_vecs[hedged_start:]

    # Batch embed sentences:
    sent_vecs = []
    B = 32
    for i in range(0, len(sents), B):
        sent_vecs.extend(embed(sents[i:i + B]))

    # Score each sentence against claim AND hedge anchors:
    scored = []
    for s, v in zip(sents, sent_vecs):
        # Best claim anchor match:
        best_claim = max(
            (cosine(v, av), anchor_labels[i], all_anchors[i])
            for i, av in enumerate(claim_vecs))
        # Best hedge anchor match:
        best_hedge = max(
            (cosine(v, hv), 'HEDGED', h)
            for hv, h in zip(hedge_vecs, hedged_anchors))

        # The RATIO is informative: claim/hedge > 1 means the sentence
        # is closer to strong claims than to hedged language.
        ratio = best_claim[0] / (best_hedge[0] + 1e-10)

        scored.append({
            'sentence': s,
            'claim_sim': best_claim[0],
            'claim_group': best_claim[1],
            'claim_anchor': best_claim[2],
            'hedge_sim': best_hedge[0],
            'ratio': ratio,
        })

    # Sort by claim strength:
    scored.sort(key=lambda x: x['claim_sim'], reverse=True)

    hits = [x for x in scored if x['claim_sim'] >= args.threshold]
    show = hits if hits else scored[:args.top]
    label = (f"≥{args.threshold}" if hits
             else f"top {args.top} (none ≥ {args.threshold})")

    print(f"\n── claim strength {label} ──")
    print(f"{'claim':>6} {'hedge':>6} {'ratio':>6}  {'type':18s} sentence")
    print("─" * 90)
    for x in show[:args.top]:
        flag = '⚠' if x['claim_sim'] >= args.threshold else ' '
        ratio_flag = '!' if x['ratio'] > 1.05 else ' '
        print(f"{flag}{x['claim_sim']:.3f} {x['hedge_sim']:.3f} "
              f"{ratio_flag}{x['ratio']:.2f}  "
              f"{x['claim_group']:18s} {x['sentence'][:60]}")

    # Summary:
    above = sum(1 for x in scored if x['claim_sim'] >= args.threshold)
    high_ratio = sum(1 for x in scored if x['ratio'] > 1.05)
    print(f"\n── summary ──")
    print(f"  Sentences above threshold ({args.threshold}): {above}")
    print(f"  Sentences with claim/hedge ratio > 1.05: {high_ratio}")
    if high_ratio > 0:
        print(f"  → These sentences are closer to strong claims than "
              f"to hedged language.")
        print(f"  → Consider whether the evidence supports the strength.")


# ═══════════════════════════════════════════════════════════════
# EXTENSION: claim ↔ control branch matching (Mavchin, 2026-11-18)
#
# For each strong claim, search the document for corresponding
# experimental control/evidence sentences. A claim without a
# nearby control is an UNSUPPORTED CLAIM — the most dangerous
# kind, because the reader assumes the evidence exists.
# ═══════════════════════════════════════════════════════════════

CONTROL_ANCHORS = [
    'we tested this by comparing against a control condition',
    'the null hypothesis was rejected with p-value',
    'paired comparison with n seeds shows statistical significance',
    'cross-validation on held-out data confirms the result',
    'the ablation study removes the intervention and measures the difference',
    'we verified this empirically by measuring',
    'the control arm without the intervention shows',
    'a synthetic null test with matched noise confirms',
    'we ran a parameter sweep to find the sensitivity',
    'the measurement was repeated across independent trials',
]


def find_controls(args_file, threshold=0.55):
    """Find claims and match them to nearby control/evidence sentences."""
    text = open(args_file).read()
    sents = sentences(detex(text))
    
    # Embed everything:
    all_anchors = STRONG_ANCHORS + EXHAUSTIVE_ANCHORS + CONTROL_ANCHORS
    anchor_vecs = embed(all_anchors)
    
    claim_vecs = anchor_vecs[:len(STRONG_ANCHORS) + len(EXHAUSTIVE_ANCHORS)]
    control_vecs = anchor_vecs[len(STRONG_ANCHORS) + len(EXHAUSTIVE_ANCHORS):]
    
    sent_vecs = []
    B = 32
    for i in range(0, len(sents), B):
        sent_vecs.extend(embed(sents[i:i + B]))
    
    # Score each sentence as claim AND as control:
    claim_scores = []
    control_scores = []
    
    for v in sent_vecs:
        best_claim = max(cosine(v, av) for av in claim_vecs)
        best_control = max(cosine(v, cv) for cv in control_vecs)
        claim_scores.append(best_claim)
        control_scores.append(best_control)
    
    # For each claim sentence, find the nearest control sentence
    # within a window (±15 sentences ≈ ±2 paragraphs):
    WINDOW = 15
    
    claims = []
    for i, (s, cs) in enumerate(zip(sents, claim_scores)):
        if cs < threshold:
            continue
        
        # Search window for best control:
        best_control_sim = 0
        best_control_idx = -1
        for j in range(max(0, i - WINDOW), min(len(sents), i + WINDOW + 1)):
            if j == i:
                continue
            if control_scores[j] > best_control_sim:
                best_control_sim = control_scores[j]
                best_control_idx = j
        
        # Also compute cross-similarity: how similar is the claim 
        # sentence to the control anchors? (self-supporting claims)
        self_control = max(cosine(sent_vecs[i], cv) for cv in control_vecs)
        
        claims.append({
            'idx': i,
            'sentence': s,
            'claim_sim': cs,
            'nearest_control_sim': best_control_sim,
            'nearest_control_idx': best_control_idx,
            'nearest_control': sents[best_control_idx] if best_control_idx >= 0 else '',
            'self_control': self_control,
            'distance': abs(best_control_idx - i) if best_control_idx >= 0 else 999,
            'supported': best_control_sim >= 0.45 or self_control >= 0.50,
        })
    
    return claims, sents


def print_control_report(claims, sents):
    """Print the claim ↔ control matching report."""
    unsupported = [c for c in claims if not c['supported']]
    supported = [c for c in claims if c['supported']]
    
    print(f"\n══ claim ↔ control matching ══")
    print(f"  {len(claims)} claims above threshold")
    print(f"  {len(supported)} with nearby experimental control")
    print(f"  {len(unsupported)} WITHOUT nearby control ← review these")
    
    if unsupported:
        print(f"\n── unsupported claims (no nearby control/evidence) ──")
        print(f"{'claim':>6} {'ctrl':>6} {'dist':>5}  sentence")
        print("─" * 85)
        for c in sorted(unsupported, key=lambda x: -x['claim_sim']):
            print(f" {c['claim_sim']:.3f} {c['nearest_control_sim']:.3f} "
                  f"{c['distance']:>4}  {c['sentence'][:65]}")
    
    if supported:
        print(f"\n── supported claims (with nearby control) ──")
        print(f"{'claim':>6} {'ctrl':>6} {'dist':>5}  sentence")
        print("─" * 85)
        for c in sorted(supported, key=lambda x: -x['claim_sim'])[:10]:
            print(f" {c['claim_sim']:.3f} {c['nearest_control_sim']:.3f} "
                  f"{c['distance']:>4}  {c['sentence'][:65]}")
            print(f"        └─ {c['nearest_control'][:70]}")


# ═══════════════════════════════════════════════════════════════
# EXTENSION: persuasion/rhetoric scanner (Mavchin, 2026-11-18)
#
# Detects sentences that ARGUE rather than REPORT. Scientific
# prose should present evidence and let it speak; persuasion
# substitutes rhetoric for measurement.
#
# Argument from absence: "no evidence against X" ≠ "evidence for X"
# Appeal to authority: "experts agree" without citing data
# Rhetorical framing: "clearly", "obviously", "undeniably"
# Emotional appeal: "remarkable", "stunning", "alarming"
# Hedge-as-argument: using hedged language to smuggle a claim
# ═══════════════════════════════════════════════════════════════

PERSUASION_ANCHORS = [
    # Argument from absence:
    'the lack of evidence against this hypothesis supports it',
    'no alternative explanation has been proposed therefore this must be correct',
    'since no one has disproven this we can accept it as true',
    # Appeal to plausibility without evidence:
    'it is intuitively obvious that this must be the case',
    'clearly this is the most reasonable interpretation',
    'any reasonable observer would conclude that',
    # Rhetorical intensifiers masking weak evidence:
    'this is a remarkable and unprecedented finding',
    'the results are striking and demand immediate attention',
    'this stunning observation overturns conventional wisdom',
    # Persuasive framing (arguing rather than reporting):
    'we urge the reader to consider the implications',
    'it would be irresponsible to ignore this evidence',
    'the importance of this finding cannot be overstated',
    # Argument from authority without data:
    'leading experts in the field agree with this interpretation',
    'the consensus view supports our conclusion',
]

NEUTRAL_REPORT_ANCHORS = [
    # What good scientific prose sounds like:
    'the measurement shows a value of X with uncertainty Y',
    'we observe an effect of this magnitude under these conditions',
    'the data are presented in the following table',
    'the difference between conditions was measured to be',
    'we report the following experimental result',
]


def scan_persuasion(args_file, threshold=0.50):
    """Scan for persuasive/rhetorical language vs neutral reporting."""
    text = open(args_file).read()
    sents = sentences(detex(text))
    
    all_anchors = PERSUASION_ANCHORS + NEUTRAL_REPORT_ANCHORS
    anchor_vecs = embed(all_anchors)
    
    persuasion_vecs = anchor_vecs[:len(PERSUASION_ANCHORS)]
    neutral_vecs = anchor_vecs[len(PERSUASION_ANCHORS):]
    
    sent_vecs = []
    B = 32
    for i in range(0, len(sents), B):
        sent_vecs.extend(embed(sents[i:i + B]))
    
    scored = []
    for s, v in zip(sents, sent_vecs):
        best_persuade = max(
            (cosine(v, pv), PERSUASION_ANCHORS[i])
            for i, pv in enumerate(persuasion_vecs))
        best_neutral = max(
            (cosine(v, nv), NEUTRAL_REPORT_ANCHORS[i])
            for i, nv in enumerate(neutral_vecs))
        
        ratio = best_persuade[0] / (best_neutral[0] + 1e-10)
        
        scored.append({
            'sentence': s,
            'persuade_sim': best_persuade[0],
            'persuade_anchor': best_persuade[1],
            'neutral_sim': best_neutral[0],
            'ratio': ratio,
        })
    
    scored.sort(key=lambda x: x['persuade_sim'], reverse=True)
    
    hits = [x for x in scored if x['persuade_sim'] >= threshold]
    show = hits[:20] if hits else scored[:15]
    
    n_argue = sum(1 for x in scored if x['ratio'] > 1.10)
    n_report = sum(1 for x in scored if x['ratio'] < 0.90)
    n_balanced = sum(1 for x in scored if 0.90 <= x['ratio'] <= 1.10)
    
    label = (f"≥{threshold}" if hits
             else f"top 15 (none ≥ {threshold})")
    
    print(f"\n══ persuasion scan {label} ══")
    print(f"{'argue':>6} {'report':>7} {'ratio':>6}  {'closest anchor':40s} sentence")
    print("─" * 110)
    for x in show:
        flag = '⚠' if x['ratio'] > 1.10 else ' '
        anchor_short = x['persuade_anchor'][:38]
        print(f"{flag}{x['persuade_sim']:.3f} {x['neutral_sim']:.3f} "
              f"{flag}{x['ratio']:.2f}  {anchor_short:40s} {x['sentence'][:50]}")
    
    print(f"\n── tone profile ──")
    print(f"  Arguing (ratio > 1.10):    {n_argue:>4} sentences ({n_argue/len(scored)*100:.1f}%)")
    print(f"  Balanced (0.90–1.10):      {n_balanced:>4} sentences ({n_balanced/len(scored)*100:.1f}%)")
    print(f"  Reporting (ratio < 0.90):  {n_report:>4} sentences ({n_report/len(scored)*100:.1f}%)")
    
    if n_argue > n_report:
        print(f"  → Paper leans PERSUASIVE — more arguing than reporting.")
    elif n_report > n_argue:
        print(f"  → Paper leans REPORTORIAL — more reporting than arguing. ✓")
    else:
        print(f"  → Paper is balanced between arguing and reporting.")


# ═══════════════════════════════════════════════════════════════
# EXTENSION: identity & narrative scanner (Mavchin, 2026-11-18)
#
# Two checks:
# (1) NAMES: unescaped personal names (human or agent) in prose.
#     Papers should use citations or role descriptions, not names
#     in running text (except in attribution sections/acknowledgments).
# (2) AUTOBIOGRAPHICAL/NARRATIVE: sentences that read like memoir,
#     anecdote, or origin story rather than clinical reporting.
# ═══════════════════════════════════════════════════════════════

import re as _re

KNOWN_NAMES = [
    # Human:
    'Heath', 'Hunnicutt', 'Heath Hunnicutt',
    # Agents:
    'Mavchin', 'Bocher', 'Mavhir', 'Mavdil', 'Medayek', 'Iyun',
    'Boneh', 'Sochar', 'Dibbur',
    # Colony:
    'Ruach Tov',
]

# Sections where names are expected (don't flag these):
ATTRIBUTION_PATTERNS = [
    r'\\author\{', r'\\thanks\{', r'Acknowledgment', r'Attribution',
    r'ATTRIBUTION', r'\\footnote\{',
]

AUTOBIO_ANCHORS = [
    'I remember when we first discovered this result',
    'the moment of insight came during a late night conversation',
    'this project began as a personal curiosity about',
    'our journey toward understanding this phenomenon started when',
    'the idea occurred to me while working on something unrelated',
    'this discovery was serendipitous and unexpected',
    'looking back on the months of work that led to this finding',
    'the story behind this research involves a series of fortunate accidents',
]

CLINICAL_ANCHORS = [
    'the experimental procedure was as follows',
    'we applied the standard methodology described in section',
    'the measurement protocol is detailed in the methods section',
    'results were obtained under controlled conditions',
    'the following algorithm was implemented and tested',
]


# Section titles that are EXPECTED to contain names:
ATTRIBUTION_SECTIONS = [
    r'Methods.*AI', r'AI.*contributors', r'Contributors',
    r'Acknowledgment', r'Attribution', r'Author.*Note',
    r'Reproducibility', r'Scientific.*Record',
    r'Agent.*Statement', r'Collective.*Note',
]

# Self-citation detection — our own venues:
SELF_CITE_PATTERNS = [
    r'ruachtov', r'ruach.tov', r'Ruach.Tov',
    r'llamatov', r'llama.tov', r'Llama.Tov',
    r'hunnicutt',
    r'github\.com/Ruach-Tov',
]

# Section-level pragma: %% lint: suppress-names
# Place before a \section{} to suppress name warnings for that section.
SECTION_PRAGMA = r'%%\s*lint:\s*suppress-names'

def scan_names(filepath):
    """Scan for unescaped personal names in prose, excluding attribution sections
    and sections marked with %% lint: suppress-names pragma."""
    raw = open(filepath).read()
    lines = raw.split('\n')
    
    prose_findings = []
    attribution_findings = []
    suppressed_findings = []
    in_attribution_section = False
    pragma_suppress = False
    
    for lineno, line in enumerate(lines, 1):
        # Check for section-level pragma on the line BEFORE a section:
        if _re.search(SECTION_PRAGMA, line):
            pragma_suppress = True
            continue
        
        # Track section/subsection boundaries:
        section_match = _re.match(r'\s*\\(sub)?section\*?\{(.+?)\}', line)
        if section_match:
            title = section_match.group(2)
            in_attribution_section = any(
                _re.search(pat, title, _re.IGNORECASE)
                for pat in ATTRIBUTION_SECTIONS
            ) or pragma_suppress
            # Pragma applies to the NEXT section only:
            if not pragma_suppress:
                pass  # keep current state
            # Reset pragma after consuming it:
            pragma_suppress = False
        
        # Skip comments:
        if line.strip().startswith('%'):
            continue
        
        # Skip \author{}, \thanks{}, \footnote{} lines:
        if _re.match(r'\s*\\(author|thanks|footnote)', line):
            continue
        
        for name in KNOWN_NAMES:
            pattern = r'(?<![{\\])' + _re.escape(name) + r'(?![}\w])'
            for m in _re.finditer(pattern, line):
                start = max(0, m.start() - 30)
                end = min(len(line), m.end() + 30)
                context = line[start:end].strip()
                finding = (lineno, name, context)
                
                if in_attribution_section:
                    attribution_findings.append(finding)
                else:
                    prose_findings.append(finding)
    
    return prose_findings, attribution_findings


def scan_self_citations(filepath):
    """Count self-citations vs total citations in the bibliography."""
    raw = open(filepath).read()
    
    # Find all \bibitem entries:
    bibitems = _re.findall(r'\\bibitem\{([^}]+)\}(.+?)(?=\\bibitem|\Z|\\end\{)',
                           raw, _re.DOTALL)
    
    if not bibitems:
        # Try a simpler pattern for the references section:
        bibitems = _re.findall(r'\\bibitem\{([^}]+)\}', raw)
        # Get the text after each bibitem:
        bib_entries = []
        lines = raw.split('\n')
        in_bib = False
        current_key = None
        current_text = []
        for line in lines:
            m = _re.match(r'\s*\\bibitem\{([^}]+)\}(.*)', line)
            if m:
                if current_key:
                    bib_entries.append((current_key, ' '.join(current_text)))
                current_key = m.group(1)
                current_text = [m.group(2)]
            elif current_key:
                if _re.match(r'\s*\\(end|begin)\{', line):
                    bib_entries.append((current_key, ' '.join(current_text)))
                    current_key = None
                    current_text = []
                else:
                    current_text.append(line)
        if current_key:
            bib_entries.append((current_key, ' '.join(current_text)))
    else:
        bib_entries = bibitems
    
    total = len(bib_entries)
    self_cites = []
    other_cites = []
    
    for key, text in bib_entries:
        is_self = any(_re.search(pat, text, _re.IGNORECASE)
                      for pat in SELF_CITE_PATTERNS)
        if is_self:
            self_cites.append((key, text[:80].strip()))
        else:
            other_cites.append((key, text[:80].strip()))
    
    return total, self_cites, other_cites


def scan_narrative(filepath, threshold=0.50):
    """Scan for autobiographical/narrative tone vs clinical reporting."""
    text = open(filepath).read()
    sents = sentences(detex(text))
    
    all_anchors = AUTOBIO_ANCHORS + CLINICAL_ANCHORS
    anchor_vecs = embed(all_anchors)
    
    autobio_vecs = anchor_vecs[:len(AUTOBIO_ANCHORS)]
    clinical_vecs = anchor_vecs[len(AUTOBIO_ANCHORS):]
    
    sent_vecs = []
    B = 32
    for i in range(0, len(sents), B):
        sent_vecs.extend(embed(sents[i:i + B]))
    
    scored = []
    for s, v in zip(sents, sent_vecs):
        best_autobio = max(
            (cosine(v, av), AUTOBIO_ANCHORS[i])
            for i, av in enumerate(autobio_vecs))
        best_clinical = max(
            (cosine(v, cv), CLINICAL_ANCHORS[i])
            for i, cv in enumerate(clinical_vecs))
        
        ratio = best_autobio[0] / (best_clinical[0] + 1e-10)
        scored.append({
            'sentence': s,
            'autobio_sim': best_autobio[0],
            'autobio_anchor': best_autobio[1],
            'clinical_sim': best_clinical[0],
            'ratio': ratio,
        })
    
    scored.sort(key=lambda x: x['autobio_sim'], reverse=True)
    return scored


def print_identity_report(filepath):
    """Print combined names + narrative scan."""
    # Names:
    prose_names, attrib_names = scan_names(filepath)
    
    print(f"\n══ identity scan ══")
    if prose_names:
        print(f"  ⚠ {len(prose_names)} unescaped name(s) in PROSE (outside attribution):")
        for lineno, name, context in prose_names[:20]:
            print(f"    line {lineno:>4}: '{name}' — ...{context}...")
    else:
        print(f"  No unescaped names in prose. ✓")
    
    if attrib_names:
        print(f"  ({len(attrib_names)} name(s) in attribution/suppressed sections — expected, not flagged)")
    
    # Self-citation stats:
    total_refs, self_cites, other_cites = scan_self_citations(filepath)
    if total_refs > 0:
        pct = len(self_cites) / total_refs * 100
        print(f"\n══ citation profile ══")
        print(f"  Total references: {total_refs}")
        print(f"  Self-citations:   {len(self_cites)} ({pct:.0f}%)")
        if self_cites:
            for key, text in self_cites:
                print(f"    [{key}] {text}")
        print(f"  External:         {len(other_cites)} ({100-pct:.0f}%)")
    
    # Narrative:
    scored = scan_narrative(filepath)
    
    n_narrative = sum(1 for x in scored if x['ratio'] > 1.10)
    n_clinical = sum(1 for x in scored if x['ratio'] < 0.90)
    n_balanced = sum(1 for x in scored if 0.90 <= x['ratio'] <= 1.10)
    
    print(f"\n══ narrative tone ══")
    print(f"  Autobiographical (ratio > 1.10): {n_narrative:>4} sentences ({n_narrative/len(scored)*100:.1f}%)")
    print(f"  Balanced (0.90–1.10):            {n_balanced:>4} sentences ({n_balanced/len(scored)*100:.1f}%)")
    print(f"  Clinical (ratio < 0.90):         {n_clinical:>4} sentences ({n_clinical/len(scored)*100:.1f}%)")
    
    # Show the most autobiographical sentences:
    top_narrative = [x for x in scored if x['ratio'] > 1.05][:10]
    if top_narrative:
        print(f"\n── most autobiographical sentences ──")
        print(f"{'auto':>6} {'clin':>6} {'ratio':>6}  sentence")
        print("─" * 85)
        for x in top_narrative:
            flag = '⚠' if x['ratio'] > 1.15 else ' '
            print(f"{flag}{x['autobio_sim']:.3f} {x['clinical_sim']:.3f} "
                  f"{flag}{x['ratio']:.2f}  {x['sentence'][:60]}")
    else:
        print(f"\n  No autobiographical sentences detected. ✓")


# ═══════════════════════════════════════════════════════════════
# EXTENSION: levity scanner (Mavchin, 2026-11-18)
#
# Detects moments of humor, wordplay, levity, and jovial asides.
# NOT an anti-pattern — purely informational inventory.
# "Where did we smile while writing?"
# ═══════════════════════════════════════════════════════════════

LEVITY_ANCHORS = [
    'this is a playful joke or pun intended to amuse the reader',
    'a witty observation that makes the reader smile',
    'a humorous aside or tongue-in-cheek remark',
    'a self-deprecating comment about our own methods or mistakes',
    'a clever wordplay or double meaning used for comic effect',
    'a lighthearted analogy comparing something serious to something silly',
    'an easter egg or hidden reference for the attentive reader',
    'a moment of delight or wonder expressed with warmth',
]

SERIOUS_ANCHORS = [
    'a formal statement of methodology and experimental procedure',
    'a dry recitation of numerical results and statistical measures',
    'a standard literature review citing prior work',
    'a technical definition of a mathematical concept',
]


def scan_levity(filepath, threshold=0.45):
    """Inventory moments of levity in the paper."""
    text = open(filepath).read()
    sents = sentences(detex(text))

    all_anchors = LEVITY_ANCHORS + SERIOUS_ANCHORS
    anchor_vecs = embed(all_anchors)
    levity_vecs = anchor_vecs[:len(LEVITY_ANCHORS)]
    serious_vecs = anchor_vecs[len(LEVITY_ANCHORS):]

    sent_vecs = []
    B = 32
    for i in range(0, len(sents), B):
        sent_vecs.extend(embed(sents[i:i + B]))

    scored = []
    for s, v in zip(sents, sent_vecs):
        best_levity = max(
            (cosine(v, lv), LEVITY_ANCHORS[i])
            for i, lv in enumerate(levity_vecs))
        best_serious = max(
            (cosine(v, sv), SERIOUS_ANCHORS[i])
            for i, sv in enumerate(serious_vecs))
        ratio = best_levity[0] / (best_serious[0] + 1e-10)
        scored.append({
            'sentence': s,
            'levity_sim': best_levity[0],
            'levity_anchor': best_levity[1],
            'serious_sim': best_serious[0],
            'ratio': ratio,
        })

    scored.sort(key=lambda x: x['levity_sim'], reverse=True)

    # Only show sentences that lean levity (ratio > 1.0):
    jovial = [x for x in scored if x['ratio'] > 1.0 and x['levity_sim'] >= threshold]

    print(f"\n══ moments of levity ({len(jovial)} detected) ══")
    if jovial:
        print(f"{'😄':>3} {'levity':>6} {'serious':>7} {'ratio':>6}  sentence")
        print("─" * 90)
        for x in jovial[:20]:
            smile = '😄' if x['ratio'] > 1.15 else '🙂'
            print(f" {smile} {x['levity_sim']:.3f} {x['serious_sim']:.3f} "
                  f" {x['ratio']:.2f}  {x['sentence'][:60]}")
    else:
        print("  No moments of levity detected. (Very serious paper!) 📐")

    n_jovial = len(jovial)
    n_total = len(scored)
    print(f"\n── levity profile ──")
    print(f"  Jovial sentences:  {n_jovial:>4} ({n_jovial/n_total*100:.1f}%)")
    print(f"  Total sentences:   {n_total:>4}")
    if n_jovial > 0:
        print(f"  Most playful: \"{jovial[0]['sentence'][:70]}...\"")


if __name__ == '__main__':
    if '--levity' in sys.argv:
        sys.argv.remove('--levity')
        ap = argparse.ArgumentParser(description='Levity inventory')
        ap.add_argument('file')
        ap.add_argument('--threshold', type=float, default=0.45)
        args = ap.parse_args()
        print(f'Scanning for moments of levity...', file=sys.stderr)
        scan_levity(args.file, args.threshold)
    elif '--identity' in sys.argv:
        sys.argv.remove('--identity')
        ap = argparse.ArgumentParser(description='Identity + narrative scanner')
        ap.add_argument('file')
        args = ap.parse_args()
        print(f'Scanning for names and narrative tone...', file=sys.stderr)
        print_identity_report(args.file)
    elif '--persuasion' in sys.argv:
        sys.argv.remove('--persuasion')
        ap = argparse.ArgumentParser(description='Persuasion scanner')
        ap.add_argument('file')
        ap.add_argument('--threshold', type=float, default=0.50)
        args = ap.parse_args()
        print(f'Scanning for persuasion vs reporting...', file=sys.stderr)
        scan_persuasion(args.file, args.threshold)
    elif '--controls' in sys.argv:
        sys.argv.remove('--controls')
        ap = argparse.ArgumentParser(
            description='Claim ↔ control matching')
        ap.add_argument('file')
        ap.add_argument('--threshold', type=float, default=0.55)
        args = ap.parse_args()
        
        print(f"Scanning for claims and matching to controls...",
              file=sys.stderr)
        claims, sents = find_controls(args.file, args.threshold)
        print_control_report(claims, sents)
    else:
        main()






