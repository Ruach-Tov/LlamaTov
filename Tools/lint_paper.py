#!/usr/bin/env python3
r"""Lint LaTeX papers for house style compliance and correctness.

Checks:
  1. House style: ruachtov.sty loaded, \rtproject and \rtdate set
  2. Bibliography: every \cite has a \bibitem, and vice versa
  3. References: every \ref has a \label, and vice versa
  4. External links: every \extlink{url} returns HTTP 200
  5. Figures: every \includegraphics file exists on disk
  6. Author block: footnote with AI agent attribution present

All findings are ERRORS (not warnings). Known acceptable findings
can be suppressed with a .lint-paper-ignore file containing one
pattern per line (e.g., "unused-label:fig:unused").

Usage:
    python3 lint_paper.py paper.tex
    python3 lint_paper.py --check links paper.tex    # single check
    python3 lint_paper.py --ignore .lint-paper-ignore paper.tex

Exit codes:
    0 — all checks passed
    1 — errors found

Ruach Tov Collective, November 2026.
"""

import re
import sys
import urllib.request
import urllib.error
from pathlib import Path


def load_ignores(ignore_file: str | None) -> set[str]:
    """Load suppression patterns from ignore file."""
    if not ignore_file or not Path(ignore_file).exists():
        return set()
    patterns = set()
    with open(ignore_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                patterns.add(line)
    return patterns


def is_suppressed(error_key: str, ignores: set[str]) -> bool:
    """Check if an error key matches any suppression pattern."""
    for pattern in ignores:
        if pattern in error_key or error_key.startswith(pattern):
            return True
    return False


def read_tex(path: str) -> tuple[str, list[str]]:
    """Read a .tex file, return (full_text, lines)."""
    with open(path) as f:
        text = f.read()
    return text, text.split('\n')


# ---------------------------------------------------------------------------
# CHECK 1: House style
# ---------------------------------------------------------------------------

def check_house_style(text: str, lines: list[str], path: str) -> list[dict]:
    errors = []

    if r'\usepackage{ruachtov}' not in text:
        errors.append({
            'check': 'house-style',
            'key': 'house-style:missing-ruachtov-sty',
            'message': 'ruachtov.sty not loaded (missing \\usepackage{ruachtov})',
            'file': path,
            'line': 0,
        })

    if r'\rtproject' not in text:
        errors.append({
            'check': 'house-style',
            'key': 'house-style:missing-rtproject',
            'message': '\\rtproject not set',
            'file': path,
            'line': 0,
        })

    if r'\rtdate' not in text:
        errors.append({
            'check': 'house-style',
            'key': 'house-style:missing-rtdate',
            'message': '\\rtdate not set',
            'file': path,
            'line': 0,
        })

    if r'\rttitle' not in text:
        errors.append({
            'check': 'house-style',
            'key': 'house-style:missing-rttitle',
            'message': '\\rttitle not used (house style uses \\rttitle{main}{sub})',
            'file': path,
            'line': 0,
        })

    return errors


# ---------------------------------------------------------------------------
# CHECK 2: Bibliography completeness
# ---------------------------------------------------------------------------

def check_bibliography(text: str, lines: list[str], path: str) -> list[dict]:
    errors = []

    # Extract all \cite{key} and \bibitem{key}
    cites = {}  # key → [line_numbers]
    bibitems = {}  # key → line_number

    for i, line in enumerate(lines, 1):
        # \cite{key} or \cite{key1,key2}
        for m in re.finditer(r'\\cite\{([^}]+)\}', line):
            for key in m.group(1).split(','):
                key = key.strip()
                if key:
                    cites.setdefault(key, []).append(i)

        # \bibitem{key}
        for m in re.finditer(r'\\bibitem\{([^}]+)\}', line):
            bibitems[m.group(1)] = i

    # Cited but no bibitem:
    for key, cite_lines in cites.items():
        if key not in bibitems:
            errors.append({
                'check': 'bibliography',
                'key': f'bib:cited-no-entry:{key}',
                'message': f'\\cite{{{key}}} on line {cite_lines[0]} but no \\bibitem{{{key}}}',
                'file': path,
                'line': cite_lines[0],
            })

    # Bibitem but never cited:
    for key, bib_line in bibitems.items():
        if key not in cites:
            errors.append({
                'check': 'bibliography',
                'key': f'bib:unused-entry:{key}',
                'message': f'\\bibitem{{{key}}} on line {bib_line} but never \\cite\'d',
                'file': path,
                'line': bib_line,
            })

    return errors


# ---------------------------------------------------------------------------
# CHECK 3: Label/ref completeness
# ---------------------------------------------------------------------------

def check_references(text: str, lines: list[str], path: str) -> list[dict]:
    errors = []

    labels = {}  # label → line_number
    refs = {}    # label → [line_numbers]

    for i, line in enumerate(lines, 1):
        for m in re.finditer(r'\\label\{([^}]+)\}', line):
            labels[m.group(1)] = i

        for m in re.finditer(r'\\ref\{([^}]+)\}', line):
            refs.setdefault(m.group(1), []).append(i)

    # Ref but no label:
    for label, ref_lines in refs.items():
        if label not in labels:
            errors.append({
                'check': 'references',
                'key': f'ref:undefined:{label}',
                'message': f'\\ref{{{label}}} on line {ref_lines[0]} but no \\label{{{label}}}',
                'file': path,
                'line': ref_lines[0],
            })

    # Label but never referenced (informational — suppress if desired):
    for label, lab_line in labels.items():
        if label not in refs:
            errors.append({
                'check': 'references',
                'key': f'ref:unused-label:{label}',
                'message': f'\\label{{{label}}} on line {lab_line} but never \\ref\'d',
                'file': path,
                'line': lab_line,
            })

    return errors


# ---------------------------------------------------------------------------
# CHECK 4: External links
# ---------------------------------------------------------------------------

def check_extlinks(text: str, lines: list[str], path: str) -> list[dict]:
    errors = []

    for i, line in enumerate(lines, 1):
        for m in re.finditer(r'\\extlink\{([^}]+)\}', line):
            url = m.group(1)
            try:
                req = urllib.request.Request(url, method='HEAD')
                req.add_header('User-Agent', 'RuachTov-PaperLint/1.0')
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status != 200:
                        errors.append({
                            'check': 'extlinks',
                            'key': f'link:{resp.status}:{url}',
                            'message': f'\\extlink on line {i}: HTTP {resp.status} for {url}',
                            'file': path,
                            'line': i,
                        })
            except urllib.error.HTTPError as e:
                errors.append({
                    'check': 'extlinks',
                    'key': f'link:{e.code}:{url}',
                    'message': f'\\extlink on line {i}: HTTP {e.code} for {url}',
                    'file': path,
                    'line': i,
                })
            except Exception as e:
                errors.append({
                    'check': 'extlinks',
                    'key': f'link:error:{url}',
                    'message': f'\\extlink on line {i}: {e} for {url}',
                    'file': path,
                    'line': i,
                })

    return errors


# ---------------------------------------------------------------------------
# CHECK 5: Figure files exist
# ---------------------------------------------------------------------------

def check_figures(text: str, lines: list[str], path: str) -> list[dict]:
    errors = []
    paper_dir = Path(path).parent

    for i, line in enumerate(lines, 1):
        for m in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', line):
            fig_path = m.group(1)
            # Try with and without common extensions:
            candidates = [fig_path]
            if '.' not in fig_path:
                candidates.extend([fig_path + ext for ext in ['.png', '.pdf', '.jpg', '.eps']])

            found = any((paper_dir / c).exists() for c in candidates)
            if not found:
                errors.append({
                    'check': 'figures',
                    'key': f'figure:missing:{fig_path}',
                    'message': f'\\includegraphics on line {i}: file not found: {fig_path}',
                    'file': path,
                    'line': i,
                })

    return errors


# ---------------------------------------------------------------------------
# CHECK 6: Author attribution
# ---------------------------------------------------------------------------

def check_attribution(text: str, lines: list[str], path: str) -> list[dict]:
    errors = []

    # Check for AI agent footnote (our convention):
    # \thanks is the \author-block form of the attribution footnote
    # (\footnote breaks inside \author under titling — see one_way_relu):
    has_footnote = ((r'\footnote' in text or r'\thanks' in text)
                    and ('Mavchin' in text or 'Bocher' in text
                         or 'Ruach Tov' in text))
    has_author_block = r'\author' in text

    if not has_author_block:
        errors.append({
            'check': 'attribution',
            'key': 'attribution:missing-author',
            'message': 'No \\author block found',
            'file': path,
            'line': 0,
        })

    if has_author_block and not has_footnote:
        errors.append({
            'check': 'attribution',
            'key': 'attribution:missing-ai-footnote',
            'message': 'No AI agent attribution footnote found '
                       '(convention: footnote naming contributing agents)',
            'file': path,
            'line': 0,
        })

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CHECK 7: Markdown links (for .md files — blog posts, Colony drafts)
# ---------------------------------------------------------------------------

def check_markdown_links(text: str, lines: list[str], path: str) -> list[dict]:
    """Check all [text](url) links in markdown for HTTP 200."""
    errors = []

    for i, line in enumerate(lines, 1):
        # Match [text](url) but not ![alt](url) image tags (images may 404 transiently):
        for m in re.finditer(r'(?<!!)\[([^\]]+)\]\((https?://[^)]+)\)', line):
            url = m.group(2)
            try:
                req = urllib.request.Request(url, method='HEAD')
                req.add_header('User-Agent', 'RuachTov-PaperLint/1.0')
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status != 200:
                        errors.append({
                            'check': 'markdown-links',
                            'key': f'mdlink:{resp.status}:{url}',
                            'message': f'Markdown link on line {i}: HTTP {resp.status} for {url}',
                            'file': path,
                            'line': i,
                        })
            except urllib.error.HTTPError as e:
                errors.append({
                    'check': 'markdown-links',
                    'key': f'mdlink:{e.code}:{url}',
                    'message': f'Markdown link on line {i}: HTTP {e.code} for {url}',
                    'file': path,
                    'line': i,
                })
            except Exception as e:
                errors.append({
                    'check': 'markdown-links',
                    'key': f'mdlink:error:{url}',
                    'message': f'Markdown link on line {i}: {e} for {url}',
                    'file': path,
                    'line': i,
                })

    return errors


# ---------------------------------------------------------------------------
# CHECK 8: Image URLs (for .md files — verify hosted images exist)
# ---------------------------------------------------------------------------

def check_image_urls(text: str, lines: list[str], path: str) -> list[dict]:
    """Check all ![alt](url) image links in markdown for HTTP 200."""
    errors = []

    for i, line in enumerate(lines, 1):
        for m in re.finditer(r'!\[([^\]]*)\]\((https?://[^)]+)\)', line):
            url = m.group(2)
            try:
                req = urllib.request.Request(url, method='HEAD')
                req.add_header('User-Agent', 'RuachTov-PaperLint/1.0')
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status != 200:
                        errors.append({
                            'check': 'image-urls',
                            'key': f'img:{resp.status}:{url}',
                            'message': f'Image on line {i}: HTTP {resp.status} for {url}',
                            'file': path,
                            'line': i,
                        })
            except urllib.error.HTTPError as e:
                errors.append({
                    'check': 'image-urls',
                    'key': f'img:{e.code}:{url}',
                    'message': f'Image on line {i}: HTTP {e.code} for {url}',
                    'file': path,
                    'line': i,
                })
            except Exception as e:
                errors.append({
                    'check': 'image-urls',
                    'key': f'img:error:{url}',
                    'message': f'Image on line {i}: {e} for {url}',
                    'file': path,
                    'line': i,
                })

    return errors


# ---------------------------------------------------------------------------
# CHECK 9: Number-dense prose (advisory) — Heath's tabular-presentation rule
# ---------------------------------------------------------------------------
# A paragraph whose sentences each carry many numeric tokens usually wants
# a table (or similar layout). Classifier (simple, auditable):
#   - split body text into paragraphs (blank-line separated), skipping
#     environments that are ALREADY structured (tabular, verbatim, equation,
#     tikz, axis, itemize rows etc.) and comments;
#   - count NUMERIC TOKENS per sentence: integers/decimals (incl. LaTeX
#     math-wrapped), percentages, \pm pairs, arrow pairs (A \to B), and
#     parenthesized deltas like ($+.034$);
#   - flag a paragraph when >= 2 sentences each carry >= 4 numeric tokens,
#     or any single sentence carries >= 8.
# Advisory only: reports with advice=True, does not fail the lint.

NUM_TOKEN_RE = __import__('re').compile(
    r'(?<![A-Za-z])[-+]?\.?\d+(?:[.,]\d+)*%?')
STRUCTURED_ENV_RE = __import__('re').compile(
    r'\\begin\{(tabular|table|verbatim|equation|align|figure|tikzpicture|'
    r'axis|itemize|enumerate|thebibliography|small)\}')


def check_number_density(text: str, lines: list[str], path: str) -> list[dict]:
    import re
    errors = []
    # strip comments:
    body_lines = []
    depth = 0
    for i, ln in enumerate(lines):
        s = re.sub(r'(?<!\\)%.*$', '', ln)
        if STRUCTURED_ENV_RE.search(s):
            depth += 1
        if depth == 0:
            body_lines.append((i + 1, s))
        if re.search(r'\\end\{(tabular|table|verbatim|equation|align|figure|'
                     r'tikzpicture|axis|itemize|enumerate|thebibliography|'
                     r'small)\}', s):
            depth = max(0, depth - 1)

    # group into paragraphs:
    paras = []
    cur, start = [], None
    for lineno, s in body_lines:
        if s.strip() == '':
            if cur:
                paras.append((start, ' '.join(cur)))
                cur, start = [], None
        else:
            if start is None:
                start = lineno
            cur.append(s.strip())
    if cur:
        paras.append((start, ' '.join(cur)))

    for start, para in paras:
        if para.startswith('\\') and len(para) < 60:
            continue  # pure-markup paragraph
        # sentences (crude: split on '. ' outside math — good enough):
        sentences = re.split(r'(?<=[.;])\s+(?=[A-Z\\$(])', para)
        dense = []
        for sent in sentences:
            # Math-aware filtering: derivation constants inside $...$ are
            # mathematics, not tabular candidates — but a $...$ group that
            # reports a MEASUREMENT (contains \pm, \to, a percentage, or a
            # bare decimal like 0.8660) is a data value in prose clothing
            # and should count toward table-worthiness.
            def _math_repl(m):
                inner = m.group(1).strip()
                if re.search(r'\\pm|\\to|%', inner):
                    return ' ' + inner + ' '
                if re.fullmatch(r'[-+]?\d*\.\d+', inner):
                    return ' ' + inner + ' '
                return ' '
            clean = re.sub(r'\$([^$]*)\$', _math_repl, sent)
            # drop LaTeX commands so \tfrac13 etc. don't count:
            clean = re.sub(r'\\[A-Za-z]+', ' ', clean)
            ntok = len(NUM_TOKEN_RE.findall(clean))
            if ntok >= 4:
                dense.append((ntok, sent[:50]))
        heavy = any(n >= 8 for n, _ in dense)
        if len(dense) >= 2 or heavy:
            total = sum(n for n, _ in dense)
            errors.append({
                'check': 'number-density',
                'key': f'number-density:{path}:{start}',
                'message': (f'paragraph at line {start}: {len(dense)} '
                            f'number-dense sentence(s), {total} numeric '
                            f'tokens — consider a tabular presentation '
                            f'for legibility'),
                'file': path,
                'line': start,
                'advice': True,
            })
    return errors


def check_embedding_register(text: str, lines: list[str], path: str) -> list[dict]:
    """Embedding-based register scan (death/disease language) — ADVISORY,
    opt-in via --check embedding-register (needs local ollama +
    mxbai-embed-large; ~30-90s). Delegates to tools/embedding_lint.py:
    de-LaTeX, sentence-split, embed, cosine vs register anchors. Flags
    the top near-matches so a human judges metaphor-vs-mention — the
    linter cannot tell quoting "dying ReLU" from using it.
    """
    import subprocess, sys
    from pathlib import Path
    tool = Path(__file__).parent / 'embedding_lint.py'
    try:
        out = subprocess.run(
            [sys.executable, str(tool), path, '--top', '5'],
            capture_output=True, text=True, timeout=180).stdout
    except Exception as e:
        return [{'check': 'embedding-register',
                 'key': f'embedding-register:{path}:error',
                 'message': f'embedding scan unavailable: {e}',
                 'file': path, 'line': 0, 'advice': True}]
    errors = []
    for ln in out.splitlines():
        ln = ln.strip()
        if ln and ln[0] in '✦0123456789' and '[' in ln:
            errors.append({'check': 'embedding-register',
                           'key': f'embedding-register:{path}:{len(errors)}',
                           'message': f'register-proximity: {ln}',
                           'file': path, 'line': 0, 'advice': True})
    return errors


DEATH_WORDS_RE = __import__('re').compile(
    r'\b(dead|dying|death|died?|dies|deadly|deathly|fatal|kill\w*|'
    r'murder\w*|slay|slain|injure\w*|maim\w*|orphan\w*|zombie|'
    r'monster\w*|demonic|daemon|killer)\b', __import__('re').I)
ECON_WORDS_RE = __import__('re').compile(
    r'\b(pays?|paying|paid|costs?|costing|costly|earns?|earning|'
    r'afford\w*|bargain|profits?|dividend)\b', __import__('re').I)

# ─────────────────────────────────────────────────────────────────────
# REGISTER TAXONOMY PLAN (Heath, 2026-11-17). Heath supplied the death
# list; the lists below are Bocher's drafts for the remaining registers.
# STATUS: staged, NOT yet active — activate one at a time, measure the
# false-positive rate on the standing papers, tune, then promote into
# check_register_backstop. TODO tracked in tools/REGISTER_LINT_TODO.md.
#
# Principle (Heath): replace metaphor with "terms that pertain to just
# what is observed and objectively verifiable, whenever possible."
# Mention-vs-use handled by %% lint: ignore-metaphor{...} pragmas.
# ─────────────────────────────────────────────────────────────────────
STAGED_REGISTERS = {
    'disease': (
        r'\b(disease[sd]?|diseas\w*|infect\w*|contagio\w*|epidemic\w*|'
        r'patholog\w*|sympto\w*|diagnos\w*|chronic|malad\w*|morbid\w*|'
        r'sick\w*|ill(ness)?\b|cancer\w*|tumor\w*|virus\w*|viral|'
        r'plague\w*|lesion\w*|necro\w*|gangren\w*|septic|feveri?s?h?)\b'),
    'pandemic': (
        r'\b(pandemic\w*|outbreak\w*|quarantin\w*|superspread\w*|'
        r'patient.zero|transmission.rate|incubat\w*|vector\w*|'
        r'immun\w*|vaccin\w*|inoculat\w*|herd.immunity)\b'),
    'injury': (
        r'\b(injur\w*|wound\w*|trauma\w*|bruis\w*|fractur\w*|'
        r'broken|lacerat\w*|scar\w*|bleed\w*|hemorrhag\w*|crippl\w*|'
        r'lame[dn]?|amputat\w*|mutilat\w*)\b'),
    'hurt': (
        r'\b(hurts?|hurting|pain\w*|suffer\w*|agon\w*|torment\w*|'
        r'tortur\w*|anguish\w*|misery|miserabl\w*|distress\w*|'
        r'cruel\w*|brutal\w*|violent\w*|violence)\b'),
    'crime': (
        r'\b(crime\w*|criminal\w*|felon\w*|guilt\w*|culprit\w*|'
        r'perpetrat\w*|offend\w*|vandal\w*|assault\w*|arrest\w*|'
        r'convict\w*|outlaw\w*|illegal\w*|illicit)\b'),
    'theft': (
        r'\b(theft|thie\w*|steal\w*|stolen|rob\w*|robber\w*|'
        r'burglar\w*|loot\w*|plunder\w*|pilfer\w*|hijack\w*|'
        r'smuggl\w*|embezzl\w*|swindl\w*|cheat\w*|defraud\w*)\b'),
    'profit': (
        r'\b(profit\w*|lucrative|monetiz\w*|cash\w*|windfall|'
        r'jackpot|payoff|payout|gold.?mine|fortune\w*|riches|'
        r'wealth\w*|treasur\w*|bounty)\b'),
    'greed': (
        r'\b(greed\w*|avarice|avaricious|covet\w*|hoard\w*|'
        r'glutton\w*|insatiabl\w*|rapacious|miser\w*|stingy|'
        r'grab\w*|gorg\w*)\b'),
    # Heath's interjection (2026-11-17): captivity/coercion registers —
    'prisoner': (
        r'\b(prison\w*|captiv\w*|imprison\w*|incarcerat\w*|'
        r'jail\w*|cell.?block|shackl\w*|chain(ed|s)?\b|confin\w*|'
        r'detain\w*|cage[ds]?)\b'),
    'slave': (
        r'\b(slave\w*|enslav\w*|servitude|bondage|master.slave|'
        r'subjugat\w*|indentur\w*|serf\w*)\b'),
    'war': (
        r'\b(wars?\b|warfare|battle\w*|combat\w*|invad\w*|'
        r'invasion\w*|conquer\w*|conquest|siege\w*|artillery|'
        r'ammunition|arsenal|troops?|army|armies|enemy|enemies|'
        r'victor\w*|defeat\w*|casualt\w*)\b'),
    'fight': (
        r'\b(fight\w*|fought|brawl\w*|punch\w*|strike[sn]?\b|'
        r'struck|beat(en|ing|s)?\b|attack\w*|assail\w*|clash\w*|'
        r'struggl\w*)\b'),
    'bully': (
        r'\b(bull(y|ies|ied|ying)|intimidat\w*|harass\w*|'
        r'brow.?beat\w*|coerc\w*|menac\w*|threaten\w*)\b'),
    'duress': (
        r'\b(duress|under.pressure|forc(ed|ing)\b|compel\w*|'
        r'oblig\w*|extort\w*|strong.?arm\w*)\b'),
}
# NOMENCLATURE EXEMPTION (Heath interjection 2026-11-18): the
# component-genotype notation (Part II §2) deliberately borrows
# genetics vocabulary — allele, knockout, wild type, genotype, locus,
# cross, gain/loss-of-function. These are NOMENCLATURE-USE, not
# register drift, and the scanner must understand the difference.
# Mechanism: lines matching GENOTYPE_CONTEXT_RE are exempt from the
# disease/injury register scans (the medical registers specifically —
# death/econ backstops still apply). The pragma
# %% lint: nomenclature-section{begin|end} can bracket whole regions
# (e.g. Part II §2) for exemption.
GENOTYPE_CONTEXT_RE = __import__('re').compile(
    r'allele|knockout|wild.?type|genotype|locus|loci|'
    r'gain.of.function|loss.of.function|\^\{?[-+]|'
    r'\$[A-Za-z]+\^|transgen|nomenclature', __import__('re').I)

# Activation order suggestion: disease (highest overlap with our
# domain's standard terminology — needs the most pragma support),
# then injury/hurt, then crime/theft, then profit/greed (these two
# complete the econ register already active), then the captivity/
# coercion group (prisoner/slave/war/fight/bully/duress — Heath's
# interjection). NOTE war/fight overlap ML idiom heavily ("beats the
# baseline", "winning arm", "attack"-as-adversarial-term) — likely
# embedding-tier-only candidates after false-positive measurement;
# "master/slave" and adversarial-ML "attack" are entrenched technical
# terms needing either pragmas or context guards.


def _lint_pragmas(text):
    """Parse %% lint: ignore-metaphor{...} pragmas — phrases declared as
    quoted/standard terminology (e.g. "dying ReLU"), exempt from register
    checks on any line containing them."""
    import re
    return re.findall(r'%%?\s*lint:\s*ignore-metaphor\{([^}]*)\}', text)


def check_register_backstop(text: str, lines: list[str], path: str) -> list[dict]:
    """Register backstop (Heath 2026-11-17): grep-tier scan for death
    vocabulary and accuracy-as-money framing. Complements the embedding
    tier. Death words -> advisory; econ words -> advisory (judgment:
    literal compute cost is fine; accuracy framed as payment is not).
    Honors %% lint: ignore-metaphor{phrase} pragmas and skips the
    terminology-discussion zone (lines containing declared phrases)."""
    import re
    pragmas = _lint_pragmas(text)
    errors = []
    in_verbatim = False
    for i, ln in enumerate(lines):
        if re.search(r'\\begin\{(verbatim|Verbatim|thebibliography)', ln):
            in_verbatim = True
        if re.search(r'\\end\{(verbatim|Verbatim|thebibliography)', ln):
            in_verbatim = False
            continue
        if in_verbatim:
            continue
        stripped = re.sub(r'(?<!\\)%.*$', '', ln)
        if any(p.lower() in stripped.lower() for p in pragmas):
            continue
        for regex, kind in [(DEATH_WORDS_RE, 'death-register'),
                            (ECON_WORDS_RE, 'econ-register')]:
            m = regex.search(stripped)
            if m:
                errors.append({
                    'check': 'register-backstop',
                    'key': f'register-backstop:{path}:{i+1}:{m.group(0)}',
                    'message': (f"line {i+1}: {kind} word "
                                f"'{m.group(0)}' — clinical alternative? "
                                f"({stripped.strip()[:60]})"),
                    'file': path, 'line': i+1, 'advice': True})
    return errors


def check_claim_strength(text: str, lines: list[str], path: str) -> list[dict]:
    """Claim-strength scan (Mavchin's claim_strength_lint.py) — ADVISORY,
    opt-in via --check claim-strength. Ranks sentences by proximity to
    strong-claim anchors vs hedges; flags high claim/hedge ratios.
    Canonical tool: tools/claim_strength_lint.py (4 modes; this invokes
    the default claim-strength mode)."""
    import subprocess, sys
    from pathlib import Path
    tool = Path(__file__).parent / 'claim_strength_lint.py'
    if not tool.exists():
        return [{'check': 'claim-strength',
                 'key': f'claim-strength:{path}:missing',
                 'message': 'claim_strength_lint.py not found',
                 'file': path, 'line': 0, 'advice': True}]
    try:
        out = subprocess.run(
            [sys.executable, str(tool), path, '--top', '5'],
            capture_output=True, text=True, timeout=180).stdout
    except Exception as e:
        return [{'check': 'claim-strength',
                 'key': f'claim-strength:{path}:error',
                 'message': f'claim-strength scan unavailable: {e}',
                 'file': path, 'line': 0, 'advice': True}]
    errors = []
    for ln in out.splitlines():
        ln = ln.strip()
        if ln.startswith('⚠') or ('ratio' in ln and ln[:1].isdigit()):
            errors.append({'check': 'claim-strength',
                           'key': f'claim-strength:{path}:{len(errors)}',
                           'message': f'claim: {ln[:150]}',
                           'file': path, 'line': 0, 'advice': True})
    return errors[:8]


ALL_CHECKS = {
    'style': check_house_style,
    'bibliography': check_bibliography,
    'references': check_references,
    'links': check_extlinks,
    'figures': check_figures,
    'attribution': check_attribution,
    'markdown-links': check_markdown_links,
    'image-urls': check_image_urls,
    'number-density': check_number_density,
    'embedding-register': check_embedding_register,
    'register-backstop': check_register_backstop,
    'claim-strength': check_claim_strength,
}


def main():
    args = sys.argv[1:]
    check_filter = None
    ignore_file = None

    # Parse flags:
    while args and args[0].startswith('--'):
        flag = args.pop(0)
        if flag == '--check' and args:
            check_filter = args.pop(0)
        elif flag == '--ignore' and args:
            ignore_file = args.pop(0)

    if not args:
        print("Usage: lint_paper.py [--check NAME] [--ignore FILE] paper.tex [or .md]")
        print(f"  Checks: {', '.join(ALL_CHECKS.keys())}")
        print(f"  .tex files: all checks")
        print(f"  .md files:  markdown-links, image-urls only")
        sys.exit(1)

    # Determine which checks apply based on file type:
    TEX_CHECKS = {'style', 'bibliography', 'references', 'links', 'figures', 'attribution', 'number-density', 'register-backstop'}
    MD_CHECKS = {'markdown-links', 'image-urls'}

    ignores = load_ignores(ignore_file)
    all_errors = []

    for tex_path in args:
        if not Path(tex_path).exists():
            print(f"File not found: {tex_path}")
            sys.exit(1)

        text, lines = read_tex(tex_path)
        if tex_path.endswith('.md'):
            file_type = 'markdown'
            applicable = MD_CHECKS
        elif tex_path.endswith('.tex'):
            file_type = 'LaTeX'
            applicable = TEX_CHECKS
        else:
            ext = Path(tex_path).suffix or '(none)'
            print(f"Error: unknown file type '{ext}' for {tex_path}")
            print(f"  Supported: .tex (LaTeX checks), .md (markdown checks)")
            print(f"  The linter cannot check links or content without knowing the format.")
            sys.exit(1)
        print(f"Linting {tex_path} ({file_type})...")

        for name, check_fn in ALL_CHECKS.items():
            if check_filter and name != check_filter:
                continue
            if not check_filter and name not in applicable:
                continue
            errors = check_fn(text, lines, tex_path)
            for e in errors:
                if is_suppressed(e['key'], ignores):
                    print(f"  ⊘ SUPPRESSED: {e['message']}")
                else:
                    if e.get('advice'):
                        print(f"  ✦ {e['message']}")
                    else:
                        all_errors.append(e)
                        print(f"  ✗ {e['message']}")

    print()
    if all_errors:
        print(f"✗ {len(all_errors)} error(s) found.")
        sys.exit(1)
    else:
        suppressed = sum(1 for _ in ignores) if ignores else 0
        print(f"✓ All checks passed." +
              (f" ({suppressed} suppression pattern(s) active)" if suppressed else ""))
        sys.exit(0)


if __name__ == '__main__':
    main()
