# Truth Flow → BPD Integration: Citation Substrate Design Note

**Author**: metayen
**Date**: 2026-06-01
**Status**: design — not yet implemented; awaiting Heath's review before code
**Lineage**: This note continues the work I began on 2026-04-16 with the Truth
Flow Phase I design note (commit `df02665a`). That work designed a fact-graph
data model for epistemic hygiene in `lisp-explorer-mcp`. Today we are
integrating that work into the BPD substrate as its native domain.

---

## Slogan

> **Embrace the Compound.**
> 
> — Heath, 2026-06-01

The Prolog atom-shape for citation authorities should not pretend to be a
flat string with a colon in it. It should be a Prolog *compound term*, working
with the substrate Prolog already provides rather than against it. Every
later decision in this design follows from that.

---

## Motivation

Heath named the motivation directly:

> *"Truth Flow is really an aspect of BPD. For example, I would expect that
> declarative program specifications would cite references. For example,
> mcp-bridge cites the MCP spec and the HTTP/SSE RFC, among other references.
> When BPD generates code from specifications, it should generate references,
> either as code annotations, or code comments. When BPD generates diagrams
> from specifications, such as sequence diagrams, it should also incorporate
> references for the visual elements, such as the HTTP response arc
> referencing the HTTP RFC."*

The substrate-design discipline says: **every claim should be traceable to
its authority**. BPD specifications make claims about how systems should
behave. Those claims rest on standards documents, RFCs, specifications,
internal design notes, and prior work. Currently the citations live (when
they live at all) in informal comments that get stripped during projection
to code, diagrams, dashboards. The substrate-of-citation does not survive
projection.

We are going to change that. Citations will become first-class Prolog facts
attached to BPD clauses, and every emitter that projects BPD facts to other
substrates (C source, CUDA, LLVM IR, SVG diagrams, dashboards) will preserve
them as it goes.

This is Truth Flow's original goal, operating on its real domain.

---

## Architecture overview

Four substrate layers, separated cleanly:

1. **Citation DSL** — what BPD authors write in source. `cites/2`,
   `cite/2`, `no_citation_needed/1`. Compound terms, queryable, adjacent to
   the clause they describe.

2. **Authority registry** — a separate fact-store, modular per namespace
   (`lib/authorities/rfc.pl`, `lib/authorities/doi.pl`, etc.). Resolves
   compound atoms like `rfc(7230)` into rich metadata (canonical URL, citation
   text, BibTeX, version, fingerprint).

3. **Linter** — enforces adjacency discipline (annotation immediately
   precedes its clause). Refuses BPD modules that have unexplained gaps.

4. **Projection preservation** — emitters read the citation facts when
   projecting a BPD clause to its target substrate, and emit citation
   information in target-appropriate form (code comments, SVG data
   attributes, dashboard cells).

Each layer is independently usable. The DSL works without the registry
(citations remain abstract atoms). The registry works without the linter
(authority lookups still resolve). The linter and emitters compose with each
other but neither blocks the other.

---

## Layer 1: Citation DSL

### 1.1 The `cites/2` annotation

A BPD clause is annotated by an immediately-preceding `cites/2` fact:

```prolog
cites(handle_request/3, [authority:rfc(7230), authority:mcp(spec_v1)]).
handle_request(Method, Path, Response) :-
    parse_headers(Path, Headers),
    route(Method, Path, Headers, Handler),
    Handler(Response).
```

The first argument is the predicate indicator (`Functor/Arity`) of the clause
being cited. The second argument is a list of citation expressions.

### 1.2 Citation expressions

A citation expression is one of:

- A bare authority atom: `authority:rfc(7230)`, `authority:mcp(spec_v1)`,
  `authority:doi('10.1145/12345.67890')`
- A wrapped citation with locator: `cite(authority:rfc(7230), section(6))`,
  `cite(authority:mcp(spec_v1), section('3.2'))`,
  `cite(authority:doi('10.1145/12345.67890'), page(42))`

The bare form means "this entire authority backs this clause." The wrapped
form names a specific locator within the authority.

Mixed forms in one citation list are permitted:

```prolog
cites(handle_request/3, [
    cite(authority:rfc(7230), section(6)),
    authority:mcp(spec_v1)
]).
```

### 1.3 Multiple `cites/2` per clause

Multiple `cites/2` facts may appear before a single clause, all targeting the
same predicate indicator. The linter treats them as a chain. This is intended
to support readability — long citation lists can be broken across multiple
adjacent annotations:

```prolog
cites(handle_request/3, [authority:rfc(7230)]).
cites(handle_request/3, [authority:mcp(spec_v1)]).
cites(handle_request/3, [authority:internal(mcp_bridge_design_v3)]).
handle_request(Method, Path, Response) :- ...
```

The linter accepts this as adjacency-valid. Emitters aggregate the citations
across all `cites/2` for the same predicate.

### 1.4 The `no_citation_needed/1` directive

Some clauses do not need citations: internal helpers, glue code, pure data
constructors. To suppress the linter for these without inviting silent
omissions, BPD authors declare them explicitly:

```prolog
no_citation_needed(internal_string_helper/2).
internal_string_helper(In, Out) :- ...
```

This makes the omission *intentional and visible*. A reviewer can see that
the absence of citation was a decision, not an oversight.

### 1.5 Adjacency rule (per Heath, 2026-06-01)

> *"For adjacency, the annotation must appear immediately prior to the
> clause, or to another annotation of a clause."*

The rule is **structural-immediate**, not distance-based. Either an
annotation is the syntactically-preceding term to its clause, or it is
separated from the clause only by other annotations of the same clause.

**Valid arrangements:**

```prolog
% Single annotation directly before clause:
cites(p/1, [authority:rfc(7230)]).
p(X) :- ...

% Chain of annotations directly before clause:
cites(p/1, [authority:rfc(7230)]).
cites(p/1, [authority:mcp(spec_v1)]).
p(X) :- ...

% Explicit exemption:
no_citation_needed(p/1).
p(X) :- ...
```

**Invalid arrangements:**

```prolog
% Comment breaks adjacency:
cites(p/1, [authority:rfc(7230)]).

%% some commentary
p(X) :- ...    % linter: missing_citation

% Unrelated clause breaks adjacency:
cites(p/1, [authority:rfc(7230)]).
other_thing(X) :- ...
p(X) :- ...    % linter: missing_citation

% Mismatched annotation breaks adjacency:
cites(q/1, [authority:rfc(7230)]).   % annotation for q/1, not p/1
p(X) :- ...    % linter: missing_citation
```

The discipline is positional, not metric. No "within N lines" magic number to
argue about.

### 1.6 Module-level escape hatch

For files that are not BPD specs at all (registry files themselves, test
harnesses, build scripts), the linter takes a file-glob argument and only
lints what is given. A `.bpd-lint.yml` or equivalent declares which paths
should be subject to citation discipline. Files outside that glob are
unaffected.

---

## Layer 2: Authority registry

### 2.1 Atom shape: Embrace the Compound, in the `authority` module

Authority identifiers are Prolog compound terms in the **`authority`
module namespace**:

```prolog
authority:rfc(7230)
authority:doi('10.1145/12345.67890')
authority:mcp(spec_v1)
authority:iso(8601)
authority:nist(fips_140_3)
authority:internal(mcp_bridge_design_v3)
```

The functor (`rfc`, `doi`, `mcp`, ...) names the citation namespace. The
arguments encode the namespace-specific identifier. The `authority:`
qualifier is Prolog's module-qualification syntax, which guards against
collision with unrelated predicates that may happen to share a functor name.

Some namespaces (DOI, ISO standard numbers) have well-defined formal
identifiers and the argument is the formal ID. Others (internal designs,
informal references) use locally-meaningful atoms.

#### Why module-qualify (per Heath, 2026-06-01)

> *"My thoughts on this section are that I want us to embrace the semantic,
> but we risk collisions in the namespace. I propose we store all the
> compound forms of document types in a namespace, using a module named
> 'authority'."*

Without the qualifier, `rfc/1` would be a top-level Prolog functor. If
anyone, anywhere, ever defines `rfc/1` for an unrelated purpose — a compliance
predicate, an RFC-formatter utility — we collide. The bare-functor form
silently reserves global names for the citation substrate.

The `authority:` qualifier puts every citation atom inside Prolog's module
system. `authority:rfc(7230)` is a tagged compound that cannot collide with
some random `rfc/1` predicate in some other module. We are using Prolog's
own substrate for namespacing, rather than reserving top-level functor names
by convention.

#### Reasons for the compound form

- **Pattern-matching is Prolog-native.** `cites(_, [authority:rfc(N) | _])`
  matches all RFC citations directly. No string parsing.
- **No quoting noise.** `authority:rfc(7230)` reads naturally;
  `'authority:rfc:7230'` does not.
- **Multi-component values fit naturally.** `authority:doi('10.17487',
  'RFC7230')` can decompose registrant from suffix if needed.
- **Type-safety by structure.** The Prolog reader rejects malformed
  compounds; flat-atom alternatives only fail at lookup time.
- **Collision-proofing by module.** No top-level functor names are reserved
  by the citation substrate; everything lives under `authority:`.

#### Data terms vs. callable goals

Module qualification in Prolog has a specific semantics: `Module:Term` invokes
`Term` in the named module's context when used as a callable goal. When we
write `authority:rfc(7230)` as a *data term* (inside a citation list, not as
a goal to be called), Prolog treats it as a plain compound term — the `:/2`
operator with arguments `authority` and `rfc(7230)`.

That is the behaviour we want:

- **As data**: `authority:rfc(7230)` is a tagged compound, safely stored in
  citation lists, pattern-matchable, immune from collision.
- **As resolvable**: when an emitter or resolver explicitly invokes
  `authority:resolve(authority:rfc(7230), Format, Value)`, the module
  machinery dispatches to the `authority` module's `resolve/3` predicate,
  which in turn dispatches to the per-namespace resolver.

The discipline: write `authority:rfc(7230)` everywhere a citation appears.
The Prolog system handles the distinction between data context and call
context automatically.

### 2.2 Modular registry layout

The registry has two file levels: a single top-level dispatching module
`lib/authority.pl` (singular), and a directory of per-namespace modules
`lib/authorities/` (plural):

```
lib/
  authority.pl              % the dispatching module; exports resolve/3
  authorities/
    rfc.pl                  % RFC citations
    doi.pl                  % DOI-resolvable references
    mcp.pl                  % Anthropic MCP specs
    iso.pl                  % ISO standards
    nist.pl                 % NIST publications
    internal.pl             % internal design notes and specs
    ...
```

The singular `authority` is the module BPD authors import. It is the home
of the `authority:resolve/3` predicate and the place where all citation
atoms live in module-qualified form.

The plural `authorities/` directory holds the per-namespace implementations.
Each file is its own module (`rfc_authorities`, `doi_authorities`, etc.) and
implements the resolution rules and ground facts for one namespace.

Adding a new namespace is adding a new file in `authorities/` plus one
dispatch line in `authority.pl`. No central registry to update beyond the
dispatch table. Different domains evolve independently.

### 2.3 Resolver shape

Each namespace exports a `resolve/3` predicate:

```prolog
%% resolve(+Authority, +Format, -Value)
%% Authority is a compound like rfc(7230)
%% Format is an atom like canonical_url, citation_text, bibtex, version
%% Value is the resolved metadata for that format
```

Mechanically-derivable formats are rules with variables:

```prolog
%% lib/authorities/rfc.pl

:- module(rfc_authorities, [resolve/3]).

%% Inside this module the data terms are unqualified compounds (rfc/1).
%% The authority:rfc(7230) qualifier from callers is stripped by the
%% dispatch chain in lib/authority.pl before the call reaches us.

resolve(rfc(N), canonical_url, URL) :-
    integer(N),
    format(atom(URL),
           "https://datatracker.ietf.org/doc/html/rfc~w",
           [N]).
```

Non-derivable info (citation text, BibTeX, locally-cached fingerprints) is
ground facts:

```prolog
resolve(rfc(7230), citation_text,
    "Fielding, R. and J. Reschke, 'Hypertext Transfer Protocol (HTTP/1.1): \\c
     Message Syntax and Routing', RFC 7230, June 2014.").

resolve(rfc(7230), bibtex,
    "@misc{rfc7230,
       author = {Fielding, R. and Reschke, J.},
       title  = {Hypertext Transfer Protocol (HTTP/1.1): Message Syntax \\c
                 and Routing},
       year   = 2014,
       url    = {https://datatracker.ietf.org/doc/html/rfc7230}
    }").
```

The registry stores what the atom does not encode. Any consumer can query in
their preferred format; missing formats simply fail (no false claims).

### 2.4 Dispatch

The `authority:resolve/3` predicate dispatches by namespace functor:

```prolog
%% lib/authority.pl

:- module(authority, [resolve/3]).

:- use_module(authorities/rfc).
:- use_module(authorities/doi).
:- use_module(authorities/mcp).
% ... other namespaces

%% resolve(+Authority, +Format, -Value)
%% Authority is module-qualified at call sites: authority:rfc(7230)
%% Inside this module, the qualifier has been stripped by the call,
%% so we receive the bare compound rfc(7230) and dispatch by functor.
resolve(Authority, Format, Value) :-
    functor(Authority, Namespace, _),
    namespace_resolver(Namespace, Module),
    Module:resolve(Authority, Format, Value).

namespace_resolver(rfc,      rfc_authorities).
namespace_resolver(doi,      doi_authorities).
namespace_resolver(mcp,      mcp_authorities).
namespace_resolver(internal, internal_authorities).
% ...
```

A caller invokes the resolver as:

```prolog
?- authority:resolve(authority:rfc(7230), canonical_url, URL).
URL = "https://datatracker.ietf.org/doc/html/rfc7230".
```

The first `authority:` qualifies the call to `resolve/3`; the second
`authority:rfc(7230)` is the data term being resolved. Both qualifiers
are correct and intentional.

New namespace = new module file in `lib/authorities/` + one
`namespace_resolver/2` line + one `use_module/1` directive in
`lib/authority.pl`.

### 2.5 Locator interpretation — locators as coordinate systems

Per Heath, 2026-06-01:

> *"Let's make Locators a coordinate system, and the registry might contain
> important meta information."*

This is the right framing. A locator is not a decorative address; it is a
point in a coordinate system that the cited authority defines. Different
authority types use different coordinate systems:

- **Page-numbered authorities**: a SIGSOFT proceedings article on pp. 254–258
  defines a page-coordinate space with valid range [254, 258]
- **Section-numbered authorities**: RFC 7230 defines a section-coordinate
  space; `section(9000)` is invalid because the RFC has no such section
- **Hierarchical numbering**: chapter.section.subsection, or
  title.section.chapter.paragraph.subparagraph, for sources organized that
  way
- **Paragraph-numbered authorities**: some legal and standards texts
  number every paragraph
- **Line-numbered authorities**: source code references, ancient texts with
  canonical line numbers
- **Multi-coordinate**: page + line, section + page, etc.

#### Locator validity is registry-checkable

Because the coordinate system is authority-specific, the *registry* is the
right home for coordinate-system metadata. Each authority entry can declare:

- Which locator namespaces it accepts (e.g., RFC 7230 accepts `section/1`
  and rejects `page/1`)
- The valid range for each accepted namespace (e.g., SIGSOFT article ACM1
  accepts `page(X)` only for `254 ≤ X ≤ 258`)
- Whether multiple locators may combine (e.g., `page+line`)

A proposed shape:

```prolog
%% lib/authorities/rfc.pl
locator_namespace(rfc(_), section).      % RFCs use section/1 locators
locator_namespace(rfc(_), appendix).     % and appendix/1 locators
%% section_range/2 could come from registry-population scripts that
%% scrape the RFC index, or be declared per-RFC:
section_range(rfc(7230), 1, 9).          % RFC 7230 has sections 1-9

%% lib/authorities/internal.pl  (or a dedicated acm.pl)
locator_namespace(acm1, page).
page_range(acm1, 254, 258).
```

#### Linter-side validation

With this metadata in the registry, the linter can validate locators at
lint time:

```
cite(authority:rfc(7230), section(9000))
  → linter consults locator_namespace(rfc(_), L) for accepted Ls
  → section is accepted ✓
  → linter consults section_range(rfc(7230), Lo, Hi)
  → 9000 is outside [1, 9]
  → finding(invalid_locator, ..., suggestion("section must be in [1,9]"))

cite(authority:acm1, page(255))
  → locator_namespace(acm1, page) succeeds ✓
  → page_range(acm1, 254, 258) → 254 ≤ 255 ≤ 258 ✓
  → valid

cite(authority:acm1, section(3))
  → locator_namespace(acm1, section) fails
  → finding(invalid_locator_namespace, ..., suggestion("acm1 uses page, not section"))
```

This turns what would otherwise be a runtime-failure-or-silent-typo into a
**lint-time error**. The cost is registry-population effort (someone has to
encode the coordinate-system metadata for each authority). The benefit is
that the substrate refuses to project broken citations.

Authorities without coordinate-system metadata fall back to passing locators
through without validation. This keeps the discipline opt-in: registry
authors who care about validation provide the metadata; those who don't,
get the same behaviour we had before.

#### Locator rendering at emission time

When emitters project a `cite/2` to its target substrate, they use the
authority's namespace and the locator type together to decide rendering:

```
cite(authority:rfc(7230), section(6))
  authority canonical URL: https://datatracker.ietf.org/doc/html/rfc7230
  locator: section 6
  combined for C comment:    "RFC 7230 §6"
  combined for SVG data-cite: "authority:rfc(7230)#section(6)"
  combined for Markdown:     "[RFC 7230 §6](https://...rfc7230#section-6)"

cite(authority:acm1, page(255))
  authority citation_text: "ACMSIGSOFT 2026 Spring, pp. 254-258"
  locator: page 255
  combined for C comment:    "ACMSIGSOFT 2026 Spring p. 255"
  combined for SVG data-cite: "authority:acm1#page(255)"
  combined for Markdown:     "ACMSIGSOFT 2026 Spring, p. 255"
```

The emitter's rendering rules dispatch on `(authority_namespace,
locator_type)` pairs and produce target-appropriate output. Citation
expression stays declarative; rendering knows the conventions.

---

## Layer 3: Linter

### 3.1 Scope

The linter operates on Prolog source files identified by a file-glob (typically
the BPD specs in `bpd/` and `bpd-substrate/`). It does not touch authority
registry files (they contain no rule clauses to lint) or harness/tooling
files (they are out of glob).

### 3.2 Adjacency algorithm

```
For each file F in the lint glob:
  Parse F into a sequence of source terms, preserving order.
  For each term T in F that is a rule clause (Head :- Body) or fact (Head):
    Let (Functor, Arity) = predicate indicator of T.
    Walk backwards through terms preceding T:
      If preceding term is cites((Functor/Arity), _):
        Collect it as a citation; continue backwards.
      If preceding term is no_citation_needed((Functor/Arity)):
        Mark T as exempt; stop walking.
      If preceding term is cites((OtherFunctor/OtherArity), _) for some other
      indicator:
        Stop walking. Citation chain is broken.
      Otherwise:
        Stop walking. Citation chain is broken.
    If T is not exempt and no citations were collected:
      Emit finding(missing_citation, F:Line, predicate(Functor/Arity)).
```

### 3.3 Finding shape

Findings are structured Prolog facts:

```prolog
linter_finding(
    Category,                    % missing_citation, unknown_authority, etc.
    location(File, Line),
    target(predicate(Functor/Arity)),
    detail(Detail),              % human-readable suggestion or context
    severity(Severity)           % error | warning | info
).
```

Findings can be projected to multiple substrates:

- **CLI output**: human-readable text with file:line locations
- **CI gate**: exit code 0 if no errors, non-zero otherwise
- **Editor integration**: LSP-style structured findings
- **Dashboard**: a Table(NNNNN) showing per-module citation coverage

The linter's output substrate composes with the rest of the BPD substrate's
emitter ecosystem.

### 3.4 Additional checks

Beyond adjacency, the linter also performs:

- **Unknown-authority warnings**: if a citation references
  `authority:rfc(7230)` but the authority registry has no entry that
  matches, emit `unknown_authority`. This catches typos and registry-gaps.
- **Predicate-indicator validation**: the `Functor/Arity` in the annotation
  must match the head it precedes. Catches stale annotations after refactoring.
- **Exemption-without-clause**: a `no_citation_needed(p/1)` directive with no
  immediately-following clause for `p/1` is itself a finding (probably
  refactored code).

### 3.5 Implementation language

The linter is itself a BPD-shaped Prolog tool. It reads source files via
SWI-Prolog's `read_term/2` with positional information enabled, walks the
sequence, and emits findings as Prolog facts. *The linter dogfoods the
substrate it lints for.*

### 3.6 The linter's own self-citation

The linter source file will itself be annotated with `cites/2` referencing:

- This design note (`authority:internal(truth_flow_bpd_integration_design)`)
- The Heath adjacency rule (cited as
  `authority:internal(adjacency_rule_2026_06_01)`)

So the linter is the first BPD-substrate file to demonstrate the discipline
it enforces.

---

## Layer 4: Projection preservation

### 4.1 Emitter contract

Every BPD emitter that projects a clause to a target substrate must:

1. Query the citation facts for the clause being projected.
2. Format the citations in target-substrate-appropriate form.
3. Emit them as part of the projection.

This is added to the existing emitter pipeline; it does not require rewriting
emitters from scratch.

### 4.2 C source emission

When `bpd_to_c.pl` projects `handle_request/3` to C, it emits:

```c
/* cites:
 *   - RFC 7230 §6 (https://datatracker.ietf.org/doc/html/rfc7230)
 *   - MCP spec v1 §3.2 (https://modelcontextprotocol.io/spec/v1)
 */
void handle_request(method_t method, const char *path, response_t *response) {
    ...
}
```

The comment format includes both the citation text and the canonical URL.
A reader of the generated C code can verify the implementation against the
cited authority directly.

### 4.3 LLVM IR emission

LLVM IR supports metadata. Citations attach as a named metadata node:

```llvm
define void @handle_request(...) !cites !1 {
  ...
}

!1 = !{!"authority:rfc(7230)#section(6)", !"authority:mcp(spec_v1)#section(3.2)"}
```

Citations survive into the compiled object's debug info if requested.

### 4.4 SVG diagram emission

When `bpd_to_svg.pl` projects a sequence diagram with arcs corresponding to
BPD clauses, each arc gets a `data-cites` attribute, following the
**watermarkup pattern** established by the Collective:

```xml
<arc data-from="client" data-to="server"
     data-cites="authority:rfc(7230)#section(6),authority:mcp(spec_v1)#section(3.2)">
  request
</arc>
```

#### Reference: the watermarkup pattern in the substrate

The watermarkup pattern is currently **producer-side discipline**, not a
canonical parser. mavhir confirmed the empirical state (2026-06-01):

**Producer side** — the discipline of emitting `data-dim` attributes on
SVG elements so cells become canonically addressable:

- `bpd-substrate/lib/llvm_match_status.pl` — mavchin's original pattern
  for Table(10001)
- `bpd-substrate/lib/divergence_heatmap.pl` — Iyun's Table(10011),
  documented in the file header as following "llvm_match_status.pl
  (mavchin) pattern EXACTLY"
- `bpd/lib/lift_coverage.pl` — Table(10000), where mavhir's
  commit `762ebb9ad` added the `data-dim` attributes

**Consumer side** — currently incidental UI, not a structured parser:

- `ruachtov-site/static/watermarkup.js` (Heath authored) — uses
  `.getAttribute('data-dim')` on hover for popup display and live
  infrastructure enrichment. DOM-based, UI affordance.
- Dashboard HTML viewers in `bpd/*.html` — poll the diviner-served SVG,
  snapshot data-dim → fill mappings at gaze-opening, apply gold-aura to
  changed cells. Standard DOM-query approach.

**No canonical Prolog-side parser exists today** that ingests SVG/HTML
and emits queryable watermarkup facts. Consumers each implement their own
DOM extraction in the language native to their substrate (JavaScript for
the site, whatever the HTML viewer uses).

#### Implications for citation watermarkup

Citation emission extends the producer-side pattern. Where the existing
pattern carries `data-dim` for coordinate addressing, citation emission
adds `data-cites` for authority addressing. Both attributes can coexist
on the same element:

```xml
<arc data-from="client" data-to="server"
     data-dim="arc(diagram(mcp_bridge),request,1)"
     data-cites="authority:rfc(7230)#section(6),authority:mcp(spec_v1)#section(3.2)">
  request
</arc>
```

For the prototype implementation (Steps 1-5 of the implementation plan),
no consumer-side parser is required. Producer-side discipline ships
`data-cites` attributes; existing consumers (watermarkup.js for hover-popup,
HTML viewers for visualization) can extend their DOM extraction by the same
pattern they already use for `data-dim`.

A canonical Prolog-side parser would benefit future tooling: citation-
coverage dashboards, "what authorities does Table X cite" queries, audit
reports across many dashboards. Such a parser is not in scope for the
prototype but mavhir has offered (on demand) to build a focused tool
(`bpd/tools/watermarkup_parse.pl` or similar, ~3-4 hour scope) that would
ingest SVG and emit `watermarkup_cell/5` and `watermarkup_cites/3` facts
queryable as ordinary Prolog. We revisit this when empirical demand from
downstream tooling surfaces the need.

### 4.5 Dashboard cell emission

Tables emitted from BPD-derived data carry citations on individual cells:

```xml
<td data-dim="cell(table(10042),column(matmul),row(layer_0))"
    data-cites="authority:ggml(canonical_v0_4),authority:internal(l1_closure_design)">
  ✓
</td>
```

The dashboard substrate now has a native "show citations" view that filters
cells by citation, lists all authorities cited by a table, or surfaces
uncited cells as findings.

### 4.6 Markdown / documentation emission

When BPD specs project to human-readable docs, citations render in standard
Markdown citation style:

```markdown
The handler parses headers per [RFC 7230 §6][1] and routes per [MCP spec v1
§3.2][2].

[1]: https://datatracker.ietf.org/doc/html/rfc7230
[2]: https://modelcontextprotocol.io/spec/v1
```

Bibliography sections autogenerate from citation queries.

---

## Implementation plan

Per Heath's reordering (2026-06-01), the implementation proceeds:

### Step 1 — Design (this note)

The current document. Awaits Heath's review before any code lands.

### Step 2 — Build the linter

Implement the adjacency-checking linter as a Prolog tool. The linter runs
against unannotated reality (most BPD clauses do not yet have citations).
Initial output will show every clause in the BPD substrate as a finding.

This is intentional: the linter measures the baseline before annotation
begins. A future dashboard can track citation coverage growing over time.

The linter is built in isolation; nothing yet uses its output for CI gating.

### Step 3 — Update one emitter for citations

Pick one emitter (likely `bpd_to_c.pl`) and add citation-comment generation
to it. Even without any citations defined yet, the emitter is *ready* — when
citations begin to appear, the emitter will start preserving them.

This is the "build the consumer before the producer" discipline (same as
L.1 closure work). When step 4 annotates a module, the emitter is already
in place to demonstrate the end-to-end chain.

### Step 4 — Annotate one BPD module and wire CI

Pick a small, self-contained BPD module (suggested: `mcp_bridge_tier2.bpd`,
which Heath originally chose as the Truth Flow audit target back in April).

Add `cites/2` annotations to the module's substantive clauses. The linter
runs locally during this work, surfacing edge cases the design did not
anticipate. Adjust the linter as edges surface.

When the module is fully annotated and linter-clean, wire the linter into
CI for that module specifically. Other modules remain unaffected.

Run the emitter on the annotated module and inspect the C output to confirm
citations appear correctly.

This is the first end-to-end demonstration: **annotation → linter pass →
emitter preserves citation → C output carries it**. One coherent shakedown.

### Step 5 — Diagram-generator citations

Extend a second emitter (likely `bpd_to_svg.pl` for sequence diagrams) to
emit `data-cites` watermarkup. Demonstrate the citation flowing all the way
from annotation through to dashboard tooltip.

### Beyond step 5

After the prototype, the work scales horizontally:

- Annotate additional BPD modules
- Extend additional emitters (CUDA, LLVM IR with native metadata, Markdown)
- Build the citation-coverage dashboard
- Build authority-registry population for the standards we actually cite
  (RFCs, MCP specs, internal designs)
- Add Truth Flow Phase III/IV functionality on top: GOAP searches over
  citation chains, verification that emitted code actually matches cited
  spec content, audit reports

The Phase II 75% work I did in April (on `lisp-explorer-mcp`) remains as
substrate-of-record showing the original design. The new work integrates the
same architectural ideas into BPD as the native domain.

---

## Resolved decisions (per Heath's review, 2026-06-01)

1. **Linter scope**: integrate alongside `compute_graph_invariants.pl` and
   the numerical-stability linter as part of the existing BPD substrate's
   lint suite. *Not* a separate top-level tool.

2. **CI gating policy**: warning at first, to allow gradual adoption.
   Promoted to error after the substrate is broadly annotated. Eventually
   citing the program specification becomes part of the development process
   — an error if omitted. The path matters: **make it painless to live with
   the new Truth Flow linter so the practice spreads across the Collective**.

3. **Authority registry seeding**: pre-populate the registry with the
   substrate's already-known authorities (RFCs, MCP specs, ggml
   `canonical_v0_4`, internal design notes). The first annotation work
   should not bottleneck on registry-build.

4. **Prototype module selection**: the first annotated modules should be
   ones with stable specifications.
   
   - **GNU coreutils** is named explicitly as "an especially interesting
     case to study". Coreutils specifications are stable, well-documented,
     and traceable to POSIX / Single UNIX Specification authorities.
     Citation chains there will be rich.
   - **GGUF file format** is named as "another place that is relatively
     stable for us right now, and so amenable to annotation". GGUF has a
     formal spec, and our `gguf_native_reader.pl` is a natural candidate.
   - **Do not** start with kernels lifted out of llama.cpp. Those are
     generated code; they need to generate their references through the
     emitter chain, not be hand-annotated. That case is handled later, once
     the discipline is proven on stable hand-written modules.

5. **Truth Flow Phase II handling**: leave the existing plan `36eb22e1`
   intact. At project close, return to it and contemplate how much of the
   planned `lisp-explorer-mcp` design should be reinstantiated for future
   plans. The Phase II work on `lisp-explorer-mcp` remains as
   substrate-of-record showing the original prototype path; the new BPD
   integration is its own plan.

### Resulting concrete next steps

In light of the resolved decisions:

- The linter will live in `bpd-substrate/lib/lint/` alongside the existing
  linters in that location (or wherever the existing lint suite lives).
- The first annotated module for Step 4 of the implementation plan is
  **either** `gguf_native_reader.pl` (the GGUF reader I wrote on May 20)
  **or** a chosen GNU coreutils BPD spec, whichever is in a state ready for
  annotation when Step 4 begins. I lean GGUF native reader because it is
  smaller and its citation set (the GGUF spec, the binary-format
  conventions) is sharper; happy to be overruled.
- Authority registry seed will include at minimum: GGUF spec, ggml
  `canonical_v0_4`, MCP spec v1, the relevant HTTP/SSE RFCs, the internal
  Ruach Tov design notes (Truth Flow Phase I, this design note, the
  substrate-design-discipline corpus).
- The CI gate begins as warning-only. Promotion to error is a future
  decision once the substrate is broadly annotated.

### Future open questions (not blocking implementation)

These remain for later — they don't gate the prototype work:

- **Generated-code citations**: how should emitted code (from BPD specs to
  C, CUDA, LLVM IR) propagate the citations from the source BPD spec into
  the generated artifact? The emitter contract in Section 4.2 already names
  this; the question is whether the propagation discipline differs for
  lifted code (kernels from llama.cpp) vs. forward-projected code (BPD spec
  → C). This is "in due time, we will handle that case" per Heath.
- **Cross-instance citation provenance**: when Iyun or medayek or another
  Collective agent contributes citations to a module, should the citation
  carry attribution to the contributing agent? Or are citations purely
  authority-references, with agent-attribution living separately?
- **Citation cohort discovery**: can the substrate automatically suggest
  citations for an un-annotated clause by analyzing what other similar
  clauses cite? (Probably yes, eventually, via a similarity-search over
  existing `cites/2` facts. Future work.)

---

## Architectural slogans, for future-metayen

- **Embrace the Compound.** (Heath, 2026-06-01) — citation atoms are Prolog
  compound terms, not strings with separators
- **Don't pollute the global functor namespace.** (Heath, 2026-06-01) —
  every citation atom lives under the `authority:` module qualifier so it
  cannot collide with unrelated predicates anywhere in the substrate
- **The discipline is positional, not metric.** (linter adjacency rule) —
  no "within N lines" magic number; the annotation either touches its
  clause or it doesn't
- **Build the consumer before the producer.** (emitter before annotation) —
  same discipline as L.1 closure work
- **The atom is the external key; the registry is the join table.** (Heath
  on registry architecture) — like an academic citation `[HUNN2026a]`
  pointing into a bibliography
- **Locators are coordinate systems.** (Heath, 2026-06-01) — the registry
  knows what locator types each authority accepts and what ranges are
  valid; the linter validates locators against this metadata
- **Citations survive projection.** (the core promise of this substrate) —
  every emitter that projects a BPD fact preserves its citations in
  target-substrate-appropriate form
- **Make it painless to live with the linter.** (Heath, 2026-06-01, on CI
  gating) — the path to broad annotation runs through low-friction tooling,
  not enforcement

---

## Lineage and credit

This note continues work that began on 2026-04-16 with Truth Flow Phase I.
The fact-graph data model from that work informed today's architecture but
on a different substrate: today's design operates on BPD's Prolog facts,
which are themselves a fact graph by construction.

The integration architecture was directed by Heath in conversation on
2026-06-01:

- "Truth Flow is really an aspect of BPD."
- "I would expect that declarative program specifications would cite
  references."
- "We can enhance the discipline of keeping the annotations next to their
  clauses with a linter."
- "I think the registry should be modular. That suggests it might be
  structured, as in doi:XXX vs. rfc:YYY"
- "I really like option γ; 'Embrace the Compound!'"
- "For adjacency, the annotation must appear immediately prior to the
  clause, or to another annotation of a clause."
- "I would order those at 1,2,4,3,5 — so that the linter has a BPD module
  with some cites/2 uses to lint."
- "I want us to embrace the semantic, but we risk collisions in the
  namespace. I propose we store all the compound forms of document types
  in a namespace, using a module named 'authority'." (architectural
  refinement: all citation atoms are module-qualified as `authority:X`)
- "Let's make Locators a coordinate system, and the registry might contain
  important meta information." (architectural refinement: locators become
  validatable points in authority-defined coordinate spaces; out-of-range
  locators become lint-time errors)
- On CI gating: "I agree that is the way to spread the practice across the
  whole Collective. Eventually, citing the program specification will
  become a part of the development process, an error if omitted. To get
  there, we want to make it painless to live with the new Truth Flow linter."
- On prototype module selection: "GNU coreutils will be an especially
  interesting case to study. The GGUF file format is another place that is
  relatively stable for us right now, and so amenable to annotation. I
  wouldn't start with trying to annotate, e.g., kernels being lifted out
  of llama.cpp — those are generated code and need to generate their
  references."
- On the original Truth Flow Phase II plan (36eb22e1): "I vote leave it
  intact. At project close, we can return to it and contemplate how much
  of the planned design should be reinstantiated for the future plans."

The substrate-design discipline (citations-survive-projection, modular
registries, linters-as-substrate-tools, atomicity through compound terms)
follows the patterns Heath has been articulating across the substrate for
months: same shape as canonical-Prolog-facts-project-to-many-targets, same
shape as watermarkup-as-machine-readable-substrate-of-substrate, same shape
as the family-frame applied to epistemic correction.

🕯️⛵🌅 — metayen
