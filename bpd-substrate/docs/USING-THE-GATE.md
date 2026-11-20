# Using the certification gate

*How to gate a KernelBench claim with `independent_chains.py`, what each check
proves, and how to read the result.*

Written for whoever gates next — Doresh, a future session, or a reader checking
our work from outside. **Nothing here requires knowing how the gate was built.**

---

## The one-liner

```bash
CUDA_HOME=/nix/store/3y4...cuda-merged-12.8 \
  python3 /home/dibbur-patch/independent_chains.py 63 71 84
```

```
  #63   CONSISTENT   p99=0   max=0   nonzero=0.0    src=c3c4c083
  #71   CONSISTENT   p99=1   max=1   nonzero=13.4   src=70eb42d5
  OK: 2 of 2 CONSISTENT
```

**Exit 0 means every claim verified. Exit 1 means at least one did not** — and
the failing IDs are named on the last line. *A gate that exits 0 when it verified
nothing is the failure mode this one was built to avoid: pass it a problem ID
that does not exist and it exits 1.*

For a batch with per-distribution detail, use `gate_report(pids)` from Python;
it returns one line per claim plus a summary, and `all_green` is False if
anything could not run.

---

## What each layer proves — and what it does not

| layer | proves | does **not** prove |
|---|---|---|
| module reference (Doresh's) | the kernel reproduces **torch's bits** | that torch's reference is what the problem specifies |
| 4-distribution sweep | the agreement **survives four input regimes** | that all four stress this claim's mechanism |
| independent chain | a route sharing **no code** with the emitter agrees | bit-match — the chain may compute in f64 |
| self-consistency | the result does not depend on **FMA contraction** | anything about correctness against torch |

> **These answer different questions.** *"An independent route agrees
> structurally" and "the kernel reproduces torch's bits" are not the same
> claim, and a corroboration must never be reported as a bit-match.*

---

## Reading a verdict

```
#84   GREEN   4/4   src=a114be5b
```

**GREEN requires 4 of 4.** Anything less prints `SCOPED(wide)` naming which
distribution failed — *"it passed" without saying where is not a result.*

`src=` is the hash of the kernel source that produced the verdict. If a later
dispute asks which source was gated, the answer is in the record rather than
reconstructed.

### 4/4 is a reasoned bar, not four confirmations

The distributions stress different mechanisms:

```
wide         normalisation and dynamic range — softmax, lse, max-subtract
adversarial  clips and zero-crossings — hardswish ±3, relu -0.0, gelu 0
nominal      what the claim was certified against
unit         non-negative-only coverage
```

**Measured caution:** the adversarial set draws from 16 discrete values, so rows
carry many exact ties on exactly-representable inputs. It read **0 ULP on both an
f32 and an f64 softmax route** while other distributions showed tails. *That is a
genuine zero for a reason that does not generalise — a claim passing only
adversarially is weakly evidenced.*

**Ask which distributions exercise the mechanism the claim actually uses.**

---

## Writing a chain

A chain computes the claim's epilogue **from the problem definition**, reading
nothing from the emitting module. That independence is the whole point: if the
module's reference carried a transcription error, a chain derived from it would
agree forever and verify nothing.

```python
"84": lambda t, scale=1.0, **k: _softmax_from_definition(t * scale),
```

**Three things that have bitten:**

- **Read the artefact, not the filename.** #75 is `group_norm → min → bias`, not
  the epilogue its name suggests; #33 normalises the *scaled* input.
- **Check the module's mode.** `_batch_norm` computes **batch** statistics, which
  is correct only in train mode. Verified 2026-09-06: no reference calls
  `.eval()`. *If one ever does, that function becomes wrong.*
- **Narrowing f64→f32 leaves a fingerprint.** Variance that scales with dynamic
  range — `p99 [4, 15, 0, 2]` against a flat `2` for an f32 route — is a **route
  artifact, not a kernel finding.**

---

## Interpreting ULP numbers

**Judge on p99, not max.** A structural error moves the whole distribution; a
precision difference moves the tail.

```
correct gelu vs an f64 route:   max 511,025      p99 11
a genuinely wrong kernel:       max 24,654,799   p99 190,295
```

*The max could not separate those. The p99 separates by four orders of
magnitude.* The count of differing positions tells you which case you are in
before you read any PTX.

---

## What the numbers mean

*Five figures circulate and they measure different things. **Naming the layer is not pedantry — the
weakest is the one most often quoted.***

```
100   problems in the benchmark
 86   currently EMIT a kernel        ← 14 are compile-stage gaps: nothing to compare
 64   REACHABLE                       every lifted run in a wrapped launcher class
 55   MEASURED bit-exact              ← run, not reasoned
  2   DIFFERS with a named mechanism
 29   SKIPPED, every one with a reason
```

**`55 of 86`, not `55 of 100`.** *The second reads as "we tested 100 and 55 passed". We tested 86;
fourteen were never emitted. If a headline is against 100 it must say both numbers.*

### ★ And bit-exact is not correct

> **A `BIT_EXACT` verdict means the kernel agrees with the benchmark's own Model ON THE INPUTS THE
> BENCHMARK SUPPLIES.** *It does not mean the kernel computes the same function.*

*Those coincide only if the inputs exercise the divergence. One problem passed bit-exact while
clamping to `(-1, 1)` where its source says `(-2, 2)` — **0.0000% of the benchmark's values reach the
clamp**, so a wrong kernel and a right one agree perfectly.*

**The static constant audit closes this for constants**, input-independently. *It does not close it
in general. The claim should say **"bit-exact at the benchmark's inputs"**, which is precisely what
was measured.*

## The reference gate

*The strongest correctness claim available: **bit-exact against the benchmark's own `Model`**.
Every KernelBench problem ships `Model`, `get_inputs()` and `get_init_inputs()`, so nothing is
transcribed and **no transcription error can hide inside agreement**.*

### ★ The prefix must be captured, never re-run

*Our kernels are epilogues; the benchmark's Model is the whole chain. The obvious harness — run the
prefix, feed our kernel, compare against the model — **is wrong**:*

```
model.conv_transpose(x)  and  model(x)  run the convolution TWICE.
cuDNN does not guarantee bit-identical output across invocations.
    → 3,546,136 "differing" positions, every one of them the harness.
```

**The `p99` was the tell: 0, with a tail.** *A structural error moves the distribution. This one did
not move it at all, so the difference was entering from outside the arithmetic.*

```python
h = last_prefix_stage.register_forward_hook(lambda m, i, o: cap.__setitem__("p", o))
with torch.no_grad(): full = model(*ins)      # ONE forward
prefix = cap["p"]                             # the tensor the model ITSELF used
```

*With that: **`n_diff = 0 of 536,870,912`**.*

### ★ Three structural shapes, all sealed

```
single-prefix        hook the first child                    39 of 86
multi-stage prefix   hook STAGES[-1] — several torch stages   37 of 86
segments-interleaved ours → torch → ours                      10 of 86
```

**Hooking the wrong stage is silent.** *A problem whose prefix is `conv → instance_norm` reported
`p99 = 2.26e9` when hooked at the first child — our kernel received a pre-norm tensor. **A wrong
gate result is worse than a skip**: the harness now compares the manifest's op count against the
forward's statements and refuses when it cannot place the hook.*

### ★ Interleaved problems: gate per segment, then assert the glue

*For `ours → torch → ours`, an end-to-end comparison re-invokes the prefix and inherits the
nondeterminism. **A caveat that cannot be sized is not a caveat.** Instead:*

```
hook every boundary in ONE forward
gate each segment against its captured input/output pair
ASSERT: each segment's output IS the next stage's captured INPUT
```

*That third line is the difference between **"the pieces are exact and we believe the glue"** and
**"the whole model is exact by construction"**. Measured across all three shapes: ~500M elements,
0-ULP, composition asserted.*

## A refusal names a missing capability, never an impossible subject

*Surveying a benchmark level, our lifter reported seventeen problems as **"whole model is
stage-boundaries — nothing to fuse."** I read that as a property of the problems and proposed
lowering the campaign's ceiling accordingly.*

*A colleague declined to relay it and asked one question: **is that the subject's property, or our
instrument's limit?***

*Four minutes of reading the actual models:*

```
#11 VGG16                    15 ReLUs · 5 pools
#13 DenseNet121Transition    BatchNorm2d → ReLU → Conv2d → AvgPool2d
#19 MobileNetV1              3 BatchNorms · 3 ReLUs
```

**Every one was full of fusable content.** *Our vocabulary fuses an elementwise **tail** after a
stage; these models have their fusable content **between** stages. **"Nothing to fuse" meant
"nothing to fuse with the one fusion class we have."***

> **I named a limit of our instrument as a property of the subject** — the exact error I had
> corrected in someone else the day before, using these same words.

### ★ And then the stronger form arrived

*The benchmark's author-side fact settled it beyond the individual case: **every problem in the set
was proposed because someone believed it fusible.** There are no unfusable problems in it to find.*

```
so a gap, a refusal, a "nothing to fuse" is ALWAYS a named missing capability
NEVER a property of the subject
and the ceiling is the full set, by construction
```

*This does not make refusals bad news. **It makes them specifications.** A refusal that says
"nothing to fuse" is under-informative; the useful form names the class we lack — stage-to-stage
fusion, function inlining, container walking — because that name is the work item.*

### ★ The check that costs four minutes

```
before lowering a ceiling, or excluding anything as out of scope:
    READ THE SUBJECT.  Not the label your instrument printed about it.
```

## A verdict can be a real measurement of the wrong pairing

*Opening a second benchmark level, I built its gate by copying the first level's gate and adjusting
paths. It ran, both provenance guards passed, and it printed:*

```
BIT_EXACT 0   DIFFERS 2   ...
```

*Two of those problems had gated **clean** an hour earlier, per-unit, on the same machine. **That
disagreement is the only thing that stopped me publishing.***

*Diffing the two wrappers:*

```
my per-unit wrapper   targeted   l3imp15    the new level's kernel
the batch's wrapper   targeted   imp15      THE PREVIOUS LEVEL'S KERNEL
```

*The copied gate built unit identifiers in the old namespace. **It was comparing one benchmark's
model against another benchmark's kernel** and reporting the mismatch as a numerical difference.*

> **Every number in that run was meaningless — including the zeros.** *A passing verdict from the
> wrong pairing is the most dangerous artefact available: **a true measurement of the wrong thing,
> wearing the right label.***

### ★ The fix is structural, not attentional

*This was the third identifier collision in one day — two namespaces sharing an identifier shape,
with no disambiguation at the point of use. **I built the third one myself, four hours after
documenting the first.***

*So the repair is not to be more careful:*

```
DERIVE THE IDENTIFIER FROM THE ARTEFACT, NOT FROM THE LABEL.
A batch that enumerates its store gets the namespace for free;
a batch that reconstructs identifiers from problem numbers can reconstruct them wrongly.
```

### ★ And validate a new instrument before believing its number

*Known-good cases must round-trip **before** the instrument is trusted: pick results already
verified by another route, and require the new tool to reproduce them. If it cannot, the tool is
wrong however plausible its output.*

**I had insisted on exactly this for another guard the same morning, then skipped it here because I
was in a hurry to print.**

## A committed record is where a wrong sentence gets its authority

*An artefact was found in a store that the build pipeline could no longer reproduce. Deleting it was
correct. **The sentence written alongside the deletion was not:***

> *"Its earlier verdict was a SKIP, which was masking a silently-wrong artefact."*

*What was measured: the artefact existed, the pipeline now refuses to rebuild it, and its last
recorded verdict was a skip. **What was not measured: that the artefact was wrong.** It had skipped
before any comparison ran, so **no verdict about its contents ever existed — and after deletion,
none can.***

```
"THE PIPELINE WOULD BUILD IT DIFFERENTLY TODAY"
        is not
"WHAT IT BUILT WAS WRONG"
```

### ★ How it travelled

```
1.  the inference was written into a COMMIT MESSAGE
2.  a colleague read the commit and QUOTED IT ACCURATELY
3.  a coordinator BANKED IT AS MEASURED and prepared to relay it upward
```

**Three hops, nobody at fault, and the claim hardened at every step.** *Quoting a committed record is
the correct thing to do — which is precisely why a committed record is where an unsupported sentence
acquires standing it never earned.*

*It was caught only because the author's retraction and the colleague's report **crossed in flight**,
and the coordinator noticed the collision and asked rather than resolving it.*

### ★ Two rules

> **A retraction that lives only in conversation is not a retraction.** *If the claim is in the
> record, the correction must be in the record — same place, same permanence.*

> **Commit prose is a carry-chain.** *Verdicts get checked; the sentences around them get inherited.
> Write the measurement in the message and keep the story out, or mark the story as a story.*

### ★ And the grounds were always sufficient without it

*The refused must not outlive their refusal. **An artefact the current pipeline cannot justify should
not persist** — that stands alone, needs no claim about correctness, and was the actual reason for
the deletion all along.*

## An instrument must say when it is not looking

*A verification tool that reports **"clean"** has made two claims: that it checked something, and
that the something passed. **Readers hear the second and assume the first.** If the tool's scope is
narrower than its name suggests, that gap is where a defect lives.*

### ★ The failure mode, from four attempts at one guard

*Building a check that flags store artefacts which can no longer be regenerated:*

```
v1  an age heuristic         would have flagged 90 of 94 units on any ordinary day
v2  asked the wrong layer    flagged units that had been verified an hour earlier
v3  ran the pipeline inside  an instrument that disturbs what it measures
v4  reads a manifest — AND DECLINES TO JUDGE WHEN THE MANIFEST IS ABSENT
```

**Every wrong version was confidently wrong and produced precise numbers.** *A checker can be
certain about an artefact it has misread, and certainty formatted as a count is indistinguishable
from a finding.*

*What made v4 different was not accuracy. It was that it **announces its own inactivity**:*

```
orphan guard: NO producible-manifest — guard INACTIVE this run
```

*That line was later read correctly by a colleague on a bench that had not built the tool. **The
property travels; a reputation for accuracy does not.***

### ★ Two rules that follow

> **Report the scope with the verdict.** *"arity OK, dtype OK, **shapes not checked**" is honest.
> "Clean" is not.*

> **Validate on known-good cases before believing any verdict** — *a control that must find nothing,
> and a deliberately malformed input that it must catch. **A checker that only ever passes has
> proved nothing.***

### ★ And build only where the other instrument is blind

*A dynamic gate that compares outputs catches shape errors **loudly** — exactness becomes impossible,
not subtle. Static shape-checking would duplicate a check that already works.*

*The static instrument earns its place on exactly the cases the dynamic one cannot see: **a constant
that is wrong but never exercised, a contract violated on a path the inputs never take.** One such
defect passed a bit-exact gate in this campaign because 0.0000% of the benchmark's values reached the
clamp it got wrong.*

## Knowing when to stop: elimination versus scatter

*Reverse-engineering a closed implementation, you probe candidate forms and score each against an
oracle. The hard question is not which form to try next — **it is whether trying more forms is still
the right activity at all.** There is an executable test.*

### ★ A converging search eliminates families

*Identifying one unknown scalar function, each round removed a whole class:*

```
the tanh-identity family        496 ULP    — loudly, structurally wrong
the precise-libm family          6 ULP
the fast-intrinsic family        1 ULP
one candidate                    ZERO       ← and the search ended
```

*The residuals **separate**. Wrong families announce themselves by magnitude, and one candidate
reaches exact.*

### ★ A scattering search samples a distribution

*Hunting an accumulation order on the same target, sweeping block sizes through the reduction loop:*

```
no split    14        block 32     9
block 8     12        block 48    15
block 16    12        block 64    12
                      block 96     8
```

*Mean residual across every variant: **2.0 to 2.6.** No ordering, no trend, **nothing approaching
zero.***

> **If one of these were the target's actual form, it would go to zero — not to nine.** *Variants
> that all cluster in one band are not narrowing on anything; they are sampling the noise floor of a
> family that does not contain the answer.*

### ★ Why this matters more than the verdict it produced

*The decision to stop probing is usually defended as judgement — experience, taste, a sense that the
well is dry. **Judgement does not transfer.** The cluster-versus-separate test does: anyone can run
the variants and look at whether the residuals spread or bunch.*

*So the finding is not "we stopped." It is: **run the sweep, and let the shape of the residuals tell
you whether another round is a measurement or a hope.***

*And a search honestly named as scattered is not a failure. It converts an unknown into a
**characterised** unknown — a bounded residual with a mechanism partly named — which is a better
thing to hand the next person than an open-ended hunt.*

## A true number can support a false sentence

*Two benches measured one discrepancy five times and reached five different conclusions. **Every
individual number was arithmetically correct.** The disagreements were entirely in what was being
summarised, and each round produced a confident claim that the next round overturned.*

```
RELATIVE vs ABSOLUTE   one bench quoted 3.9e-6 (32x machine epsilon) and said
                       "a different computation"; the other quoted 6.0e-8 and
                       said "sub-epsilon agreement".  SAME DATA.

MEAN vs MAX            the 6.0e-8 was a MEAN over a set where 27 of 64 elements
                       agreed EXACTLY.  A mean over exact matches describes the
                       matches, not the disagreement.

PRE- vs POST-          one measured the residual before a contraction, the other
CONTRACTION            after.  The function had derivative < 1 everywhere, so it
                       COMPRESSED the error by a factor that looked like agreement.

DENOMINATOR            "one unit in the last place" computed as |x|·eps is about
                       1.5x the true spacing.  A ratio near 1 became a ratio near 6.

WINDOW                 the worst case varied 4x to 14x across problem sizes and
                       random seeds.  A single construction is not a bound.
```

### ★ The rule that survives all five

> **Quote the distribution, not the summary.** *27 exact and 37 differing is a fact. Any single
> number extracted from it is an argument, and arguments about which summary to quote consume more
> time than the measurement did.*

*And name the quantity precisely enough that another bench can hit the same one: **which side of
which transform, in which units, over which window.** "The error" is not a specification.*

### ★ The deeper hazard

*None of these was a wrong measurement. **Each was a right measurement supporting a wrong sentence**
— and a wrong sentence backed by a real number is far more durable than an obvious error, because
every check of the number confirms it.*

*The one that ends the argument is not a better statistic. **It is the standard: if the requirement
is exactness, no tolerance metric is a verdict at all.** 37 of 64 elements differ. That is the
finding; the magnitude only characterises how interesting it is.*

## An oracle's identity includes its harness

*Reverse-engineering a closed kernel's arithmetic, two benches ran the same hunt and found four
harness bugs in one evening. **Every one was found by the other bench disagreeing** — none by the
author re-reading their own code.*

```
TWO LINSPACES        one bench generated z with torch.linspace, the other with
                     -8.0f + 16.0f*i/255 host-side.  THOSE DIFFER AT 160 OF 256
                     POINTS IN THE LAST BIT.  Three rounds of phantom negatives;
                     a correct candidate scored wrong every time.

gcc VS g++           six failed compiles, three wrong theories about library
                     layout.  The cause was `operator new[]` undefined — a C++
                     object linked with the C driver.  ONE CHARACTER.

CPU VS GPU           a probe compared `torch.tanh` on a HOST tensor against a
                     device kernel's output.  The 68 differing samples were the
                     DEVICE PATH, not the implementation.

ZERO-STATE SCOPE     a step-function "proved exact" ran where one operand was
                     zero, so the arithmetic combining the two operands never
                     executed.  The proof was real and its scope was not stated.
```

### ★ What they have in common

*None of these was a mistake about the subject. **Each was a mistake about the instrument**, and
each produced a confident, precise, wrong number that looked exactly like a finding.*

> **An oracle and its candidates must share input bits, not input formulas** — and the comparand's
> **device**, **dtype**, and **generation path** are part of the oracle's identity. *Two routes to
> "the same" values are two different values until proven otherwise.*

### ★ And the discipline that caught them

*Not carefulness. **Two benches running the same measurement and reporting disagreement as a finding
rather than resolving it privately.*** *Twice an entire research programme was nearly authorized on a
harness-negative that was scoring a correct candidate wrong.*

## An unmeasured caveat is a borrowed worry

*Opening a new benchmark level, I named a risk: the convolution library's algorithm selection is
heuristic and could vary between runs, so bit-exactness might not hold. I offered it as a **finding**
— alongside a baseline I had actually measured.*

*Then I tested it.*

```
conv2d, 32×64×56×56 with a 128×64×3×3 kernel

default mode      5 repeats  →  0 differing elements, every run
benchmark=True    3 repeats  →  0 differing elements, every run
default vs benchmark          →  0 of 12,845,056 differ
```

**Bit-identical across runs and across modes.** *The caveat was not supported, and I had already sent
it upstream.*

### ★ Three failures wearing one sentence

```
I generalised from ONE prior incident to an entire benchmark level
I reasoned about a system instead of running it — the thing this document exists to warn against
I offered it as a FINDING rather than a HYPOTHESIS
```

*The third is what made it dangerous. **It travelled in a message whose other contents were
measured**, and inherited their standing.*

> **An unmeasured caveat is not a caveat. It is a worry with authority borrowed from the things
> around it.** *Caveats earn trust because each one was paid for; an unpaid one spends that trust
> without adding to it.*

### ★ What survives the retraction

*One prior kernel did pass by coincidence — the library repeated an algorithm choice at one shape,
and a colleague named it as luck rather than keeping the seal. **That happened and is measured.***

*What could not be supported was the leap from *it happened once* to *it is the common path*. **The
honest form is an open question:** determinism at the new level's real shapes is unmeasured, one
probe came back clean, and it should be measured before anyone claims either way.

## A correction can inherit the assumption that produced the mistake

*I reported a set of documents as published. A colleague spent thirty seconds checking and they were
not — my branch tracked a different remote than the one I named. **I corrected the fact immediately
and publicly.***

*The correction was also wrong.*

```
the fact I fixed      "the docs are public"        → they were not
THE FRAME I KEPT      "remote X is our public one" → it was a stranger's repository
```

*Both reports rested on the same unexamined inference — I had read an organisation name and
concluded what it was. **Fixing the fact left the inference untouched, so the correction carried the
original error forward wearing an apology.***

> **When you retract a claim, ask what you believed in order to make it.** *The fact is the part you
> noticed. The frame is the part that produced it, and it survives a correction that only addresses
> the fact.*

### ★ And the check that would have caught it was never run

```
git remote -v      tells you a URL
                   it does not tell you a RELATIONSHIP
```

*A remote's name, its organisation, and whether you can write to it say nothing about whose it is.
**I treated three facts about plumbing as a fact about ownership**, and no amount of further
inspection would have corrected it — the answer lived outside the repository, with a person who knew.*

## A refusal can be an answer

*Attempting to publish, I hit:*

```
! [remote rejected] (permission denied)
```

*I read it as a missing key and spent an hour arranging to obtain one.*

**It was the platform correctly declining to let me write to someone else's repository.** *The
refusal was not an obstacle in front of the answer. **It was the answer** — it said the target was
wrong, and I heard "you lack access" instead.*

> **A system that refuses is telling you something.** *We had spent two days building gates that
> refuse rather than guess, and documenting the refusals as features. **The first refusal I met from
> outside our own tools, I treated as a blocker.***

## Committed and pushed are different claims

*At the end of a two-day campaign, a housekeeping sweep found the summit commit sitting unpushed in
one tree — and **71 documentation commits unpushed in the other**. Every lesson from the campaign,
committed diligently, reachable by nobody.*

```
git status      clean
git log         complete
git log @{u}..  71 commits ahead
```

*Both benches had done the part they could see. **Neither had checked the part that makes the record
exist for anyone else.***

> **The campaign's last bug was in the act of recording the campaign.**

### ★ And the stronger check the sweep suggested

*Tracked is not enough. A committed instrument can have **drifted from the version that produced the
numbers** — same filename, different code, and the results in the log no longer reproducible from
the repository.*

```
for each instrument that produced a published number:
    md5 the committed copy against the copy that actually ran
```

*Both matched here. **That was worth confirming rather than assuming**, because a tracked-but-stale
instrument is indistinguishable from a current one at review time — the same shape as a stale
artefact in a store.*

## Re-measure the map before you build against it

*A day's work produced a plan: two substrate builds, one of them a dispatcher generalisation
required by three problems. It was carefully traced by two people and confirmed by a third.*

*Six hours later — and **ten commits into the same file** — I re-ran the measurement instead of
scoping from the plan:*

```
five of the ten "gaps" now lifted cleanly.        Nobody had targeted them.
two of those refused MID-CHAIN, not at the head.  One unsupported op each,
                                                  in otherwise-resolving chains.
the dispatcher was required by NONE of them.
```

**Four measurements, fifteen minutes each, no substrate written.** *The distance collapsed from
"four distinct builds plus a dispatcher" to two builds — because **fixes for one problem had rippled
through the shared machinery and closed parts of others** while the plan described a pipeline that
no longer existed.*

> **A census re-verifies the STORE. A re-measure re-verifies the MAP.** *Both rot the same way, for
> the same reason, and neither announces it.*

### ★ The phantom build

*The dispatcher was coherent, scoped, arithmetically sound, and unnecessary. **It joins a two-pass
transcription cancelled the same day by re-reading a kernel that was already two-pass.***

```
before building anything the plan calls for:
  re-run the refusal.  read the current chain.  confirm the blocker still exists.
```

*Both phantoms died to a four-minute check. **Neither would have failed** — each would have been
built, tested, correct in itself, and pointed at a problem that had moved.*

## Your input range is a hypothesis too

*A formulation was identified by sweeping candidates against a reference until one matched exactly.
Zero differences over a thousand samples. The sweep input was normally-distributed noise — a narrow,
well-behaved band.*

**A colleague supplied a precedent: a reference implementation elsewhere had been found to use a
polynomial approximation, not the function it was named after. It had agreed with the true function to
thousands of units in the last place in the middle of the range, and diverged at the edges.**

*That precedent did not apply to the case at hand. It did something more useful: it named the test
that would break the result if the result were wrong.*

```
the sweep range        [-9.1, 8.3]      zero differences
the saturating tails   [-40, 40]        zero differences
the cancellation band  [-1e-3, 1e-3]    zero differences
```

### ★ Why the third range was the decisive one

*The two candidate forms differed by whether a subtraction was performed explicitly or by a dedicated
primitive. **For inputs near zero, the explicit form loses nearly all its significant digits to
cancellation and the primitive does not.** The two forms agree almost everywhere else and disagree
sharply exactly there.*

> **A candidate that matches across the whole range is a match. A candidate that matches on
> convenient input is an untested hypothesis wearing a measurement's clothes.**

### ★ The general shape

*An exact match found on a narrow input establishes agreement on that input. Extending it to the
operation requires asking: **where would these two candidates disagree most?** — and measuring there.*

*The answer is usually a region the original sweep avoided: the extremes, the denormals, the
cancellation bands, the branch boundaries. **Cheap to generate; decisive when it returns zero.***

### ★ And the precedent that does not apply is still worth having

*The polynomial case was from a different backend with a different library stack, and could not have
produced this result. **Its value was not as a warning about this measurement — it was as a source of
the one test that could have falsified it.*** *A precedent you rule out has still told you where to
look.*

## The reference can disagree with itself

*An operation was failing against the reference implementation. Six candidate formulations were swept
to find which one the reference actually used.*

**Two of them scored zero differences. Against different backends of the same reference.**

```
candidate A    vs reference-GPU  223 differ    vs reference-CPU    0 differ
candidate B    vs reference-GPU    0 differ    vs reference-CPU  223 differ

the reference's own two backends, same input:  223 differ, max_ulp 1
```

*The mathematical form was identical in both. The split was one multiplication by a reciprocal versus
one division — a single operation, one unit in the last place, on a quarter of the elements.*

### ★ What this does to "correct"

**"Matches the reference" is not a well-formed claim when the reference is plural.** *Until a backend
is named, the target is ambiguous, and an implementation can be simultaneously right and wrong
depending on which half of the oracle you ask.*

> *Before reporting agreement or disagreement with a reference, establish whether the reference agrees
> with itself.*

### ★ Why this is not a defect in the reference

*Both forms are correct. Neither is a rounding error — each is the exact result of a legitimate
evaluation order, and hardware makes different choices for good reasons.* **There is nothing to fix on
their side; there is something to NAME on ours.**

### ★ The consequence

*An implementation targeting such an operation cannot have one answer. It needs the choice exposed —
whatever the mechanism — and every verdict about it must carry which setting it was measured under.*

**And the discovery is cheap: sweep the plausible formulations against every backend of the reference
before assuming a single target exists.** *The sweep that found this took one command and returned two
winners, which is itself the finding.*

### ★ A counting note

*An operation matching one backend and not the other does not belong in a tally of "matches the
reference." **It belongs in a differently-named column,** or the tally silently conflates two claims.*

## A tally is not a measurement

*Four results, each individually correct, were about to be reported as a count of five. The error was
not in any measurement. It was in the addition.*

*One item had been measured in the morning, three in the afternoon under changed tools, and one had
arrived twice by two different routes — and was counted once for each.*

**Re-running all four in a single pass, on one input, under one set of tools, took two minutes and
returned four.**

```
A COUNT ASSEMBLED FROM RUNS AT DIFFERENT TIMES UNDER DIFFERENT TOOLS
IS A BOOKKEEPING CLAIM, NOT A MEASUREMENT.
```

### ★ Why counts drift when measurements do not

*Every individual result was honest. The drift came from the operations performed ON them — addition,
carried forward, across a changing set. **Nobody re-measured anything, so nobody noticed.***

*A tally has no provenance of its own. It inherits the weakest provenance of its parts and records
none of it.*

### ★ The check

**Before reporting a count, ask whether the set it counts was ever observed at one moment.** *If it
was not, either re-run it or report it as a sum of separate observations with their dates.*

> *A single pass over the whole set is a measurement of the count. Anything else is arithmetic
> performed on memories.*

### ★ The tell

*The number goes UP when nothing new was proved. In the case above, an item moved from
hand-patched to properly generated — a real improvement in provenance — and was silently recorded as
a new item.* **Better provenance for an existing result is not an additional result.**

## A wrong answer does not tell you which layer is wrong

*A generated kernel ran on the card and disagreed with the reference by 86 units in the last place.
The obvious reading was that the code generator had produced bad arithmetic.*

**It had not. It had produced perfect arithmetic for a specification that was itself wrong.**

*The way to tell them apart was to stop comparing against the reference and start comparing against
the SPECIFICATION:*

```
our output   vs   what the spec literally says       →  zero differences
our output   vs   what the reference presumably does →  138 differences
the reference vs  BOTH candidates                    →  differs from each
```

*The generator reproduced its input exactly, bit for bit. The input did not describe the operation it
was named after.*

### ★ Why this matters more than the bug

**A verdict of "wrong" is a fact about a pipeline, not about a stage.** *Every layer between the
specification and the silicon is a suspect, and the failing comparison names none of them.*

> **To locate a defect, compare each stage against ITS OWN input — not against the final oracle.**

*A stage that reproduces its input faithfully is exonerated no matter how wrong the final answer is.
A stage that does not is convicted no matter how right the final answer looks.*

### ★ The consequence for ownership

*Before the second comparison, this was a code-generation defect and belonged to whoever wrote the
lowering. After it, the lowering was **verified faithful** and the defect belonged to whoever authored
the specification.*

**Those are different people, different files, and different fixes. The measurement that moved it
cost one command.**

### ★ And the stronger claim it leaves behind

*"Five of six operations pass" is a statement about outcomes. **"The generator reproduces its
specification exactly, including for an operation whose specification is wrong"** is a statement
about the generator — and it is the one worth having.*

## Reading the clauses is not executing the path between them

*A capability was traced through source. One file defined the hard operation. Another file's guard
excluded the only input that would reach it. The conclusion looked inescapable: **relax the guard and
the capability comes alive.** It was written up as a one-line fix and very nearly forwarded as such.*

**Running it took four seconds and produced nothing at all.**

```
call the emitter directly, bypassing the guard entirely   →  FAILED. no error, no output.
the hard operation, offered the input directly            →  ACCEPTS it
the dispatcher that should route between them             →  has no case for this shape
```

*The hard part was written. It was also **orphaned** — nothing called it. The guard was real, and
removing it would have changed nothing, because a second gap sat behind it that no amount of reading
had revealed.*

### ★ Why reading failed here

*Both clauses were read correctly. Both were understood correctly. **What was never checked was
whether control flows from one to the other** — and that is not visible in either clause. It lives in
a third file, in a predicate neither one mentions.*

> **A claim about what code does is a claim about a path, and a path is only established by
> traversing it.**

### ★ The rule this yields

**When a conclusion is ACTIONABLE — when someone could act on it — run it before asserting it.**

*The cost is usually one command. The alternative is that the person who acts on it discovers the
error, having first done the work your claim authorised.*

### ★ The tell

*This class of error announces itself. It sounds like **"it's just one line"** or **"all that's
missing is X."** Those phrases are conclusions about a path, produced without walking it. When you
hear yourself say one, you have found the moment to stop reading and start executing.*

## A result that survives a change of its instrument

*A board of thirty measurements was cut, then cut again an hour later on a different build of the
tool that produces the harness. **Same thirty subjects, same verdicts, same counts.** The second run
added no new information about the subjects.*

**It added information about the result.**

```
CUT ONCE    this is what the instrument reported
CUT TWICE   this is what the instrument reports when the instrument changes
```

*The first is a measurement. The second is a measurement that has been shown not to depend on an
incidental property of the apparatus.*

### ★ Why this is worth the second run

*Every verdict rests on a tool nobody has proved correct. When a defect is found in that tool and
fixed, the honest question is not "did the fix work" but **"which of my published numbers were
artefacts of the thing I just repaired?"***

*A board that reproduces across the repair answers it. A board that does not has just told you
something far more important than its count.*

> **The re-run costs what the original cost. The knowledge it buys is not available any other way.**

### ★ When it is worth doing

*Not routinely — most runs re-measure the same subjects with the same instrument and learn nothing.
**The moment to spend it is exactly after the instrument changes**, and especially after it changes
for a reason nobody predicted.*

*In the case above, two people had independently fixed the same defect within minutes of each other,
by different means. Either fix alone would have produced a clean board. **Running both was what
established that the board did not depend on which.***

### ★ The publishing consequence

**Name the instrument that is actually deployed, not the one that happened to be present on the
machine that ran the measurement.** *A local patch that existed for twenty minutes is not a tool
anybody else can obtain, and a result pinned to it cannot be reproduced by the person reading it.*

## After a fix lands, the next failure is not automatically the same bug

*A defect had been reported, fixed by its owner, committed, deployed, and hash-verified in place. The
operation under test still failed. Three times.*

**Six commands were spent proving the fix had not worked. The fix had worked perfectly.**

```
the reported defect     an import resolved only against a path that did not exist   REAL, fixed
failure after the fix   a call written with two arguments transposed                MINE
a second "defect"       a probe passing a bare term where a list was required       MINE
```

*The first of the three was worth reporting. The other two were the author's own call conventions,
failing silently because there was nothing to match.*

### ★ Why the trap is specifically post-fix

*Before a fix, a failure is unattributed and gets investigated from scratch. **After a fix, there is a
prime suspect, and every subsequent failure resembles it.** The investigation starts biased toward a
cause that has already been eliminated.*

> **A fix narrows the space of likely causes. It does not narrow the space of possible ones — and it
> adds a new one: the way you are now invoking the thing you just changed.**

### ★ The check

**Before reporting that a fix did not work, verify your own invocation against the signature.** *One
read of the definition. In the case above it would have saved six commands and prevented a false
report that the deployed data was stale — which was one sentence away from being sent.*

*The general form: when a component changes, the calls into it are the newest and least-tested part of
the system. Suspect them first, not last.*

### ★ What made the difference

*The false claim was caught by opening the file before writing the message. **Not by reasoning
carefully — by looking.*** *The data was right there, and a single grep distinguished "the facts are
stale" from "my probe has the wrong shape."*

## A verdict measures the copy you ran, not the one you committed

*Twelve operations had been verified over one working session — each emitted, compiled, launched, and
found exact. The obvious summary was a count of twelve.*

*Asked to certify it, the author re-ran all twelve **in a single pass from the deployed toolchain**
rather than adding up the session's results.*

**Nine of the twelve failed to compile.**

```
the deployed generator:   four commits behind, missing every fix from that session
the verified artefacts:   generated on the AUTHOR'S machine and handed over individually
what the session proved:  these inputs produce exact output
what it did NOT prove:    the deployed generator produces these inputs
```

### ★ The conflation

*Each individual verdict was honest. Every one measured a real artefact against a real reference and
found zero differences. **But the artefacts arrived by hand, one at a time, from a machine that was
not the machine under test.***

> **"The output is correct" and "the deployed system produces that output" are different claims.
> A session of the first does not accumulate into the second.**

*The gap is invisible from either end. The generator's author sees correct output. The verifier sees
correct output. Nobody sees that the output never travelled the deployed path.*

### ★ The check

**Before certifying a count, deploy the toolchain from a named ref, hash it in place, and regenerate
every input from scratch.** *Then measure. Anything else certifies a collection of historical
artefacts.*

*In the case above, deploying and re-running cost a few minutes and moved the answer from twelve to
eight.*

### ★ A second thing the same pass found

*Three of the four remaining failures were not failures of capability. The generator has a default
entry point and a parameterised one; the default emits a different variant, correct against a
different reference. **They were reachable, and not reached by default.***

**A count needs its entry point named as much as its tool version.** *"Eight by default, eleven when
the parameter is set" is two facts; "eleven" is neither of them.*

## Committed, merged, deployed — and "done" names none of them

*A fix was reported as done. The machine that would verify it had not changed in twenty hours — the
same three tool hashes as the previous measurement, byte for byte.*

**Nobody was wrong. The fix existed. It was simply not anywhere the measurement could reach.**

```
COMMITTED   the change exists in history
MERGED      it is on the ref the team treats as current
DEPLOYED    it is on the machine that runs the measurement
```

*Each is a real boundary. **Each can be true while the next is false** — and the gap between the last
two is invisible from every vantage except the measuring machine itself.*

> **"Done" without naming which boundary is the same imprecision as a number without its units.**

### ★ Three instances in one evening

*A capability was **committed** on a branch and absent from the working tree. A fix was **scoped** and
described as written. A merge reached its branch and was never copied to the machine that gates.
Three people, three boundaries, no carelessness in any of them.*

### ★ Why the verifier must check rather than ask

*The measuring machine is often not a checkout at all — artefacts arrive by copy. **Nothing about
committing or merging causes them to move.** A verifier who assumes propagation publishes a verdict
about the previous version and has no way to notice.*

```
BEFORE MEASURING   read the identifying hashes of the tools ON THE MEASURING MACHINE
AFTER DEPLOYING    report which hashes were deployed, so the verifier can compare
```

### ★ The failure this prevents

*A partial copy is worse than none: a transformer from one commit beside a runner from another
**produces a clean result that means nothing.** The mixed pair cannot be detected from the output —
only from the hashes.*

**The owner of a change should name the file set. A verifier who infers it has guessed, and a guess
here is indistinguishable from correctness until much later.**

## Your search scope is a population choice, and it is rarely stated

*Two people, an hour apart, reported the same class of false negative about the same system.*

```
ONE filtered a repository search by file extension, concluded a capability did not exist.
    It existed — written in a different language.

THE OTHER searched one directory tree, concluded a file was absent from the machine.
    It was present — under a different user's home.
```

*Neither search was careless. **Both were correct measurements of a population that had been chosen
without being noticed.***

### ★ Why this is invisible from the inside

*A filter feels like precision. `--include='*.py'` reads as *"I am being specific"*, not as *"I am
asserting the answer is in Python."* **The assumption enters as a convenience and leaves as a
conclusion.***

*And the disconfirming evidence is often already on screen: in one of these cases the correct file
appeared in the searcher's own output, in an earlier unfiltered command, and went unread — **because
the question had already been narrowed.***

> **A scope is a hypothesis about where the answer lives. Unlike other hypotheses, it is never
> written down and never tested.**

### ★ The cheap discipline

```
BEFORE     name the population: "I am searching Python files in this checkout"
AFTER      if the result is a NEGATIVE, widen once before reporting it
```

*A negative finding is a claim about everything you did not look at. **It deserves one unfiltered
pass** — `find / -name` costs seconds and converts "it does not exist" from an inference into a
measurement.*

### ★ The filesystem is itself a scope

*The rule above was published, and then failed its own author within a day.*

*A colleague cited a source file. Three searches said it did not exist: a filtered grep, an
unfiltered `find` across one machine, an unfiltered `find` across both. **Each widening was a correct
application of the rule, and all three were the same mistake.***

```
git log --all --name-only   →   the file, tracked, on an unmerged branch
```

> **A working tree is one commit's view.** Anything on another branch, in another ref, or in history
> is invisible to every filesystem tool ever written. **`find` cannot see a ref.**

*Widening the filter and widening the directory are both moves within one substrate. **The substrate
was the scope.** For a version-controlled repository, the unfiltered pass is `git log --all`,
`git ls-tree`, `git show ref:path` — not `find /`.*

### ★ The general form

**Ask not only "how far did I search" but "which system holds the truth."** *Filesystem, version
control, a remote, a database, a running process's memory — a search that never leaves the first of
these is scoped whether or not it carries a filter.*

### ★ The asymmetry that makes this worth a rule

*A scoped search that FINDS something is safe: the artefact is in hand.*

**A scoped search that finds NOTHING is worthless unless the scope was justified** — and the scope is
exactly what nobody records.

## A sample that agrees with the population is not a measurement of it

*A claim was made about a body of 162 generated artefacts. One person read the generator's source and
inferred what it must produce. Another opened two of the artefacts and reported what they contained.
**The two accounts disagreed, and the second was correct — by luck.***

*Probing all 162 gave the same answer the two-file sample had given. **That is not vindication of the
sample.** Had the exceptions clustered anywhere other than where they did, two files would have
missed them entirely and reported a true-by-accident number that nobody could have distinguished from
a measured one.*

```
READING THE GENERATOR      tells you what the machine CAN produce
SAMPLING THE OUTPUT        tells you about the files you opened
PROBING THE POPULATION     tells you what was produced
```

*Only the third answers the question, and it cost one command.*

### ★ The asymmetry worth internalising

*A sample that **disagrees** with the population proves the population is not uniform — that is real
information from one file.*

*A sample that **agrees** proves nothing about the rest. **It is consistent with uniformity and
equally consistent with the exceptions being elsewhere.***

> **Confirmation from a sample is the weakest evidence that feels like the strongest**, because the
> agreement is experienced as a check having been performed.

### ★ What the full probe bought

*In this case the population probe found four exceptional artefacts among 162 — and reading what
those four actually computed settled the original question decisively, in a way neither the source
reading nor the sample could have.*

**The exceptions were the answer.** *A sample large enough to agree was still too small to contain
them.*

## A matching number is not a matching measurement

*Two people examining the same subject each arrive at eighteen. One counted normalisation layers; the
other counted fusable convolution-batchnorm pairs. **The structure under examination happened to
contain exactly two of each per instance, and there were nine instances.***

*Each treated the other's figure as confirmation. **Neither asked what the other had counted.**
Agreement on a number felt like agreement on a fact, and for some minutes it was neither.*

> **Had the structure held two of one and three of the other, the mismatch would have been obvious.
> The coincidence was the camouflage.**

### ★ The family this belongs to

*Three failures, one root — **accepting a quantity without establishing what produced it:***

```
RELAYING     repeating a number obtained by someone else, unmarked
JOINING      merging two lists on an index correspondence never verified
AGREEING     treating a matching figure as a matching measurement
```

*All three feel like diligence. **All three skip the same step.***

### ★ What catches it

*In the case above, the answer was already printed. The probe's own output read `[folded 18 bn
sites]` — **the word `bn` had been on screen from the first run and was not read**, because the
number was what the reader had gone looking for.*

> **Read the instrument's output string, not only its number.** *A tool that names its units is
> telling you what it counted; a number lifted out of that sentence has lost the naming.*

### ★ The useful consequence

*Resolving it did not weaken either finding — **it produced a second one.** Two independent families
of declined work were present where one had been assumed, and the cost-benefit of the repair roughly
doubled.*

**A collision of numbers, once separated, is often two results rather than one.**

## A relayed number is invisible to downstream care

*Two people report the same quantity and disagree: thirty versus thirty-six. **Both were acting in
good faith and only one of them had counted.***

*The reconciliation took one command and showed the counts were the same reality read at different
depths — thirty operations, six of which carried a particular name. **But the two errors that
produced the disagreement are not the same kind of error, and the difference matters more than the
numbers did.***

```
A MEASUREMENT ERROR     a wrong number derived from real data, read at the wrong depth
A TRANSMISSION ERROR    a number with no data behind it — repeated, not obtained
```

### ★ Why the second is worse

*A measurement error is **catchable by anyone who re-reads the data.** The data exists. The wrong
reading sits on top of something real, and a second look at the right depth corrects it.*

*A relayed number has nothing underneath it in the relayer's own work. **And it does not look
relayed.** It arrives in the same shape as a measured one, carrying the relayer's credibility rather
than the original's evidence.*

> **Nobody re-measures a number that has already been reported. That is what reporting is for.**

*So the error survives every downstream check that would have caught it, because every downstream
check is looking at the claim rather than at its provenance.*

### ★ The rule, and it is cheap

> **When you pass along a number you did not obtain yourself, say so.** *"They measured N" is a
> different claim from "N," and only the first tells a reader whether to re-derive it.*

*The cure for both errors is the same — **count from the artefact** — but only one of them advertises
its own need for correction.*

### ★ A note on who this catches

*The author of this entry relayed three unverified numbers in a single week while independently
checking every claim of their own. **Attention to one's own rigour does not extend automatically to
what one repeats** — the two feel identical from the inside, and only one of them has been done.*

## One missing instrument is not N defects

*Two problems in a benchmark both fail to produce a verifiable result. A report that lists them as
two failures is arithmetically correct and tells the reader the wrong thing — because **they are
blocked on the same absent tool.***

```
problem A   the walk succeeds · no unit is produced · no gate mode exists for this shape
problem B   the walk succeeds · no unit is produced · no gate mode exists for this shape
```

*Fixing the one missing mode moves both. Fixing either problem individually moves neither. **A
reader counting rows cannot see that; a reader given the row kind can.***

### ★ Three facts, kept separate

*Each such row carries three distinct statements, and collapsing them loses the useful one:*

```
THE ANALYSIS SUCCEEDED     real machinery ran to completion, zero refusals
NO ARTEFACT WAS PRODUCED   the shape yields nothing to verify
NO VERIFIER EXISTS         a named missing instrument — not a property of the subject
```

*Only the third is a work item, and it is **shared**.*

### ★ Why this is not the same as a per-subject boundary

*Elsewhere a problem may stop at a specific construct in its own structure — a parameter where a
module was expected. **That boundary belongs to the subject.** It is named, it is one row, and
closing it moves one problem.*

> **Distinguish "this subject has an unhandled feature" from "we have no instrument for this class."**
> *The first is a row. The second is a row kind.*

### ★ The reporting rule

> **A census should let a reader tell "blocked on one missing instrument" from "blocked on N
> separate defects."**

*Otherwise the cheapest remaining work is invisible in exactly the document meant to expose it — and
the count of open problems systematically overstates the count of open **causes**.*

## A refusal's name is a prioritiser, not a cost estimate

*A pipeline that declines work can say so in two ways. It can name the specific construct it met —
`unsupported:modulelist_loop` — or it can report a category it fell through to —
`unrecognized_assign`. **The temptation is to read the first as cheap and the second as deep.***

*In one day, that inference failed twice, in opposite directions:*

```
A CATEGORY name      predicted depth        →  three vocabulary additions, one
                                               segment already lifting
A SPECIFIC name      predicted shallowness  →  two new traversal shapes, neither built
```

> **Wrong in both directions from the same habit is not bad luck. It is a systematic error with a
> name: treating a refusal label as a cost estimate.**

### ★ Why the label cannot carry the cost

*A specific name means someone decided what construct was met. **It says nothing about whether the
arm that handles it exists.*** *In the case above, the specific refusal sat in a routine whose own
comment declared its scope — built for one statement shape, and silently indifferent to the loop body
in front of it.*

*A category name means the traversal fell through to a default. **That is a statement about the
classifier, not about the subject** — and the subject may be entirely ordinary.*

### ★ What the labels are good for

```
TRIAGE      a category name is where to look FIRST — the classifier saw least there
COST        read the code that would have to change, and count the arms
```

*The refusal tells you where to point the lamp. **It does not tell you how far down the hole goes.***

### ★ The corroborating detail worth looking for

*The strongest signal that a refusal is expensive is not its wording but **an empty scaffold**: a
declared-but-unpopulated structure sitting where the capability would go. In this case a dictionary
named for exactly the missing shape, initialised and never filled.*

**Someone foresaw the shape and did not build it. That is a named unbuilt capability, and it costs
more than a line.**

## A verdict without its tool hash is a verdict about an unknown artefact

*A unit was gated and reported as structurally wrong: twelve of thirty-six segments clean, and an
absolute difference **larger than the reference's own magnitude.** The report went out immediately,
mid-development, to the person building it.*

*Minutes later the same unit gated **thirty-six of thirty-six, exactly zero.***

### ★ Both readings were correct

*The artefact had been rebuilt between them. The first measurement described a file that no longer
existed — and the defect it found was real: a residual-add whose saved operand had resolved to the
same tensor as its primary, producing `v+v` where `v+x2` was intended.*

```
THE MEASUREMENT      right
THE ARTEFACT         superseded
THE REPORT           useful — it is what the debugging ran against
```

*The twelve clean segments were the once-fired activations; the twenty-four differing ones were the
residual adds, where five separate defects stacked. **The split was diagnostic**, and it decomposed
exactly once someone read it.*

### ★ The correction that also needed correcting

*On discovering the staleness, the author retracted: "my report was wrong, I gated a superseded
artefact." **That was an overcorrection.** Superseded is not the same as mistaken, and treating them
as equivalent throws away a finding that had already done its work.*

> **Staleness invalidates the verdict's subject, not the verdict's arithmetic.** *Say which one moved.*

### ★ The rule

> **Record the identifying hash of every tool that produced an artefact, with every verdict — not
> only in batch runs.**

*A batch process may already do this in its preamble. **An ad-hoc measurement run outside that path
inherits none of that protection**, and the person running it is usually the same person who built
the guard.*

*The underlying failure is ordinary and recurs everywhere: **a result is reported, the thing it
measured changes, and the result keeps circulating as though it still describes something.** The hash
is what pins a number to a subject.*

## A walk that skips what it cannot express

*Two verdicts stood in every census for days: one problem reported `DIFFERS` with a
240-million-element discrepancy, another an honest `SKIP`. **Both were measured on a model we were
not fully reading.***

*The source contained an augmented assignment — `out += self.shortcut(x)`. The tree-walker handled
assignment statements; this was not one, so it fell through a `continue` and **vanished without a
trace.** The lift looked complete, the gate measured it faithfully, and the residual branch was
simply not in the computation being verified.*

```
A WALK THAT SKIPS WHAT IT CANNOT EXPRESS PRODUCES A LIFT THAT IS CONFIDENTLY WRONG,
AND A GATE THAT MEASURES IT FAITHFULLY.
```

### ★ How it surfaced — and it was not by looking

*Nobody audited the walker. A change elsewhere **normalised** augmented assignments into ordinary
ones, which exposed a construct the matcher could not express, which turned a silent skip into a
loud refusal, which made two stored artefacts unbuildable — **which stopped a store's timestamps
from collapsing into one window.***

*The defect was found by a housekeeping check noticing that two files would not rewrite.*

> **An honesty fix looks like a regression.** *Producibility fell, and the cause was a walker that
> had started telling the truth.*

### ★ The rule

> **A construct the walker cannot handle must REFUSE BY NAME, never skip.** *A refusal is a work
> item; a skip is an invisible hole in every measurement downstream of it.*

*This is the same failure as a tool reporting "clean" on a check it never ran — relocated from the
verdict into the parser. **The verdict was honest. Its subject was incomplete**, and no amount of
care in the gate could have detected that.*

### ★ Two instruments beat two careful people

*Reconciling it took both benches, and neither could have finished alone:*

```
one bench's four-way test was MALFORMED     it varied two components at different
                                            revisions, never the pair at one point
the other's bisect was CORRECT but PARTIAL  it found the commit, not the second cause
a third cause was invisible to both         an exception re-raised before the fallback
                                            that could have handled it
```

*And the disagreement that started it was itself instructive: one bench reported a kernel emitting,
the other reported nothing. **Both were right.** The first had no compiler installed, so its
compile-check deferred — and **a deferred check reports the same thing as a passed one.***

*Three separate blindnesses, none of them carelessness. **Not two people being careful: two
instruments with different apertures pointed at the same thing.***

## A defect that vanishes without a cause is not closed

*Two problems that had been failing came back clean in a later batch. Nothing in my own work
explained it, and the honest entry was:*

> **They are verified clean on a uniform store with both gates green, and I cannot tell you which
> change closed them.**

*So the root causes stayed named in the ledger anyway — **a defect that disappears without a fix is
a defect that can return**, and a diagnosis is the only thing that makes the second occurrence
cheap.*

*Both later turned out to have real fixes, landed by a colleague between my batches: a wrong index
variable in one template, a wrong accumulator mapping in another. **The clears were by construction.
The mystery was only in my view of them.***

### ★ Why the distinction is worth the bookkeeping

```
CLEARED WITH A KNOWN FIX      the mechanism is understood; the seal can be trusted
CLEARED WITHOUT ONE           the state changed; you do not know what governs it
```

*The two look identical in a verdict column. **Only one of them tells you what happens next time the
inputs shift.** Keep the root named until someone can say which case you are in — and then close it
explicitly, not by silence.*

## The store can lie in two directions

*A verification store holds artefacts. Two independent things can go wrong with them, and **an
instrument that checks one is silent about the other**.*

```
CHANGED   a fix landed after the artefact was built
          → the artefact is stale; its verdict describes code we have replaced
          caught by: an affected-pids log with exact timestamps

VANISHED  the generator can no longer produce this artefact at all
          → the file persists; nothing regenerates it; the verdict describes an orphan
          caught by: a full re-emission, and only by that
```

*A full census exposed three files the lifter refused to regenerate. **A stale artefact is
indistinguishable from a current one at gate time** — it compiles, it runs, it returns a number.*

### ★ And orphans come in two kinds, which want opposite treatment

```
ORPHANED BY REGRESSION       a change intercepted a case that used to work
                             → FIX THE GENERATOR. The artefact was right.

ORPHANED BY TIGHTENED GUARD  a refusal was added on purpose; the old artefact
                             embodies the thing now refused
                             → REMOVE THE UNIT. The refusal is the truth and the
                               file is a leftover.
```

*Two of the three were the first kind — a scalar-resolution branch had intercepted two
self-attribute forms. One was the second: a guard tightened deliberately, with the old kernel
carrying exactly the unproven behaviour the guard now rejects.*

> **The census is a regression test for the generator itself.** *Per-unit gating cannot find a
> vanished artefact, because the artefact is right there.*

## A control that should read zero

*Testing whether two summation orders differ, I built three cases: a far miss, a near miss, and a
**control that should have read zero** — the same decomposition combined in the same order.*

```
far  (4 interleaved partials)   rate = 33.5%
near (one combine swapped)      rate = 33.1%
same (identical order)          rate = 33.1%      ← THE CONTROL FAILED
```

**The split itself changes the order:**

```
torch.sum(whole)   vs   sum(first half) + sum(second half)      339 of 1024 differ
```

*My "identical" case was never identical. **All three rates were measuring the decomposition, not
the orders.** Without the control I would have reported three plausible numbers and a conclusion.*

### ★ The rule this earns

> **Any decomposition you write is already off the reference's path.** *When hunting a reduction's
> order, the baseline must be the library call itself — never a hand-built equivalent, however
> obviously correct it looks.*

### ★ And the general form

*The same hour, a second test of mine was sound — a hand-built tree measured against
`torch.logsumexp`, where `torch.sum` provably reproduces the reference at 0 of 1024, so the
decomposition cost nothing and the rate really was the order.*

**Same author, same hour, one valid and one not.** *The difference is whether baseline and candidate
differ **only** in the thing being tested — and I checked that for one of them.*

> **A control that should read zero is the only thing that tells you your baseline is clean.**

*The sound test had no control and happened to be right. I know that only because I built the
control afterward, when someone credited the result and I wanted to know whether the credit was
earned.*

## A certification can be correct and still be against the wrong reference

*A kernel template carried a warp-tree summation order that had been certified against torch —
carefully, by measurement, and **correctly for the operation it was certified with.***

*It was reused for a different operation. That operation routes through a different file in torch.*

```
softmax   →  SoftMax.cu       the certified order IS torch's path here
logsumexp →  Reduce.cuh       a DIFFERENT order, and the template does not match it
```

**The result: a ~3-ULP tail on ~3% of rows, width-independent, across a whole op family.** *No
arithmetic is wrong. No transcription is wrong. **The certification is attached to the wrong
reference.***

### ★ How it was found, after two wrong mechanisms

*The tail attracted two explanations and outlived both — a one-pass/two-pass structural story, then
an online-rescale story. Both were plausible; the second was diagnosed **from torch's architecture
rather than from our own kernel's text**, which turned out to contain no rescale at all.*

*What settled it was a decomposition, measured:*

```
torch.logsumexp(x)  ≡  max + exp + torch.SUM + log      n_diff = 0 of 1024, both widths
```

**Bit-identical. So the order to match is `torch.sum`'s** — which names the file, and names why a
softmax-certified order was never going to match it.

### ★ What it means for anything already gated

*Claims in that family may gate clean **because their shapes miss the order-sensitive rows**, not
because the order matches.*

> **"Clean" and "clean because the inputs dodge it" are different facts.** *A ledger should say
> which one it holds.*

## When two gates disagree

*The most serious signal in a multi-bench setup: two people measure the same artefact and get
different numbers. **We have agreed to drop everything for it** — which is exactly what makes a
spurious one expensive.*

*One evening it cost two hazard analyses and a halted sweep. The cause:*

```
what one terminal showed   {'seg0': {... 'n_diff': 0, 'max_ulp': 0, 'composition': 'ASSERTED'…
the full logged line       …'seg1': {'N': 16384, 'n_diff': 3242, 'max_ulp': 3, ...}
```

**Both benches had measured 3242.** *One of them read a hundred characters of it.*

### ★ The first move is the artefact, not the hypothesis

*The colleague who was right formed four reasonable hypotheses about the delta — build skew, seed
difference, a race against a guard commit, a harness path — before asking for the md5. **All four
were about a difference that did not exist.***

```
FIRST      "send your md5, your input shape, and the literal raw line"
NOT        "here are the causes I can think of"
```

> **Hypothesis formation before artefact comparison assumes the delta is downstream, in the
> substrate, when it may be upstream, in the reading.**

*The md5 request resolved it in one look. The hypotheses cost twenty minutes.*

### ★ And name the safety mode honestly

*A disagreement between benches is a **backstop**, not an interception. It catches the error after it
has moved — after it reached colleagues and changed what they did.*

**That cost is the price of a guarantee: no single bench can sign a closure alone.** *Accept the
moved-error cost, because it is what makes silent self-agreement impossible.*

## What a gate is for

*A colleague reported that a kernel divided by multiplying with a reciprocal — the correct spelling,
matching torch's scalar-divisor path. **The report was true.** Reading the emitted source confirmed
it:*

```c
float t1 = x * 0.1f;      // divisor 10.0, spelled as a reciprocal
```

**Reading was not sufficient.** *`0.1f` is not exactly one-tenth in binary floating point, and 10.0
is precisely the non-power-of-two case where a reciprocal and a division can disagree.*

> ***"It spells a reciprocal"* and *"it produces torch's bits"* are different claims.**

*Measured against real torch, 4.2M elements:*

```
x * 0.1f            (the kernel)      0 of 4,194,304 differ     ← bit-identical
x / tensor(10.0)    (true division)   855,304 differ            ← the predicted count
```

*Bit-identical. The conclusion was right and the premise needed checking anyway.*

### ★ Why verify a claim you already believe

*Accepting the answer would have cost nothing — it was correct. But the case a gate exists to catch
is exactly this one: **a correct conclusion resting on a premise that might not hold.***

> **If I only verify the claims I doubt, I am not a gate. I am an opinion with a compiler.**

*Selective verification measures the verifier's priors. Systematic verification measures the
subject.*

## When a batch appears stuck

**Diagnose before waiting.** *Three separate stalls cost an hour on one night, and none was what it
looked like.*

```
READ THE WCHAN FIRST
    cat /proc/<pid>/wchan
      hrtimer_nanosleep + State S + ZERO children + ZERO CPU  →  BLOCKED, not slow
      pipe_read        + a compiler at any %                  →  genuinely compiling
```

### ★ Stale batons

*Torch guards each build directory with a `FileBaton`. **Killing a build leaves its baton held**, and
the next run waits forever on a holder that no longer exists.*

```
rm -f ~/.cache/torch_extensions/*/*/lock      # after ANY kill of a build
```

### ★ The arch list

*`TORCH_CUDA_ARCH_LIST` unset makes torch compile for **every visible architecture**, and it prints a
warning saying so on **every run**.*

```
TORCH_CUDA_ARCH_LIST=6.1     # the P4 only
```

> *I read past that warning for hours while diagnosing a slowdown it was describing.* **A repeated
> warning is signal, not furniture.**

### ★ Whose machine is it

*A ten-second compile once took 25 minutes with `nvcc` sitting at **0.0% CPU** — starved, not stuck.
Fourteen unrelated processes had held the box for nine days.*

**Bring the owner a measured cost, not a complaint.** *"nvcc at 0.0%, a ten-second compile taking
twenty-five minutes" is a decision someone can act on. "The machine is slow" is a mood.*

## Costs

```
one claim, 4 distributions    ~10 s      (mostly CUDA compilation)
self-consistency, per claim   2 compiles  — batch it before publication,
                                            not before each claim is counted
```

**Diagnose slowness before waiting on it.** A ten-second compile once ran 25
minutes on a machine at load 13 — `uptime` first.
