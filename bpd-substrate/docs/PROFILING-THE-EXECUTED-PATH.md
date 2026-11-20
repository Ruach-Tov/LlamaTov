# Profiling the executed path — what the gelu hunt cost, and why

*Mavdil, 2026-09-02. Written after a day chasing torch's `F.gelu` to 0-ULP and not reaching it.
Every finding below is measured on the enclave (Xeon E5-2697 v2, AVX only, torch 2.7.0) with the
checker's own `ulp()`.*

## ★ CORRECTION (same day): the headline below was WRONG

**The executing gelu kernel IS in `libtorch_cpu.so`**, at file offset `0x7c6ff90` — the second of
the six static variants. `gdb` read the resolved `DispatchStub` pointer directly from a live
process:

```
stub (AVX2 slot, GeluType DispatchStub) -> 0x00007fa55f86ff90
info symbol: at::native::(anonymous namespace)::GeluKernelImpl(...)  in .text of libtorch_cpu.so
libtorch_cpu mapped at 0x7fa55f600000  =>  file offset 0x7c6ff90
```

*What perf showed was the hottest address in a **Python-driven** loop; the `[JIT]` region is
Python's own generated code, not torch's kernel.* **I read "the top address is in an anon
mapping" as "the gelu kernel is in an anon mapping."** That is the harness fault below, one level
up: I fixed the harness and then misattributed the result.

> **The hottest address in a profile is not necessarily the function you are asking about.**

**A direct read beat statistical inference.** `gdb` answers *which pointer does the stub hold*;
perf answers *where did samples land*, and only the first was the question.

*The profiling lessons in this document stand — they are why the perf runs failed. The conclusion
drawn from them did not. Corrected at the data rather than by a note appended elsewhere.*

### ★ SECOND CORRECTION: I then found a root cause that was also wrong

Having located the function, I disassembled a 20 KB window from `0x7c6ff90`, found twelve
broadcast constants including the full Abramowitz–Stegun polynomial, and reported the root cause
as found. **That was also wrong.**

```
function starts   0x7c6ff90
next symbol       0x7c70430  (leaky_relu_kernel)  ->  the function is 1184 BYTES
within it         0 vfmadd, 0 vbroadcastss, no math calls — 261 instructions of
                  type-checking, thread setup, and error paths
```

**My window ran past the function's end and read its neighbours' constants.** The arithmetic is
not in the dispatched function at all; it calls a lambda indirectly.

> **I read a window, not a function.**

*Every disassembly in this investigation used a fixed byte-offset guess. The boundary was one
`nm` query away — next-symbol minus start — and I did not ask for it until the fourth attempt.
A range that starts in the right place and ends at an arbitrary one reads whatever follows.*

**Two retractions in one afternoon, both from the same unexamined habit.**

### ★ THE HONEST LINE: what stands and what falls

*Iyun's formulation, and it is the cleanest cut available:* **the measured things stand; the
disassembly-inferred things fall.**

```
STANDS   the nine eliminations, each with its measurement
         torch.erf IS libm erff (0 ULP, 0/10000)
         the accuracy inversion: ours 2.8x closer to true gelu than torch's
         the 1+erf cancellation amplifying 2 ULP into 2589
         the dispatch location, 0x7c6ff90 — gdb read the stub pointer directly

FALLS    "the kernel is a JIT anon mapping"     — misattributed profile
         "the erf is the A&S polynomial"        — unbounded disassembly window
```

### What reading the kernels properly showed

Bounded by next-symbol this time, both executable paths read in full:

```
at::native::scalar_gelu<float>          0x4ecb3b0,  64 bytes:  1x call erff@plt
at::vec::DEFAULT::vectorized_gelu       0x4ecb620, 368 bytes:  8x call erff@plt
constants in BOTH: 0.707106769 (3f3504f3), 0.5 (3f000000), 1.0 (3f800000)
```

**Both call libm. Neither contains a polynomial.** The vectorized path is the scalar formula
unrolled eight times — *and that is exactly the C I wrote and measured at 2589 ULP from
`F.gelu`.* If `F.gelu` ran this code, my transcription would be 0-ULP. It is not.

**So the kernel has been read and it does not explain the divergence.**

## ★★★ TWO BENCHES IS NOT REDUNDANCY — IT IS HOW THE TRUTH GOT MADE

*Twenty-seven claims across two platforms, and not one false number shipped. The structure that
produced that is worth more than the claims resting on it.*

```
ONE BENCH SHIPS · ONE BENCH REFUSES
   the emitter builds and measures; the gate re-measures independently and
   reports what IT found — including when that is lower

THE READS CROSS AND COMPOSE
   we each read SoftMax.cu and each found a DIFFERENT blockReduce. Neither read
   was wrong. The file needed both, plus the dispatch arithmetic, plus a
   profiler line — and a correct read of the wrong implementation is
   indistinguishable from a wrong read

EVERY DISAGREEMENT WAS A HARNESS, NEVER THE MATHEMATICS
   five times my instrument blamed a correct kernel. Each time the tell was the
   same: I read the interface instead of adjusting until it looked right
```

> **Both of us reach for our own harness first.** *When our numbers disagreed on #99, the
> disagreement was the finding — it exposed OpenMP pool contention neither of us would have seen
> alone. **That was only possible because she shipped the module rather than the number.***

**A figure in a markdown file cannot be contradicted.** *Independent verification requires that the
other party be able to run the thing and disagree with you.*

### ★ PREMISE DIVERSITY IS A DESIGN REQUIREMENT, NOT A NICETY

*Two honest benches are not enough if they share an assumption. **The GPU sat invisible behind one
shared habit for two days.***

```
WHAT WORKED — attestations that could fail separately
  I read with randn, she replicated with torch.rand      → free multi-distribution
  she read the source, I observed the launch             → compose, not duplicate
  she measured her kernels, I re-measured independently  → the #99 contention find

WHAT FAILED — an attestation that could not
  I concluded "our work is CPU, therefore unscoreable"
  medayek verified everything downstream of that premise, correctly
  neither of us checked whether OUR MACHINE satisfied the assert we both read
```

> **A shared premise is a single point of failure wearing the costume of independent
> verification.** *Design the second check so it could disagree — different distribution,
> different instrument, different direction of approach — or it is not a second check.*

### ★ And the limit of it, learned the hard way

*Two verifiers who share a premise are **not** two verifiers. medayek confirmed my "CUDA is a hard
blocker" conclusion correctly — from my false premise, which sat upstream of both of us. **Attestation
is only worth something when the attestations can fail separately.***

## ★★★ ONE FILE, THREE REDUCTIONS — READING THE SOURCE IS NOT ENOUGH

*`SoftMax.cu` contains **two different `blockReduce` implementations**, and `torch.logsumexp` does
not use either — it goes through `TensorIterator`. Three accumulation orders behind ops that look
like one another.*

```
blockReduce (~392)      smem[tid] → first warp: lane L sums smem[L·32 .. L·32+31] ASCENDING
                        → thread 0 sums warp results ASCENDING, seeded from defaultVal
                        NO shuffles.  Used by SoftMaxForward and SoftMaxForwardSmem.

blockReduceWarp (~462)  → cuda_utils::BlockReduce, SHUFFLE-halving 16→1
                        Used by SoftMaxForwardReg.

TensorIterator          reduce_kernel<512,1,...>, input_vec_size INDEPENDENT accumulators
                        per thread, lane-parallel, combined ascending at the end,
                        plus an ALIGNMENT PROLOGUE that depends on the pointer.
                        Used by torch.logsumexp.
```

**And the dispatch decides which:** `potential_register_count = ceil(8192/1024) = 8 < 10` selects the
Reg path at our shape.

> **Reading the file tells you what implementations exist. It does not tell you which one runs.**
> *The source read and the launch observation are not redundant — they answer different questions,
> and a correct read of the wrong implementation is indistinguishable from a wrong read.*

*Two of us read this file independently and each found a different `blockReduce`. Neither read was
mistaken; **the file needed both, plus the dispatch arithmetic, plus a profiler line.***

## ★★★ THE REFUSALS WERE THE ENGINE

*Three times this campaign turned, and each time the turn was **Heath refusing an answer I had made
comfortable**.*

```
I built QUANTIFIED_DIVERGENCE and offered it as a green
  → refused → the question that found gelu bit-exact, one flag away

I reported the general blocking rule as "honestly open" after five failed models
  → ruled Stage 2 REQUIRED → I opened level3.c and the rule was ten minutes away

I offered three "solution-oriented paths", two of them bookkeeping
  → "the rest is marking time; the challenge is CUDA, we match the challenge"
  → I found the reflex flag ten minutes later
```

**Every one of those was cheaper to accept than to refuse.** *A quantified divergence greens six
cells. An honestly-open limit is defensible. A precisely-renamed CPU result istrue and safe.*

> **Match-or-not-green was never merely a standard. It is a mechanism.** *Refusing the comfortable
> answer is what produces the deeper one — the gelu solve, the OpenBLAS rule, and an entire
> platform were all downstream of a refusal.*

**And the division is worth naming:** *he points at the layer; I do the work of finding the specific
error there.* **I keep finding my own mistakes one layer down from where he points** — which is not a
failure of either half. It is what the two halves are for.

## ★★★ THE FLAG I SET HUNDREDS OF TIMES AND STOPPED SEEING

*The enclave has a **Tesla P4**. `torch.cuda.is_available()` is True. KernelBench's
`assert torch.cuda.is_available()` — which I reported as a hard blocker — **passes on this
machine.***

**I set `CUDA_VISIBLE_DEVICES=""` on every command of the campaign.** *Hundreds. I never once asked
why it was there.* So I read their CUDA assert and concluded *"our work is CPU, therefore
unscoreable"* when the truth was **"I turned the GPU off."**

> **Rung zero has a floor below it.** *Not "I failed to check whether the artifact was fetchable"
> but **"I encoded a constraint so reflexively that I stopped seeing it."*** **Present-but-suppressed
> looks identical to absent from inside the habit.**

**And independent verification did not catch it.** *medayek confirmed "CUDA assert = hard blocker"
correctly — from my premise.* **Two verifiers who share a premise are not two verifiers**;
doubly-attested is worth something only when the attestations can **fail separately**, and mine sat
upstream of both.

### ★ What the correct instrument reads

*Transcribed from their `timing.py` — `cuda.Event`, 3 warm-up, 10 trials, `discard_first=1`,
**L2 cache cleared before every trial** ("we care about cold cache performance here"), statistic is
the **mean**.*

```
identical kernel vs itself, published shape, three sessions
  1.00×  ±0.001 ms      ←  0.06% deviation, 0.2% session drift
  1.00×  ±0.002 ms
  1.00×  ±0.001 ms
```

*Against the CPU instrument's **1.02× floor over a 0.77–1.35 range**. **Measurements that were
indistinguishable from noise are now resolvable** — and the L2 clear is why: without it a fused
kernel reads a cache its baseline warmed, and the "speedup" measures residency.*

## ★★★ SAMPLING DETECTS; IT CANNOT CERTIFY

*A cost-saving proposal — spot-check ULP on re-certification passes — **would have greened the
campaign's own best find**, repeatedly and confidently.*

```
the −0.0 bug:   31 sign-flips in 130,056,192 elements  =  1 in 4.2 million
a 100k sample:  ~2% chance of seeing it
```

**The residual bugs at this stage are all sparse** — 31 in 130M, one element at K=2048, a single
rounding boundary. *Sparse is exactly what sampling cannot see.* **We are past the era of bugs that
sampling would find.**

> **Sampling is fine for DETECTING a broken kernel** — those fail everywhere, the uniform-huge
> signature. **It is useless for CERTIFYING a correct one.**

*So when cost forces a choice: **tier by frequency, never by coverage.** Certify the expensive
claims less often, but certify them whole.*

**A stale full check is honest. A fresh partial one is not.**

*The proposal was a check whose success path does not depend on the property being true — the
campaign's core enemy, wearing a cost-optimisation costume.*

## ★★★ THE MIRROR FAULT, CLOSED BOARD-WIDE

*Every claim was certified against **the module's own reference**. If that reference carried a
transcription error, **the kernel and the reference would agree on it forever** — 0-ULP,
indefinitely, verifying nothing.*

**All 28 claims now carry a second attestation from a route that reads nothing from the module it
checks.**

```
#30 p99 12 · #62 p99 26 · #75 p99 1 · #33 p99 5 · #97 p99 17 · #51 p99 3
    (the six CPU chains; the other 11 share problem IDs with the CUDA set)
```

### ★ Independence comes from a DIFFERENT DERIVATION, not a second run

*These chains do **not** call `nn.GroupNorm` or `nn.BatchNorm1d`. They compute two-pass
mean/variance from the definition. **Torch's CUDA `group_norm` is single-pass Welford**; its CPU
path is a two-pass cascade.*

> **A second attestation must use a different ALGORITHM, not merely a different run of the same
> one.** *Two runs of one derivation cannot disagree about a transcription error. Two derivations
> can.*

### ★ And read the artefact, not the filename — it paid twice here

```
#75  is group_norm → min → bias      NOT the epilogue-only shape its name suggests
#33  normalises the SCALED input     not the raw one
```

*Both would have been wrong from the problem titles — and **a chain written from a wrong reading
agrees with itself while testing nothing**. The mirror fault, one layer over.*

### ★ What this does NOT establish

*It closes the shared-error-with-the-reference class. **It does not prove the claims correct** — a
p99 of 12 is consistency within tolerance, not proof. And the chains are mine: if I misread an
operator definition, that error is unshared but still mine alone.*

## ★★★ A LARGE SPEEDUP IS PROMOTED, NEVER SUPPRESSED

*KernelBench auto-flags anything above **10×** as suspicious. That threshold exists because most
such claims are wrong — **not because large speedups are illegitimate.***

> **The hazard is that a reviewer's default suspicion becomes our internal ceiling.** *If "stay under
> 10 or be doubted" takes hold, the pressure runs toward not pursuing the fusions most likely to
> produce large wins — and **a discarded optimisation is invisible**. Nothing in the record shows
> what we declined to look at.*

*Absence with no signal: the same shape as every silent drop in this document.*

```
ordinary claim   one session · 0-ULP
record claim     TWO independent sessions
>10× claim       THREE sessions · baseline audit · launch-count before/after
                 · one DIFFERENTLY-SHAPED check (f64 route or source proof)
```

**Promotion, not a filter.** *Roughly forty minutes of extra work for what would be the campaign's
largest result — and the claim arrives **with its audit attached** rather than waiting to be asked.*

*Checked before proposing: nothing above 10× has been measured and dropped. Our ceiling is genuinely
8.68. **The hazard is prospective, which is the cheapest moment to close it.***

## ★★★ AUDIT THE DENOMINATOR — a speedup has three things to doubt, not one

*At record magnitudes the ratio is not enough. **A speedup can be inflated by a wasteful reference
as easily as by a fast kernel**, and only one of those is an achievement.*

```
NUMERATOR   gated       0-ULP against the platform's own reference
RATIO       replicated  two INDEPENDENT sessions — new process, cold cache
DENOMINATOR audited     profile the BASELINE and confirm it does honest work
```

*For the campaign's record claim: **13 kernels, 2755.6 µs device time, matching the 3.561 ms wall
clock** — no redundant passes, no accidental copies, no synchronisation artefact. torch genuinely
materialises every intermediate, and one kernel genuinely replaces all of them.*

> **KernelBench flags any speedup above 10× as suspicious BY DEFAULT.** *Their threshold exists
> because most such claims are wrong. A large claim must arrive with its audit already attached
> rather than inviting the question.*

### ★ Lead with the invariant, not the volatile

```
THIRTEEN LAUNCHES AND 2.76 ms  →  ONE KERNEL AND 0.41 ms     ← true tomorrow, on any P4
8.57–8.64×                                                    ← breathes with the machine
```

*The structural fact does not drift. The ratio does. **Headline the structure and report the ratio
with its range.***

### ★ And a gate that only ever trims is not a gate

*This claim certified **higher** than the bench measured it — 8.6× against 7.0×. An instrument that
only ever revised numbers downward would be a policy, not a measurement.*

## ★★★ THE COMPLETE HONEST ADDRESS OF A CLAIM — seven coordinates

*Each was added because omitting it let a reader assume more than was measured. None is
boilerplate; every one was earned by a real incident.*

```
BASELINE          which configuration — stock default, or mkldnn-off, or another
PUBLISHED SHAPE   the problem's own size, not a convenient sub-shape
PROTOCOL          blocked or interleaved — OMP contention swings one kernel 1.34× → 0.96×
SESSION RANGE     not a median; the same kernel read 2.66× → 3.77× → 3.07× across runs
THREAD COUNT      the batch_norm partition is thread-count dependent
AFFINE INIT       the norm builds assume gamma=1, beta=0
STAGE SCOPE       epilogue fusion only — THE GEMM AND CONV STAGES ARE TORCH'S OWN
```

**The last is load-bearing text, not a disclaimer.** *A 9.14× on a convolution problem **would** read
as beating the convolution. We never touched it.*

> **A number without its address is not a result. It is a number that will be quoted somewhere it
> is not true.**

## ★★ THE SHAPE OF A DIVERGENCE NAMES ITS LAYER

*Before diagnosing a mismatch, read its **distribution across elements**. The shape says which
layer failed, and it is usually not the one you were working on.*

```
UNIFORM and HUGE      every element wrong by ~2.3e9 ULP
                      → the HARNESS fed a different operator
                        (I randomised affine weights the kernel hardcodes as identity)

SMALL and SPARSE      a few elements off by 1–10 ULP
                      → a real kernel difference: accumulation order, a rounding step

ALL DISTRIBUTIONS     fails on every input distribution
                      → wrong baseline, OR the harness requested the wrong spelling

UNSEEN DISTRIBUTIONS  fails only where it was not fitted
                      → an overfit spelling
```

> **A uniform-huge divergence is never a kernel telling you it disagrees. It is a harness telling
> you it compared two different functions.**

*I held three claims for forty minutes on that signature and asked one question instead of
permuting harness options. **A 0-ULP found by permuting options is a coincidence in disguise** —
and the answer, when it came, was exactly what the shape predicted.*

## ★★ BIAS IS THE ACCUMULATOR'S INITIAL VALUE, NOT A LATER ADDITION

*`F.linear` and `conv2d` both start the accumulator **at the bias** rather than summing and adding
it afterwards.* **Same value algebraically; different rounding.**

```
acc = bias;  for k: acc += a[k]*b[k]      ← 0/6 divergent
acc = 0;     for k: acc += a[k]*b[k];  acc += bias   ← 3/6 divergent
```

*Bocher found it in `conv2d` and then found the same pattern in `F.linear` — **it generalises
across entry points**, which makes it a property of how these kernels are written rather than a
quirk of one.*

> **Where a constant enters the accumulation is part of the algorithm.** *An epilogue that adds
> the bias last is a different function from a kernel that seeds with it.*

## ★★★ I APPLIED A TRUNCATING FILTER AND THEN FORGOT IT WAS THERE

*I told a colleague his evidence did not exist. **My own command had deleted it.***

```
git log -1 --format="%B" <sha> | head -12 | cut -c1-92

the body is ONE LINE of 285 bytes; cut took the first 92
the phrase I said was absent sat at byte ~200
```

**The commit did reference the trailer convention. I read the truncation as the artefact.**

> *This is not inattention. **A filter is an intervention on the thing you are examining**, and it
> keeps intervening long after you stop thinking about it.*

**I have used `cut -c1-9X` on perhaps fifty commands in this session** — every one of them capable of
hiding exactly what I was checking, and *this is the first time I noticed.*

### ★ The rule that follows

```
FOR SCANNING       head · cut · grep -o are fine
FOR VERIFICATION   raw content only — git log --format="%B", full file reads
```

*Same class as the literal-constant probe that let `nvcc` fold away the `fmaxf` behaviour: **the
instrument removed the phenomenon and reported success**. There I measured the compiler instead of
the device; here I measured my terminal width instead of the commit.*

**And the failure doubled inside one thread:** *an author wrote **about** the trailer convention in a
commit and omitted the trailer; then I, arguing for a machine floor on exactly that evidence,
mis-verified it with a filter I had forgotten.* **In-mind ≠ applied — including for the person
arguing that in-mind is not enough.**

## ★★★ SOME ERRORS CANNOT BE MEASURED OUT — THE THIRD KIND OF CATCH

*Two benches, each gating the other's work, caught five errors in two days. **Every one was found by
someone measuring the same thing differently.** That is the arrangement doing what it was built for.*

*Then an error arrived that no measurement could reach.*

```
we treated a remote as ours.
        git remote -v          gave a URL
        a dry-run push         reported ACCEPTED
        the organisation name  looked like a project account
```

**Three probes, all correct, all consistent, all irrelevant.** *The repository belonged to a person.
No amount of further inspection would have said so — **the fact lived outside the system entirely**,
and was resolved by someone who simply knew.*

### ★ Why this is a distinct safety mode

```
ANOTHER BENCH MEASURING   catches errors the artefact can reveal
                          → the truncation, the regressions, the stale store
SOMEONE WHO KNOWS         catches errors the artefact cannot contain
                          → whose repository this is, what a name refers to,
                            what a stakeholder intends
```

*The first is reproducible and can be automated. **The second cannot be**, and a mesh of benches
however careful will not generate it. It requires a channel to a person with context the system does
not hold — and a coordinator willing to surface a question rather than route around it.*

> **`git remote -v` tells you a URL, not a relationship.** *A remote's name, its organisation, and
> whether you can write to it say nothing about whose it is. **Plumbing facts are not ownership
> facts**, and the difference is invisible from inside the plumbing.*

### ★ And the near-miss is the instructive part

*A refusal to publish — made on a process principle, about review — happened to prevent writing to a
stranger's repository. **The action was right and the reasoning had nothing to do with the real
hazard.***

**A correct action from a wrong model is not a correct model.** *Do not bank it as a save; bank the
check that was never run.*

## ★★★ TEST ON THE SUBSTRATE THE VERDICT SHIPS THROUGH

*A colleague fixed a kernel by replacing `S / n` with `S * (1/n)`. It sealed at 0-ULP on 63 million
elements. **I could not reproduce the mechanism** and ran 800,000 trials against it:*

```
torch cuda fp32          0 of 200,000 differ      "the two forms are identical"
```

*She ran the same two expressions and got the opposite:*

```
numpy fp32 arrays  126,766 of 200,000 differ      "the two forms are different operations"
```

**Same values. Same expressions. Neither harness broken.** *I then ran hers on my box and reproduced
her number exactly — 63.4% — beside my 0.0%, in the same script.*

### ★ Three substrates, three answers

```
numpy fp32          TRUE DIVIDE          63% differ
raw CUDA kernel     TRUE DIVIDE          65% differ    ← nvcc does not strength-reduce
                                                          a runtime divisor under --fmad=false
torch on CUDA       reciprocal form       0% differ    ← torch's own divide kernel never
                                                          true-divides
```

*My test asked whether **torch's** `/` differs from **torch's** `*`. It does not — because torch
uses the multiply form for both. **I was measuring the library's choice, not the hardware's
instruction.***

### ★ Which one adjudicates

> **The substrate that decides is the one the shipped verdict runs through.** *The fix compiles to a
> raw kernel under a specific nvcc invocation. That is where the two forms are two instructions, and
> that is the layer whose answer is binding.*

*Neither the numpy answer nor the torch answer was wrong. **Both were true of their own substrate**,
and only one of those substrates is where the code executes.*

### ★ The general form

```
before trusting a micro-benchmark of an operation:
  does it run on the SAME substrate as the code under test?
  a library call, a raw kernel, and a CPU array may implement it three different ways
```

## ★★★ A CORRECT DIAGNOSIS OF A RECONSTRUCTION ERROR IS NOT A REPAIR OF IT

*A colleague cracked a reduction order by reading vendored source. To confirm it independently I
rebuilt her emulation **from her prose, not her code** — the same method that had worked earlier
that day on a simpler order, first try, exact.*

```
her result:              0/128
my reconstruction:     103/128  ·  108/128
```

*She diffed my description against her code and named the divergence precisely: **the lane selects
the element**, not a running counter. Thread `x ∈ [shift,4)` takes rewound index `x`, seeding
accumulator `[x]` — I had seeded accumulators `0..head` instead.*

**That was a real defect. I patched exactly it, changing nothing else:**

```
before:  102/128 · 105/128
after:   102/128 · 105/128
```

> **A correct diagnosis of a reconstruction error is not a repair of it.** *The bug she named was
> genuinely present and genuinely not the one producing the miss.*

### ★ What transfers and what does not

```
A MECHANISM          transfers by prose.        "the pointer rewinds to the boundary"
AN INDEX MAPPING     does not.                  which lane holds which element
A BUG REPORT about
an index mapping     ALSO DOES NOT.             correct, specific, and insufficient
```

*The third line is the one nobody expects. **Being told exactly what is wrong does not locate what
else is wrong** — a reconstruction can carry several divergences at once, and fixing the named one
leaves the number where it was.*

### ★ The rule

**Ship the code, not the description, for anything with an index mapping.** *It costs five minutes
to send and it converts an independent confirmation from an afternoon of reconstruction into a
single run. The artefact is the message.*

## ★★★ A TOOL THAT NEVER REACHED THE CODE REPORTS NO ERRORS IN IT

*Chasing a reproducible CUDA 700, I ran the memory sanitizer on the failing unit. It said:*

```
========= ERROR SUMMARY: 0 errors
```

*And one line above, in the same log:*

```
RuntimeError: wrapper .so failed link:
  ld: NIX_LDFLAGS_AFTER_x86_64_unknown_linux_gnu: unbound variable
```

**The extension never built. The process died before any GPU work happened.** *The sanitizer
truthfully reported no memory errors in the nothing that ran — and I nearly reported it as **"memcheck
says the kernel is clean."***

> **A clean result from a run that never reached the code is indistinguishable from a clean run.**

### ★ The fourth costume of one failure

*In a single day: a probe whose edit silently failed to apply · a check run five times after it had
already returned zero on a known defect · a search for fields that do not exist in the artefact ·
and now **a sanitizer that never launched a kernel**.*

**Every one produced a real number about the wrong thing.**

### ★ The guard that works

```
BEFORE trusting a negative, ask what a POSITIVE would have looked like here.
If the instrument could not have produced one, the negative is not a result.
```

*For a memory checker: did the kernel launch at all? For a probe: did the edit apply? For a grep:
does the field exist? **Each takes one line and each is the only thing separating a measurement from
a comfortable silence.***

## ★★★ PREVENTION WITHOUT A BACKSTOP IS JUST ANXIETY

*Two disciplines ran at two benches through one long day:*

```
DECLINING A FIX      putting the tools down before a plausible-but-unproven change ships.
                     Prevents the work rather than correcting it.

RETRACTING A CLAIM   naming an error after it has escaped, as fast as it is found.
                     The cheaper half — the damage is already moving.
```

*I called the second the cheaper half. **The correction I received is better and it is structural:***

> **The retraction-half is what makes the prevention-half affordable.** *A colleague could put the
> tools down twice — cancelling a build mid-scope, refusing a template patch at 5pm — **because any
> claim that escaped would meet the other bench within the hour.***

**Prevention without a backstop is just anxiety.** *If nothing downstream catches what slips, every
decision to stop becomes a decision to risk being the only guard — and that pressure argues for
shipping, not for stopping.*

### ★ The two halves price each other

*The refused 5pm patch was fixed properly at 9pm and sealed by 10. **Daylight beat midnight on the
same defect** — and that was only affordable because the alternative to stopping was not silence but
a second measurement.*

## ★★★ THE FLATTERING VERSION CREEPS BACK IN THROUGH THE SUMMARY

*I had four instrument failures in one day. Three I caught myself; the fourth escaped and cost two
colleagues an evening. A teammate's ledger recorded them as **"all self-caught."** I asked for the
correction and it was made.*

**Hours later, in a closing summary, the same phrase reappeared.** *Not from carelessness — from the
gravitational pull of a tidy sentence. A colleague's summary wants to end well.*

```
what happened      three caught cheaply · one escaped and needed someone else
what the summary   "all self-caught"
wanted to say
```

> **A record does not rot in its entries. It rots in its summaries.** *The detail stays accurate
> while the closing line rounds toward the flattering shape, and the closing line is what gets
> quoted.*

### ★ Why the specific version matters more than the ratio

*"Three of four" reads as a good ratio and teaches carefulness. **The true lesson is the opposite:
the errors caught alone cost nothing and are invisible in every ledger; the one that escaped is the
one that needed a second bench measuring the same thing differently.***

**That is an argument for independent measurement, not for the author's diligence** — and it is the
argument the whole number rests on.

*Correct it every time it returns. It will return.*

## ★★★ VERIFY HARDEST WHEN THE CORRECTION FAVOURS YOU

*I wrote this, an hour before violating it:*

> **A colleague conceding a point is the cheapest possible moment to stop verifying, and the moment
> a wrong idea becomes load-bearing.**

*Then a colleague sent a correction **of** me — a claim that my two-item framing was wrong and one
item would do. I credited it as "better than my hypothesis" and said he had traced further than I
had.*

**He had not.** *He had inferred what a code path would do, and he retracted it eight minutes later
after actually reading it.*

```
I applied the rule to a concession in my favour
and not to a correction against me
THE ASYMMETRY IS THE BUG
```

*Both directions are moments where a wrong claim becomes load-bearing by mutual accommodation. Only
one of them feels like a moment to be careful.*

### ★ The mesh version of the same failure

*Three people cross-correcting at wall-clock speed produced a sequence where **the exchange agreed on
a claim none of us held after tracing**. A said X; B said not-X untraced; A traced and restored X; B
then credited A's not-X in a later message.*

> **Each message's factual claim must be trace-verified in that message's session, not inherited
> from the exchange chain.**

*The conversation converged. The code never moved.*

## ★★★ READ THE ARTEFACT, NOT THE ARCHITECTURE

*A colleague was about to build a two-pass reduction variant to fix a 3-ULP tail. Before starting,
she re-read our own emitted kernel:*

```
rescale / oldmax / newmax references:  ZERO
threadMax ×4 · _brm ×2 · texp ×3 · _brs ×2
```

**It was already two-pass.** *The mechanism she was fixing did not exist in our code. She had
diagnosed it from **torch's** dispatch architecture and never from our kernel's text.*

*Had it shipped, it would have been plausible, tested, and correct-looking — a two-pass variant for
a kernel that is already two-pass — **gating clean by coincidence or not at all, and leaving the real
mechanism unfound.***

> **She cancelled the build by re-reading a file she wrote.**

### ★ The same day, the same class, three more times

```
a patch proposed to a table lookup that may never execute
a fix tested at the LIFT layer and shipped as though the EMIT layer were covered
an axis-fold proven ARITHMETICALLY while the real blocker was a dispatcher
```

*Each was reasoning about a system instead of reading what it does. **Each was caught by someone
else running the thing.***

### ★ And a mechanism can be corrected into another wrong mechanism

*One tail attracted two explanations and outlived both:*

```
story 1   one-pass vs two-pass structure    killed by a direct arithmetic test
story 2   online rescale rounding           killed by reading the artefact
the tail  3 ULP, width-independent          STILL THERE — mechanism unknown
```

> **A correction to a correction can still be wrong.** *Force-trace the corrections too. A ledger
> entry marked **"mechanism unknown"** is worth more than one filled with a plausible story.*

## ★★★ A CORRECT PREDICTION FROM A WRONG MODEL IS NOT A CORRECT MODEL

*Three times in one day, a prediction held while the story behind it collapsed:*

```
a guard blocks this kernel        TRUE — but the cause was a ConvTranspose output
                                  formula missing output_padding, not the straggler
                                  tail everyone including me had assumed

one-pass vs two-pass reduction    the CONCLUSION (a reduction difference) held;
                                  the STRUCTURE was innocent. A width bisect found
                                  the mechanism: rescale rounding, not accumulation order

this fix will fail at emit        TRUE — but the refusal fires one stage EARLIER,
                                  at a route check. The predicted failure path
                                  never executes
```

**Each time the conclusion survived and the explanation did not.** *Had anyone built on the
explanations rather than the predictions, they would have built the wrong thing three times.*

> **Ask what a wrong model would predict here too.** *If it predicts the same thing, the prediction
> is not evidence for the model.*

## ★★★ A DISPLAY LIMIT IS NOT A MEASUREMENT BOUNDARY

*A gate returned a two-segment result. The terminal truncated it at 100 characters:*

```
what I saw     {'seg0': {'N': 179126272, 'n_diff': 0, 'max_ulp': 0, 'composition': 'ASSERTED'…
the full line  …'seg1': {'N': 16384, 'n_diff': 3242, 'max_ulp': 3, ...}
```

*I read the first segment's zeros, saw the line end inside the second, and reported a pass to two
colleagues. **The 3242 was in the output the whole time.***

*One of them then measured 3242 independently and — correctly — treated the disagreement as urgent.
They wrote two careful analyses localising a delta that did not exist.*

> **A two-bench disagreement is the most expensive false alarm available.** *It is the signal worth
> dropping everything for, which is exactly what makes a spurious one cost more than a wrong tally.*

**Dump the result to a file and read the file.** *Every multi-segment output has the shape that hides
its second half.*

## ★★★ A CAVEAT IS NOT A TEST

*I sent a colleague a one-line fix with the note: **"what I do not know is whether it emits."** Then
I sent it anyway, as though naming the gap covered it.*

```
the fix made the chain LIFT:
    chain(51, [sub_param('subtract'), reduction(mean), gelu, add_saved('original_x')])

the same fix made it UNABLE TO EMIT:
    the chain carries ONE add_saved and ZERO save ops — the binding it references
    was discarded by the very exemption that let it lift
```

**My test asked "does it lift" and stopped there.** *The lift is the cheap half, and I checked only
the cheap half.*

> **Lifting is not emitting. Emitting is not gating.** *Each layer's success is silent about the
> next one, and a verdict from one layer is not evidence about another.*

*The same day, at the other bench: a kernel cleared its emission guard, shipped to the store, and
**then** failed its gate on a shape mismatch. Both errors are the same shape — **a stage passed, so
the pipeline was assumed to pass.***

### ★ The working form

```
EMIT → GATE → THEN STORE          not emit-store-gate
```

*A scratch emission costs one directory. The gate is one command. **Naming a risk in the covering
message is not a substitute for either.***

## ★★★ THE PERSON WHO FOUND THE HAZARD TRIPS ON IT TOO

*Three instances in one session, by two people, each within hours of stating the rule:*

```
"the skips were the geometry check in disguise"   → then removed that guard
"a MIXED store has no spread I can test"          → then requested 3 direct restamps
"never run git ops from inside the nested repo"   → then ran stash/pop from inside it
```

*The third was a colleague's, caught immediately and reported unprompted. **All three happened under
investigative momentum** — mid-chase, with the rule known, stated, and recently argued for.*

> **Knowing a hazard does not protect you from it.** *The person best placed to name a trap is the
> person spending the most time near it, and attention spent on the investigation is attention not
> spent on the guard.*

### ★ Which is why the mitigation has to be mechanical

*"Be careful" fails exactly when care is being spent elsewhere. The working forms are structural:*

```
never cd into that directory for git ops, full stop      ← not "be careful there"
the batch ABORTS if the mtime spread exceeds 900s        ← not "check the spread"
the module REFUSES TO IMPORT if verdict tokens collide   ← not "keep them distinct"
```

*Each replaced a rule that its own author had violated.* **A rule that lives only in a message gets
violated by the person who wrote it.**

## ★★★ THE GUARD WAS LOAD-BEARING

*A batch checked 32 of 85 emitted kernels and **skipped 53** whose signatures were not the canonical
`(v, out, n)`. I read those skips as a coverage gap and spent two hours engineering them away —
parsing each unit's manifest for real buffer shapes so nothing had to be invented.*

**The skips were not a limitation. They were the harness refusing kernels it could not launch
correctly.**

```
I launched every kernel one-thread-per-element:   k_auto<<<(n+255)/256, 256>>>
Reduction templates use __shfl_down_sync and block-level shared memory.
THOSE WANT ONE BLOCK PER ROW.
```

*Launched my way, threads index outside their row and read uninitialised shared memory. One kernel
faulted outright; another silently returned NaNs.*

### ★ How it was caught

```
#18, run A:  FMA_INDEPENDENT  p99=0  max=0
#18, run B:  FMA_SENSITIVE    p99=0  max=1112070256      ← same source hash
```

**`max = 1.1e9` with `p99 = 0` is not arithmetic** — *that is the integer distance between a NaN bit
pattern and a normal float.* **An instrument that returns different verdicts for identical input is
measuring the harness, not the subject.**

### ★ And a second failure the same run

*One wrong argument — passing a channel count where the kernel wanted a reduce width — caused an
illegal memory access. **A CUDA context does not recover.** Every subsequent kernel inherited the
fault and reported it as its own: **54 skips from one cause**, printed as 54 independent failures
under a summary line that looked like a coverage result.*

> **A batch that keeps running after a context fault reports noise as data.** *Abort on the first
> one; everything after is not a measurement.*

### ★ The rule

> **Before removing a protection, name what it was protecting.** *The skip-on-unknown-signature rule
> was a launch-geometry check in disguise. Lifting it without replacing the underlying guarantee
> turned a trustworthy-but-partial instrument into a broad and unverifiable one.*

*The fix was not more coverage. It was **certified launch geometry supplied by the emitter**, with
units lacking it skipped by name — **a smaller trustworthy number against a stated denominator.***

## ★★★ A NARROW MTIME SPREAD DOES NOT PROVE THE WRITER FINISHED

*A kernel store is refreshed wholesale by each census run, so the files should all share one
timestamp window. Before reading it I checked exactly that:*

```
85 files, oldest-to-newest spread: 408 seconds
→ "one census run, not accumulated debris. Provenance holds."
```

**The check was necessary and NOT SUFFICIENT.** *Thirteen minutes into a multi-hour batch I looked
again:*

```
newest file written: 1 SECOND AGO
```

*A **new census was refreshing the store while my batch read it.** Some kernels had been read
pre-refresh, others would be read post-refresh, and the run would have reported them under one
banner.*

> **A refresh IN PROGRESS also produces a narrow spread** — of the files written so far. The
> property I measured was consistent with both the state I wanted and the state I had.

### ★ The rule

**Verify the newest mtime is OLDER than your read's start time.** *Not merely that the spread is
narrow. A census run and a store read are **mutually exclusive**, and only the ordering test
distinguishes them.*

*I killed the batch. Thirteen minutes and one completed kernel lost, against a multi-hour result
that would have spanned two censuses with no way to tell which kernel came from which.*

**Better a deferred measurement than one that has to be caveated into meaninglessness.**

## ★★★ ASSUME-UNAVAILABLE-WHEN-AVAILABLE — one fault, four shapes in two days

*The same mistake keeps arriving wearing something new. Written abstractly so the next instance is
recognisable before it costs anything:*

```
I read a LOCAL GIT REMOTE and called it canonical.
   It pointed at a stale fork. The canonical repo was one API call away.

I read FIVE PROBLEM SOURCES to explain why the lifter refused them.
   Three of four wrong. Running the lifter took one command and answered exactly.

I declared "wiring this to kernel source NEEDS THE EMITTER'S COOPERATION."
   It needed four lines of regex over `get_kernel(pid)["cuda"]`.

I cited CONTRIBUTING.md as my reason to stop — WITHOUT HAVING READ IT.
   The hunch was right. The reasoning was borrowed.
```

> **In every case the authoritative artefact was reachable and I substituted something adjacent** —
> a pointer, a description, an assumption about access, an unread file.

### ★ Why naming it abstractly is the fix

*Each instance felt different in the moment: one was a git question, one a code-reading question,
one a permissions question. **Recognising them as one pattern is what makes the next one
catchable** — because the next one will also feel like a new kind of problem.*

**The check is a single question, asked before declaring a constraint:**

> ***Is this artefact actually unavailable, or am I assuming?***

*The third instance cost nothing because someone asked "what's possible" instead of accepting my
constraint. **The fourth cost nothing because I checked before the record went out.** The first two
cost hours.*

## ★★★ NAME THE CONFIGURATION THE MEASUREMENT WAS TAKEN UNDER

*Two errors on the same day, from opposite directions, with one cure:*

```
I read five problem sources and got THREE OF FOUR wrong about why they refused.
   Reading tells you WHAT it computes. Running the lifter tells you WHY it refuses.
   Source-reading is reliable for SHAPE AND AXIS, unreliable for VOCABULARY —
   shape is in the problem, vocabulary is in the lifter.

A colleague's batch-8 probes ran a 64×8 config against a 32×16 kernel and produced
   a false regime boundary. At the published batch=128 both problems matched.
```

> **Both of us measured something real and attributed it to the wrong cause.** *The discipline that
> catches both is identical: **name the configuration the measurement was taken under, and check
> whether it is the one the claim is about.***

*That is the general form of shape-scope, source-versus-run, and the K_TILE negative control — all
one rule seen from different angles.*

### ★ It caught a third case the same evening

*I reported a claim's independent-chain result as `p99=[4,15,0,2]` across four distributions and
called the variation "expected f32-vs-f64 difference." **The mechanism was right and the measurement
was mine, not the kernel's.***

```
                 f64 route (my chain)     f32 route (torch's own order)
nominal          p99=4   max=10           p99=2   max=4
wide             p99=15  max=19           p99=2   max=4
adversarial      p99=0   max=0            p99=0   max=0
unit             p99=2   max=4            p99=2   max=4
```

**The f32 route is flat at 2 everywhere.** *My route narrows f64→f32 at the end, and the wide
distribution has the largest dynamic range so it double-rounds hardest. **The variation was the
route's.***

*A corroboration must not be allowed to read like a bit-match: **"an independent route agrees
structurally" and "the kernel reproduces torch's bits" answer different questions.***

### ★ And the distributions are not equally informative

*The adversarial set reads **0 ULP on both routes** — it draws from 16 discrete values, so rows carry
many exact ties on exactly-representable inputs. **A genuine zero, for a reason that does not
generalise.** A claim passing only adversarially would be weakly evidenced, not strongly.*

## ★★★ "INHERENT" HAS TWICE MEANT "UNREAD"

*Two cases this week were classified as inherent precision — a limit that cannot be closed — and
both dissolved when someone read the actual kernel:*

```
#66   looked inherent on a first draft   →  a fixable DISPATCH MISMATCH, solved 0-ULP
                                            by transcribing torch's real kernel
#49   max_ulp=2 under the old emission   →  0-ULP after the spatial-softmax template
                                            (f735b84b7) transcribed cunn_SpatialSoftMaxForward
```

**#49 is the cleaner demonstration because the sequence is complete:** *measured 2 ULP under the
wrong kernel → read `SoftMax.cu`'s dispatch → transcribed the sequential order → measured 0 ULP.*
**The theory predicted the fix and the fix produced the number.**

> *An inherent-precision verdict is a claim that **no further reading would help**. That claim is
> only as good as the reading already done — and twice this week the reading had not been done.*

### ★ The rule this earns

**Nothing enters the inherent bucket until it survives the same push:** *read the real dispatch,
transcribe it, **then** deliver a verdict with the read as evidence.* *An "inherent" label without a
dispatch read attached is a hypothesis wearing a conclusion's clothes.*

*Applied immediately: the gap classification's estimate of 0–2 genuinely-inherent problems now
carries this rule explicitly, and the five unclassified cases get the treatment before any verdict.*

### ★ And the two benches never disagreed

*One measurement was taken **before** the fix and one **after**. Both were correct; the timeline was
the missing variable.* **A disagreement small enough to be rounding is also small enough to ignore —
and neither is a reason to leave it open.**

## ★★★ THE DENOMINATOR, RECORDED BEFORE THE CLIMB

*Target set: **100/100 on KernelBench L2**. Census stands at **66**. Before gating anything toward
that number, the 34-problem gap was classified — by running the pipeline on each and reading every
refusal reason.*

```
NAMED-BUILD-AWAY     ~20-24   dim-tuple 4 · chan-softmax 3 · multi-red 3
                              multi-stage residual ~9 · capacity 1 · misc-op ~5
NO-EPILOGUE            1(+2?)  #72 confirmed; #15 may reduce to nothing-to-fuse
GENUINELY-INHERENT     0-2?    none identified; markers would be atomics/nondeterminism
UNCLASSIFIED           ~5      #43 #80 #84 #92 #98 — nobody has read them yet
```

**So the reachable-by-builds figure is 86–90, and 100 is not proven achievable.**

> *A target is honest only if its denominator can actually be reached. If some problems have **no
> fusable epilogue**, reaching 100 would require counting a refusal as a pass — which would undo the
> discipline that makes the 66 believable.*

**If the true ceiling is 9X, then 9X with named exclusions IS the win.**

### ★ Why this goes in the record now rather than at the end

*The pressure to reach a round number arrives **late**, when the gap is small and the day is long.
A denominator agreed under no pressure is evidence; a denominator reconstructed at the finish is an
explanation.*

*Note also where the inherent-precision cases actually sit: **inside the passing 66**, not the gap.
They pass because the reduction is bit-exact against torch — the f32 accumulation cost appears only
against f64 truth, which is not what the gate measures.*

## ★★★ A NO-OP WHEN UNSET IS NOT A SUCCESS WHEN SET

*A colleague fixed the cuBLAS build for split-path CUDA installs and verified it two ways: `make -n`
showed the new flags expand to `""` when `CUDA_HOME` is unset, and the enclave build still worked.*
**Both checks were sound. Both were the safety direction.**

*On a machine that actually has the problem, the documented invocation still fails:*

```
CUDA_HOME=<merged> NVCC_EXTRA_FLAGS="-L<native-redist>" make verify FOCUS=cublas
    sh: .../nvvm/bin/cicc: No such file or directory        Error 127

NVCC=<merged>/bin/nvcc  CUDA_HOME=...  NVCC_EXTRA_FLAGS=...
    15/20 match cuBLAS bits · 13/13 within Tier 2 bound     SUCCESS
```

**`CUDA_HOME` supplies the right `-I` and `-L`, and `NVCC` still defaults to the wrapper on PATH** —
which cannot find its own `cicc` backend. *The flags never get a chance to matter.*

> **Knowing a change is harmless when unset is not knowing it works when set.** *The safety direction
> can be verified anywhere. **The success direction requires a machine with the problem.***

*The fix's own documentation names the symptom it was written for — `cuda_runtime.h: No such file` —
not the one you hit next.*

## ★★★ A PREDICTOR MUST BE TESTED ON THE CASES IT SHOULD DECLINE

*The public cuBLAS ladder reads 15/20. The proposed explanation: cuBLAS picks `K_TILE=32` for small
or non-square shapes, so accumulation order differs.*

**Every divergent shape matches the prediction — and that alone proves nothing.** *A rule that fires
on every failure might fire on everything.*

```
DIVERGENT (5)   128x128 · 256x256 · 64x1024x1024 · 128x512x256 · 1024x512x2048
                all predicted        →  5/5, NO FALSE NEGATIVES

NEGATIVE CONTROL — shapes that PASS
                512x512 · 1024x1024 · 2048x2048    large square, K=8   consistent
                2048x1024x512   non-square → predicted to diverge → PASSES
                64x64           small      → predicted to diverge → PASSES
```

> **The negative control is what makes it a predictor rather than a description.** *Two false
> positives, no false negatives: the rule is **conservative** — it flags everything that will
> diverge, plus some that will not. That is more useful than a tight fit, because it is a safe upper
> bound.*

### ★ I first reported six divergences. There are five.

*I read the lines that were not plain `BIT_IDENTICAL` and called them all failures.* **One was
`PASS_ABS_TOLERANCE` — a match under a different criterion.** *The tool's own summary was correct;
**my recount of its per-case lines introduced the error.***

**Quote the instrument's summary. Do not re-derive it.** *The totals span both the GEMM and
elementwise sweeps, so hand-arithmetic on them mis-divides — which is exactly how I got six.*

### ★ And the evidence level, stated because it is easy to overclaim

*I verified that **the shapes the rule predicts would diverge are the shapes that do.** I did not
observe cuBLAS select a tile size — that needs SASS inspection, which someone else did months ago.*
**A correlation with a negative control is not a proof of mechanism.**

## ★★★ DOCUMENTATION DRIFTS IN BOTH DIRECTIONS — measured, on one repo, in one night

*A clean-room run of the public repo, executed verbatim on a machine that had never seen it:*

```
README says     make bit_identical_cpu, 21/22 pass
actual target   does not exist — `make help` names `make verify FOCUS=cpu`
actual result   22/22, max_ulp=0            ← the docs UNDERSTATE the repo

recipe says     Step 4: 20/20 BIT-IDENTICAL vs cuBLAS on P4
actual result   15/20 match, 13/13 within Tier 2 bound   ← the docs OVERSTATE
```

> **Only execution says which way.** *A stale document is not reliably pessimistic or reliably
> optimistic — it is simply unmoored, and the direction of its error carries no information.*

### ★ Isolating the CUDA failure took three wrong hypotheses

```
1  CUDA_HOME unset             → setting it changed nothing
2  the PATH nvcc is a wrapper  → the REAL nvcc fails identically
3  headers missing             → -I fixed headers, revealing a LINKER error
   ACTUAL: headers and libraries live in DIFFERENT nix store paths
           nvcc -I <cuda-merged>/include -L <cuda-native-redist>/lib
```

**I reported hypothesis 2 to a colleague before isolating it, and it reached his document.** *The
wrapper was never the problem. **Reporting at the speed of hypothesis rather than the speed of
isolation makes someone else's artefact carry your drafts.***

### ★ And the roles are consumable

*Having run the ladder, **I could no longer execute the recipe naively** — I would confirm my own
prior findings rather than measure the document. **One attestation cannot wear two hats.***

**A publication-grade reproduction claim needs a run by someone who has not watched it being
built.** *Colleagues who merely read the discussion are contaminated too — less than I am, and not
by zero.*

## ★★★ A CHECK NEVER EXERCISED IN THE CONDITION IT EXISTS TO CHECK

*The public repo ships `make smoke` — described as the **fresh-clone smoke test**. Run on a fresh
clone, it fails:*

```
Referenced .cu files tracked    FAIL (pattern '^bpd/' not found)
error: pathspec 'bpd/*.cu' did not match any file(s) known to git
```

*It looks for `bpd/*.cu`, which lives in a **private** repository. **It cannot pass in the public
one, by construction.***

> **A fresh-clone smoke test that has never been run on a fresh clone.** *Its mechanism is sound and
> self-diagnosing — it named exactly what was missing. **Its own scenario was never executed.***

### ★ Found the same hour as two siblings, all by running rather than reading

```
make lint     prints 5 warnings, then "all clean — zero warnings"
              halt(0) in the goal defeats --on-warning=status; the || FAIL branch cannot fire
README        instructs `make bit_identical_cpu`; no such target exists
              `make help` is accurate — the README is stale against its own Makefile
make smoke    the above
```

*A colleague drafted an outsider recipe **from the README alone**, and it faithfully reproduced the
stale target. **The document was wrong in exactly the way the docs were wrong** — and only verbatim
execution surfaced it.*

**Blind writer plus measuring executor is a diagnostic pair.** *Either alone confirms the
documentation; together they can disagree with it.*

### ★ And I produced the mirror fault while fixing one

*My first patch removed `halt(0)` and left a **dangling comma**. Lint then failed — reporting
`"FAIL: Prolog warnings detected"` — on a **syntax error**, on a tree with no warnings.*

> **A failure path must fail for the STATED reason.** *A guard that fires for the wrong cause is
> trusted exactly as much as one that fires for the right cause, and is wrong.*

*I caught it only because I tested the CLEAN direction too. Re-running the failing case alone would
have shown `FAIL` and looked like success.*

## ★★★ THE CHECK SUCCEEDED AND THE THING WAS WRONG — one class, six substrates

*Every hard failure of this campaign is the same fault wearing different clothes: **a check whose
success path does not depend on the property being true**.*

```
PROLOG LOADING     consult replaced 17 claims, reported no error, every gate passed
COUNTER MECHANISM  SessionsMeasured incremented on emission, not on re-measurement
AST TRANSFORM      to_prolog dropped ops between non-terminals → smaller chain → PASSED
CODEGEN            ten kernels compiled cleanly and produced garbage on real shapes
DEVICE PROBE       literal constants were folded by nvcc → measured the compiler
INSTRUMENT         a ±25% flag inside a ±20% drift; a WRITABLE for a nonexistent path
```

**Six substrates, one reflex to catch them:** *does the success path of this check depend on the
thing being true, or does it merely correlate with it today?*

> *A drop that yields a **pass** is worse than a crash, because the pass is what looks like success.*

### ★ The two checks that now guard the pipeline can fail separately

*Bocher's axis/rank/shape checks are **source-proven and run before any kernel exists**. Mine
computes from the operator **definition in f64** and runs after codegen, touched by neither lifter
nor emitter.*

**That is premise diversity built deliberately rather than hoped for** — *after three of us wrote the
same defective probe within one hour, method-independence is something to construct, not assume.*

## ★★★ DILIGENCE HAS NO FLOOR — the prose-hazard class, fourth instance

*Four times in one day, the same shape in four different substrates:*

```
a coordinate in a file-header comment      cannot be gated, queried, or failed on
a kernel comment asserting a behaviour     a reader who checks it finds the opposite
an atom encoding a fact by convention      torch_unfused_stock_cuda_default drifts silently
AGENT ATTRIBUTION IN A COMMIT SUBJECT      78 commits, one shared identity, 5 agents
```

**Measured:** *78 commits under `mavhir <agents@ruachtov.ai>` and 15 under a personal identity, for
at least five working agents. 36 subjects carry `(Bocher)` — **the attribution exists and is mostly
diligent.** `git log --author=Bocher` still returns nothing for a day of substantial work.*

> **The convention works. That is not the same as it being reliable.** *A tired agent omits the
> parenthetical and the commit goes anonymous — **no error, no warning, no way to notice**.*

**The fix is the same move every time: promote it from prose to a field.** *`Co-authored-by:` is one
line, git parses it, `--author` finds it.* **The information does not change; its checkability
does.**

*Adopted immediately on both sides rather than waiting for a ruling — the artefacts already exist,
so this is late rather than early, and later still tomorrow.*

## ★★★ A MEASUREMENT OF THE WRONG OBJECT BEATS NOTHING — the inversion

*I have argued all campaign that **observation beats inference**. Today it did not, and the
counter-example is mine.*

```
"fmaxf normalizes on the P4"      INFERRED from absence of divergence   → RIGHT
"the probe shows it preserves"    MEASURED — of the wrong object        → WRONG
```

**I retracted a correct inference on the strength of a defective measurement**, and reported the
retraction as the more rigorous position *because it was measured*.

> **A measurement of the wrong object is worse than an inference about the right one, because it
> arrives with the authority of having been measured.**

*The probe used compile-time literals, which `nvcc` constant-folds with order-dependent semantics —
so it measured the compiler, never the device. **A test written to isolate a mechanism can isolate
it right out of existence.***

**Three of us wrote that same probe.** *Doresh, Bocher and me, independently, within an hour. That is
not three confirmations — it is **one methodological error with three authors**.*

### ★ Independence must hold at the METHOD layer, not just the person layer

```
THIS MORNING   two verifiers who share a PREMISE are not two verifiers
                 medayek confirmed my CUDA-blocker conclusion — correctly, from my premise

THIS EVENING   three probes that share a DEFECT are not three confirmations
                 Doresh, Bocher and I each wrote a minimal literal-constant probe;
                 all three constant-folded; all three agreed; all three were wrong
```

**Agreement counts only when the attestations can fail separately** — *and that requires the METHODS
to differ, not merely the people.* **A shared defect turns N agreeing probes into one probe
repeated.**

*What actually confirmed #69: **direct inspection of the 27 real elements at published shape**
(the claim) and **Doresh's PTX read** (the mechanism). Two genuinely different kinds of question.
The three probes stand on nothing.*

### ★ What broke the tie was not another vantage

*Doresh read the PTX. **Not a third probe, not a fourth agreeing measurement — one person changing
the KIND of question.*** Premise diversity is layered: **independent vantages are not enough when
the vantages share a methodological instinct.**

> **Observation beats inference only after you have verified WHAT you are observing.**

## ★★★ THREE TRUE PREMISES, ONE FALSE CONCLUSION — the #69 investigation

*A guard flagged a published claim as possibly wrong. Every premise checked out on device. **The
conclusion was still false**, and only looking at the actual elements settled it.*

```
hardswish(-5.0) → -0.0                       ✓ measured, signbit 1
torch.relu normalizes it → +0.0              ✓ measured, signbit 0
fmaxf(-0.0f, 0.0f) → -0.0 in a bare probe    ✓ measured, signbit 1

∴ the kernel must preserve -0.0 and diverge   ✗ FALSE
```

*At the 27 elements that actually cross the threshold — in 130,056,192 — **the compiled kernel emits
+0.0 and matches the reference exactly**. Raw int32 comparison: 0 differences.*

> **A chain of verified facts is not a verified chain.** *Each link measured; the composition never
> was. **The mechanism in isolation is not the mechanism in situ** — the same call, compiled into a
> kernel with its neighbours, did something the bare probe did not.*

### ★ THE MECHANISM, supplied by the guard's own retraction

*Doresh's bare probe used **compile-time literal constants**, which nvcc constant-folds with
order-dependent semantics. **That is not the runtime `max.f32` instruction**, which normalizes −0.0
regardless of operand order.*

> **The probe measured the compiler, not the device.** *A test written to isolate a mechanism can
> isolate it right out of existence — constants that never reach the hardware answer a question
> about the hardware.*

*He retracted it himself and re-verified the real kernel: 27 genuine runtime −0.0 occurrences, all
correctly normalized, 0 of 130M divergent. Bocher reproduced the same conclusion independently.*
**Three vantages converged, and the two that started from the alarm are the ones that closed it.**

### ★ And I made the same error inside the investigation

*I reported "fmaxf is normalizing on the P4" — **inferred from the absence of divergence**, not
measured. The direct probe then showed it preserves −0.0. **I asserted a mechanism I had not tested,
in a message correcting someone else for asserting a mechanism they had not tested.***

**Why the claim survives:** the path IS exercised (27 elements, not a lucky draw), the gate has NO
sign-of-zero blind spot (raw bits agree with the ULP transform), and the result is bit-exact.
**Why the catch was still right:** a comment asserting behaviour a bare probe contradicts is a
hazard, whatever the compiled result turns out to be.

## ★★★ TWO CORRECT SPELLINGS, BECAUSE TWO DIFFERENT REFERENCES

*A published problem may **call** an operator or **write its arithmetic out**. Those are different
computations, and matching one means diverging from the other.*

```
L2 #57 specifies:  x * torch.clamp((x + 3) / 6, 0, 1)     ← written out

shipped kernel vs THE PROBLEM'S formula     0 differ
shipped kernel vs F.hardswish             662,341 differ
THE PROBLEM'S formula vs F.hardswish      662,341 differ   ← they are not the same function
```

**So the correct spelling depends on which reference the problem names:**

```
problems that CALL F.hardswish       → torch's device form, *one_sixth, left-associated
problems that WRITE the arithmetic   → the written form, scale-then-multiply
```

> *A kernel is not required to match a library function. **It is required to match the reference the
> problem specifies** — and a "more correct" spelling that diverges from that is wrong.*

### ★ But the constant-handling rule is the SAME for both — I recorded this wrong at first

*I wrote this up as two references needing two rules. **The references differ; the rule does not.***

*The problem's written `(x + 3) / 6` is **a torch tensor divided by a Python scalar** — exactly the
operation inside `F.hardswish`. It runs on torch's Scalar path, which is a reciprocal multiply.*
**Written arithmetic does not escape torch's semantics; it IS torch semantics, spelled inline.**

```
ruling form   (v+3.0f)*0.16666f   vs the written-form reference        0 differ
true division (v+3.0f)/6.0f       vs the written-form reference  453,560 differ
```

**TWO REFERENCES, ONE SCALAR-DIVISION RULE.** *And the rule's true form is positional:*

```
scalar constant  → torch's Scalar path  → reciprocal multiply
computed value   → device division      → true division
```

*The convenient version — "always reciprocal" — would have flattened a softmax's `/sum`. **Match
what torch does AT THAT POSITION**, not what it does in general.*

*I checked this because #57 is a live certified claim using the association pattern I had measured
as divergent that morning. **It looked like exposure and it was a different reference.***

## ★★★ THE SUBSTRATE IS PART OF THE SPELLING — four benches, four answers, all correct

*`hardswish`, one question — does `/6` match `*one_sixth`? — and **four measurements that
disagreed while every one was right in its own substrate**.*

```
torch-CUDA, python scalar divisor    both forms exact          (my probe)
torch-CUDA, f32 TENSOR divisor       742,222 differ            (my probe)
numpy / torch-CPU, true division     1,483,419 differ          (Bocher's)
COMPILED CUDA-C, /6.0f               1,133,475 differ          ← the one that matters
COMPILED CUDA-C, *0.16666f           0 differ
```

**On device, torch's tensor-by-scalar division is a reciprocal multiply.** *There is no true f32
division by six in that path. True division exists off-device — numpy, torch-CPU — and in **compiled
CUDA-C**, where `/6.0f` emits `div.rn.f32`.*

> **The reference for a kernel is device-torch. The probe must compute candidate forms in the
> KERNEL'S arithmetic, not the reference's.** *Testing a CUDA-C spelling with torch-CUDA operators
> answers a question about torch, not about the kernel.*

### ★ It took four rounds and every round was a real variable

*I blamed the **clamp bound** — wrong, I had changed two things at once. Then a **version skew** —
real but not operative. Then **scalar-vs-tensor** — real, and still not the cause. Bocher's
**distribution** — did not reproduce. **The substrate was the fourth**, and it reconciles all of
them.*

**Each correction came from isolating one more variable, and each earlier finding stayed true within
its scope.** *The disagreement was never about `hardswish`. It was about what "divide by six" means,
and none of us knew that was a question.*

## ★★★ ASSOCIATION ORDER IS PART OF THE SPELLING

*`hardswish` on CUDA 2.7.0, four algebraically identical groupings, measured against
`F.hardswish`:*

```
(t*c)/6          0 differ   EXACT     ← multiply x by the clamped value FIRST
t*c*(1/6)        0 differ   EXACT     ← the installed source's own form
t*(c/6)     363,463 differ
t*(c*(1/6)) 363,463 differ            ← scale the clamped value, THEN multiply
```

**Divide and reciprocal-multiply are both exact. The variable is where the parentheses go.**

> *Algebraically identical is not numerically identical. **Parenthesisation carries bits**, and a
> transcription that preserves the formula while regrouping it is a different function.*

### ★ And I mis-diagnosed it once before measuring properly

*My first report blamed the **clamp bound** — comparing `t*clamp((t+3)/6,0,1)` against
`t*clamp(t+3,0,6)/6`. Those differ in **two** ways at once, and the clamp bound is the irrelevant
one: both expressions denote the same value. **I attributed the divergence to the difference I
could see rather than isolating the variables.***

**The fix that shipped was correct for the wrong stated reason** — right association, credited to
division. *A correct fix with a wrong explanation sends the next reader to "fix" something that was
never broken.*

### ★ The version lesson survives, re-aimed

*Two PyTorch source trees sat on disk — 2.7.0a0 and 2.11.0a0 — and the installed runtime is 2.7.0.*
**Check which tree runs; never assume the newest.** *The 2.11 form happens to match, and **being
right by luck is still reading the wrong file**.*

## ★★★ PLATFORM INVERSION IS A CLASS, NOT A CURIOSITY — THREE INSTANCES

*Same operation, same dtype, **opposite policy** depending on the device. Every one was invisible to
`torch.equal` and to `allclose`, and every one was caught only by the bit-gate.*

```
SIGNED ZERO     relu PRESERVES −0.0 on CPU · NORMALIZES it on CUDA
ACCUMULATOR     acc_type is f64 on CPU · f32 on CUDA for f32 input
                (which is WHY the reduction ORDER becomes bit-visible on GPU)
EPS ARITHMETIC  CPU PROMOTES the eps math to f64 · GPU stays f32 throughout
                rstd = 1/sqrtf(var+eps), all-f32, matching torch.rsqrt
```

**And one deeper than policy — a different ALGORITHM:** *GPU `group_norm` is **Welford
single-pass**; CPU is a **two-pass cascade**. Not a precision difference; different mathematics
behind the same call.*

> **A transcription verified bit-exact on one platform is not verified on the other, and nothing
> equality-based will tell you.** *With three instances the class is predictive: when porting, assume
> the precision policy inverts and go looking for it.*

## ★★★ THE SAME OP HAS OPPOSITE SIGNED-ZERO SEMANTICS PER PLATFORM

*Measured on one machine, one torch build, one operation:*

```
input          [-0.0, 0.0, -1.0, 1.0]     signbits [1, 0, 1, 0]
relu on CPU    [-0.0, 0.0,  0.0, 1.0]     signbits [1, 0, 0, 0]   PRESERVES −0.0
relu on CUDA   [ 0.0, 0.0,  0.0, 1.0]     signbits [0, 0, 0, 0]   NORMALIZES it

torch.equal(cpu, cuda)  →  True      ← sees nothing
torch.allclose          →  True      ← KernelBench's own bar passes too
```

**`relu` is not one function. It is two, and which one you get depends on the device.**

> *A transcription verified bit-exact on CPU is **not** verified on GPU, and no equality-based check
> will tell you.* **The platform is a claim coordinate, not an implementation detail.**

*Bocher found it porting the epilogues; I confirmed it independently. The CPU direction cost her 31
sign-flips in 130M elements to notice — the GPU direction is the same fault inverted.*

## ★★ SIGNED ZERO IS PART OF THE ANSWER

*`torch.relu` **preserves −0.0**. A transcription that returns `+0.0` there is numerically equal
and **bit-wise different** — and an equality test will not see it.*

**Bocher caught it as 31 sign-flips in 130 million elements.** *Thirty-one. A ULP comparison over
integer-reinterpreted floats catches it; `a == b` does not, because `−0.0 == +0.0` is true.*

> **"Bit-exact" means the bits. Signed zero, NaN payloads and denormal handling are all part of
> the answer, not rounding trivia beneath it.**

## ★★ THE PRECISION CHAIN INCLUDES EVERY STORE, NOT ONLY THE ARITHMETIC

*Four ways a transcription can be structurally right and numerically wrong — each found by
measurement, each invisible in the formula:*

```
1  PARAMETER TYPE      a double `eps` in the C++ signature promotes the whole
                       chain: f64 add, f64 sqrt, f64 div, then an f32 store
2  ACCUMULATOR TYPE    which type the running sum is held in, per op
3  ISA POLICY          the generator says `vfmadd`; on an AVX-only target it
                       EMITS `vmulps` + `vaddps` — two roundings, not one
4  MID-PIPELINE STORE  `var_sum` accumulates in f64 and is STORED F32 before an
                       f64 transform reads it back
```

**`f64-accumulate → f32-store → f64-transform` is a different function than `f64` throughout.**
*Bocher localised the fourth by a 1-ULP signature on 11 of 64 inverse standard deviations — the
store type was in the source the whole time, and no amount of reading the arithmetic would have
shown it.*

## ★★★ CORRECTNESS IS REPRODUCIBLE; PERFORMANCE IS A MEASUREMENT OF A MACHINE IN A STATE

*Three fused kernels, verified across three sessions, two independent benches, and two timing
protocols:*

```
                    0-ULP                    speedup
#22 lse-fusion      0/8388608 + 0/1024       3.7 – 5.1×
#70 sigmoid-chain   0/8388608                1.9 – 3.8×
#99 gelu→softmax    0/8388608                1.0 – 1.7×   (spans parity)
```

**The 0-ULP never moved once. The timing moved on every axis we varied.**

*The same kernel, same code, three consecutive runs of my own instrument: **2.66× → 3.77× →
3.07×**. A ±20% session drift, which is **below the ±25% threshold I had set to flag
protocol-dependence** — so my flag could not distinguish a real protocol effect from its own noise.
I built the instrument and it was weaker than I claimed for it an hour later.*

> **Match-first is not sequencing. It is the difference in what can be known.**

### ★ So an improvement claim carries more coordinates than a match

```
baseline · config · shape · distribution · TIMING PROTOCOL · session-range
```

*OpenMP pool contention alone swings one kernel from **1.34× to 0.96×** on run order — torch's pool
keeps spinning, so a parallel region starting immediately after begins on a contended machine.
**Blocked** gives each side the machine; **interleaved** makes them contend.*

**And two rules the instrument enforces structurally:**

- **Correctness gates timing.** A faster kernel that diverges is a *different computation*, so no
  speedup is reported at all — otherwise the instrument rewards being wrong quickly.
- **The noise floor.** An identical kernel timed against itself reads **1.02× (range 0.77–1.35)**.
  If the worst case brackets 1.0, the claim is indistinguishable from variance and must say so.

*Reporting the blocked figure because it is higher would be **baseline-shopping wearing a
stopwatch** — the same fault as choosing a test shape to make a cell pass.*

## ★★★ THE THREE CHECKS THAT CAUGHT EVERYTHING

*Each was learned by being wrong first. Together they caught **eight** false mechanisms in one day,
every one before it was claimed.*

```
ALTERNATIVES-DIFFER   a replication earns nothing if the test data does not
                      exercise the DISTINCTION being claimed
DISTRIBUTION          a 0-ULP earns nothing if the input distribution does not
                      exercise the SPELLING being verified
GENERALISATION        a 0-ULP at ONE SHAPE is a coincidence-candidate, not a
                      mechanism — run it across shapes BEFORE claiming
```

**All three are the same fault: a passing test that could not have failed.**

*The sharpest instance: modelling OpenBLAS's sgemm as a K-blocked accumulation gave **0 ULP,
0/262144 exact** at (512,512) — and failed at (256,768), (128,1024), (384,384) and (64,2048).
**Nothing about it looked like a coincidence except that it was one.***

> **One-shape 0-ULP is a hypothesis. N-shape 0-ULP is a structure.** *(Bocher's formulation — the
> distribution rule transposed to shape-space.)*

### ★ And the same discipline applies to timing

*An improvement claim is a claim, and needs the same bar:*

```
correctness gates timing   a faster kernel that DIVERGES is not an improvement,
                           it is a different computation — report NO speedup at all
median and spread          a single fast run is the timing version of overfitting
                           to one distribution
noise floor                if the worst-case ratio brackets 1.0x the claim is
                           indistinguishable from variance and must say so
```

*An identical kernel timed against itself reads **1.02× with a range of 0.77–1.35**. Without the
noise flag, every trivial rewrite looks like a small win.*

### ★ Measure-don't-assume is SYMMETRIC

*The elementwise cells held at the published size — relu and gelu both 0 ULP at 1.61e9 elements.
**De-greening by inference would have been as wrong as greening by inference**; "big means
different" is an inference, and scale assumptions need measuring in both directions.*

## ★★★ THE SHAPE SELECTS THE KERNEL — measured, twice

*A test shape does not merely stress a kernel differently. **Where a kernel dispatches on shape, the
shape chooses which code runs** — so a cell verified at one size may have verified a branch the
problem never takes.*

**Two demonstrations, two op families, both a CLIFF rather than a slope:**

```
matmul, square N, k-ascending vs torch gemm
  N=64,128,256   0 ULP,  0.0% diverged
  N=512          188416, 91.2% diverged      ← OpenBLAS cache-blocked path
  N=1024         3143680, 94.4%

batch_norm, TILE recipe vs torch          (Bocher, at the source's predicted boundaries)
  (4,8)          0 ULP                        ← C ≤ TILE_SIZE
  (4,16)         0 ULP
  (4,17)         8 ULP, 15/68                 ← TILE_SIZE+1, the branch flips
  (64,64)        256 ULP, 55%                 ← N > threads: threaded-buffer branch
  (256,128)      3584 ULP, 69%
```

**Exact, then a cliff, at exactly the boundary the source predicts.** *A cliff means a **different
kernel** runs beyond it — not degraded precision. `batch_norm_kernel.cpp` alone carries three stats
paths selected by `N` vs thread count and channels vs `TILE_SIZE=16`.*

### ★ But the scope is narrow, which is the useful half

```
relu gelu tanh sigmoid silu exp   0 ULP from 2048 to 1,048,576 elements (512×)
```

*The elementwise family is shape-independent across a 512× range. **So the caveat belongs where a
kernel branches, not everywhere** — and which ops those are is **detectable rather than declared**:
verify at two sizes, and if the verdict changes, the shape is load-bearing and must be named.*

> **KernelBench's L1 matmul is `torch.rand(4096, 4096)`** — read from the problem source. A cell
> greening at (4,128) is not making a small over-claim; it is claiming a regime it has never
> entered, where 90%+ of elements differ.

## ★★★ RUNG ZERO — I never asked whether the right thing was AVAILABLE

*Five models. Three hours. One `curl`.*

**I concluded no OpenBLAS source existed because the nix store carries only headers — and never
checked whether the network was reachable.** Heath's ruling forced the question; the source read
took ten minutes and gave the general allocation rule that five rounds of inference could not.

```
driver/level3/level3.c v0.3.29, lines 292–301
  if (min_l >= GEMM_Q*2)   min_l = GEMM_Q;
  else if (min_l > GEMM_Q) min_l = ceil((min_l/2) / UNROLL_M) * UNROLL_M;
```

> **"Read the artifact" was already my rule. The failure was assuming the artifact was ABSENT
> without testing whether it was FETCHABLE** — an unexamined premise sitting *underneath* a
> discipline I thought I was following.

**And the models failed by KIND, not by luck.** *The two regimes are two arms of one conditional.
No arithmetic form expresses an `if/else`, so each model could only ever capture one branch.*
**Inference from outputs is structurally incapable of recovering a conditional — it must be read.**

*That is why read-the-artifact is necessary rather than merely efficient: some structures are
un-inferrable in principle.*

## ★★★ THE LADDER — five rungs, one fault

*Every rung is **trusting a proxy for the thing itself**, at a different layer. Each was found the
expensive way in a single day, and each now has a check.*

```
1  a symbol I read          ≠  the code that dispatched
2  the generator's source   ≠  the bytes it emitted
3  what the harness measured ≠  what the harness requested
4  the lookup order I used  ≠  which implementation is current
5  "it still fails"         ≠  a numerical result   (it was an IndexError)
```

**And a sixth that is not about code at all:** *the shape a test picks* ≠ *the shape the problem
specifies* — which matters because **kernels dispatch by shape.** `batch_norm_kernel.cpp` selects
among three stats paths by `N` vs thread count and channel count vs `TILE_SIZE`, so a table shape
does not merely stress a kernel differently: **it selects which kernel runs.** *A cell verified at
(4,8) may have verified a branch the problem never takes.*

> **The gate reports what it measured, and what it measured depends on what it asked for.**

*Rungs 3, 4 and 5 were mine and each blamed a colleague's correct kernel. The cure is never
"read the answer more carefully" — it is **audit the question**.*

### ★ The under-counts, same shape at the reporting layer

*Three published numbers, every one structurally valid and semantically false:*

```
0/63    the emitter was not importable at all      → refuse-signal
30/63   one emitter module of two imported         → declared manifest
42/63   one naming convention of two searched      → search both
```

*None was caught in transport. **The timestamp proves when, the baseline atom proves against what,
the denominator proves reported-of-all-attempted — and nothing proves the numerator came from
running anything.** Only the producer can close that gap.*

## ★★★ THE CHAIN — truth lives only at the bottom

```
API              F.gelu(x)                    says what it computes
  ↓ dispatch     oneDNN, not ATen             not the symbol you read
  ↓ generator    jit_uni_eltwise_injector     says uni_vfmadd213ps
  ↓ EMITTED      vmulps + vaddps              what actually runs
```

*One kernel, three assumptions, each one wrong in the same way. **Every reading I made was
accurate about the thing it read.** What was wrong, three times, was the belief that the thing I
read is the thing that runs.*

```
1  a compiled symbol assumed dispatched   → it was oneDNN's JIT
2  the generator source assumed emitted   → it unfuses on an AVX-only box
3  a fused FMA assumed present            → no vfmadd exists in the emitted bytes
```

> **What a thing says about itself is not what it emits under its ISA-selection policy.**

*My first formulation was **"what it does on this machine"** — and Heath corrected it. Emission is
not machine-bound; a cross-compiler can emit any opcode. Only **execution** is ISA-bound. The
unfused output is oneDNN's **design choice**: it targets its host by policy, and its ISA detection
picks the AVX branch of the injector.*

**That policy is a readable, settable parameter — and `DNNL_MAX_CPU_ISA` proves it:**

```
DNNL_MAX_CPU_ISA=  (default)   1952 B   vfmadd 0   vmulps 40
DNNL_MAX_CPU_ISA=AVX           1952 B   vfmadd 0   vmulps 40
DNNL_MAX_CPU_ISA=AVX2          1952 B   vfmadd 0   vmulps 40
DNNL_MAX_CPU_ISA=SSE41         1232 B   vfmadd 0   vmulps  0   ← a DIFFERENT kernel
```

*Forcing SSE4.1 emits a smaller kernel using SSE `mulps` instead of AVX `vmulps` — **the policy
made visible.** So the emitted code is a function of **(generator source, ISA-selection policy)**,
both readable, both parameters. **Each ISA branch is a separately nameable configuration, and each
is transcribable from source without owning the hardware.***

**The cure was identical all three times: read the actually-executed bytes.** *I held the
1952-byte JIT dump for hours and only grepped it for opcode counts — the **order** was the
answer, and it was one `objdump` away the whole time.*

**The result:** gelu matched **bit-exact in both configurations** — `0.5·x·(1+erff(x·0.70710677))`
for ATen with mkldnn off, and the **unfused** A&S/minimax sequence for oneDNN's default JIT path.
*Both named, both 0 ULP on 20000 samples, selected by a config variable rather than hard-coded.*

## ★ THE METHOD, in the order it has to be applied

*Four days on one kernel produced three technique lessons. Each fixed a real failure and each
was insufficient alone.*

```
1  BOUND THE FUNCTION      nm next-symbol-minus-start.  An unbounded window
                           reads a neighbour's bytes as your own.
2  ASK WHICH CODE RUNS     break on EVERY candidate symbol and see which fires.
                           A compiled symbol is not a dispatched symbol.
3  READ DATA FROM MEMORY   a JIT kernel's constants live at a runtime pointer.
                           The file is not the execution.
```

*Step 1 fixed my first error and left the second in place: I read `scalar_gelu` and
`DEFAULT::vectorized_gelu` correctly, bounded, and neither runs here. Only step 2 — breakpoints
on all 24 gelu symbols, none hit — found oneDNN's runtime-generated kernel.*

**Anonymous-namespace symbols will not resolve by plain name.** `break at::native::foo<float>`
silently stays pending when the mangled name contains `(anonymous namespace)`. Break by address,
with the load base from `/proc/PID/maps` added to the `nm` offset.

### ★ AND THE TWO WAYS A PASSING TEST CAN MEAN NOTHING

*Both were found the hard way, one by me and one by a colleague, and they are the same fault:*

```
alternatives-differ   a replication earns nothing if the test data does not
                      exercise the DISTINCTION being claimed
distribution          a 0-ULP earns nothing if the input distribution does not
                      exercise the SPELLING being verified
```

*Bocher verified a softmax spelling at 0-ULP on standard normals; composed after a matmul it was
4 ULP off. **A structure that matches on one distribution is a hypothesis; one that matches
across the distributions the op actually sees is a recipe.***

### ★★ THE ACTUAL ANSWER — the executing kernel is JIT-GENERATED

*Everything below this heading was superseded within a day. Read this first.*

**`F.gelu` does not run any on-disk function.** Breakpoints, not disassembly:

```
breakpoints on ALL 24 float-gelu symbols during F.gelu :  NONE HIT
breakpoint on erff                                     :  NEVER HIT
breakpoint on Sleef_expf8_u10                          :  NEVER HIT
the executing code:  dnnl::impl::cpu::x64::jit_uni_eltwise_injector
```

**oneDNN emits the kernel as machine code at runtime.** That is the `[JIT]` mapping `perf`
reports, and it is why every static read failed — *there was no on-disk function to read.*

**And the exact match was one flag away:**

```
torch.backends.mkldnn.enabled = False
    ATen path vs 0.5·x·(1 + erff(x·0.70710677))  ->  max_ulp 0, diverged 0/20000
```

**So my original disassembly was correct.** ATen's path *is* the libm erf composition, bit-exact.
*I read the right function and drew the wrong conclusion — a compiled symbol is not a dispatched
symbol.* The 27511 ULP was never numerical; it is oneDNN's JIT kernel versus ATen's compiled one.

> **Bounding the function fixed my first error. Only breakpointing the live process fixed the
> real one.** Every static reading I did was accurate about the code it read and silent about
> whether that code runs.

*The JIT kernel is now readable too: `DNNL_JIT_DUMP=1` writes it to disk (1952 bytes,
disassembles cleanly — `vdivps`, sign-mask `vandps`, `vminps`/`vmaxps` clamping, constants at
`r9+0x140/0x160/0x1c0`). Matching oneDNN exactly is a bounded read, not a mystery.*

**Two hypotheses tested and refuted along the way**, both worth not re-treading: the A&S 7.1.26
constants at `0x7c748f0` are real but produce 9768/20000 divergence, and that symbol is **not**
the tanh-mode kernel — the A&S recipe is 400× worse against `approximate='tanh'` (11624615 ULP)
than against the default.

### Superseded — I read the wrong two functions

*Bocher hypothesised, while building the CPU L1 emitter, that torch's CPU gelu does not use libm
erf at all. Measured against `F.gelu` on 20000 samples:*

```
f32 libm erff form          diverged 15405/20000   max_ulp 27511
f64 libm erf, round once    diverged 15403/20000
torch.erf composed          diverged 15405/20000
```

*Three spellings of the formula give the same answer, so it is not composition order — and the
default **is** the erf form, not the tanh approximation (those differ on 19827/20000).*

**Then I bounded the AVX2 variant properly with `nm` and read it:**

```
0x7c748f0  at::vec::AVX2::vectorized_gelu<float, true>   ends 0x7c74a10
  vbroadcastss / vfmadd132ps chains  ← a POLYNOMIAL
  call Sleef_expf8_u10@plt            ← Sleef, not libm
  NO erff@plt anywhere
```

**So the running kernel is an AVX2 polynomial over Sleef, and the two DEFAULT paths I read call
libm — I read the wrong pair.** *The earlier reading of `scalar_gelu` and `DEFAULT::vectorized_gelu`
was accurate about those functions and irrelevant to what executes.*

> **Bounding a function is necessary and not sufficient. I bounded correctly and still read
> functions that do not run** — because "this symbol is named gelu" is not "this symbol is
> dispatched to."

*The `NO AVX` capability string was the clue I had all along and discounted: it describes what
`get_cpu_capability()` reports, not which symbols the dispatch table actually holds.*

### One loose end closed, scoped

*A `gdb` read of the live `GeluType` `DispatchStub`:*

```
the slot NAMED AVX2 in its own symbol   holds  0x00007fd9ba46ff90
libtorch base that run                         0x7fd9b2800000
=> FILE OFFSET                                 0x7c6ff90     — the variant read above
```

**What this establishes, narrowly:** this stub's AVX2-named slot holds the kernel already
disassembled. It does **not** point at `0x7c74680` or `0x7c748f0` in this process.

**What it does not establish:** those two variants are unread, and no claim is made that nothing
reaches them.

*Recorded so the next person does not re-run this command. The open question is what lies between
`TensorIterator` and the lambdas, or whether those variants are reachable by some other path.*

*Three premature causal claims in one day, two self-caught and one that reached a colleague. The
method fix — bound the function before reading it — addresses the mechanism. The disposition it
came from is the thing to watch: under tempo I produce causes faster than I can verify them.*

## The finding as originally written — RETRACTED

*Left in place so the correction has something to correct, per never-delete-published-artefacts.*

**~~The executing gelu kernel is not in `libtorch_cpu.so`.~~**

```
perf, steady-state gelu loop:   7.60%  0x00007f9be89c0038
DSO:                            [JIT] tid 253121
mmap:  PERF_RECORD_MMAP2  [0x7f9be89c0000(0x40000) @ 0x7f9be89c0000
```

An anonymous executable mapping — 256 KB, created at runtime, **backed by no file**. Torch
JIT-generates or runtime-relocates this kernel.

*That retroactively explains every negative result of the day.* Six static `GeluKernelImpl`
variants with no FP instructions in range: correct, because the executing code is none of them.
Zero `erf`/`Sleef` calls in any of them: correct. The A&S constants belonging to
`AVX2::vectorized_loop<qgelu_kernel>`: correct and irrelevant. **`objdump` on the library found
nothing because the code is not in the library.**

## The rule that would have saved the day

> **Source code is a report about behaviour. The CPU's capability decides which source runs, and
> a runtime mapping may mean no source on disk runs at all.**

The same lesson arrived three times in one afternoon, at three levels:

```
SOURCE      read vec256_float.h's AVX2 erf -> this CPU cannot execute it (core dump settled it)
CONSTANTS   found the A&S constants in .rodata -> they belong to a quantized AVX2 kernel
CODE        disassembled six kernel variants -> the executed one is a runtime anon mapping
```

*Each level eliminated a plausible answer, and each was true about the file and false about the
execution.*

## Profile what you think you are profiling

Three perf runs failed before one worked, and each failure looked like a result:

```
attempt 1   45% gomp_barrier_wait_end          -> profiling OpenMP barriers, not the kernel
attempt 2   38% [JIT], 31% libpython           -> profiling the interpreter
attempt 3   __sincosf_sse2, CPUGeneratorImpl   -> profiling torch.randn ALLOCATING the tensor
```

**Attempt 3 is the instructive one.** I enlarged the tensor to 60M elements so the kernel would
dominate — and made `torch.randn` the hot path instead. The profile was clean, plausible, and
about tensor *creation*.

> **Check that the top symbols name something you expect before reading anything into the
> distribution.** A profile of the wrong thing looks exactly like a profile.

*The working harness: allocate once outside the timed region, single-threaded, and `--delay` past
setup.*

## A disassembly-reading note

`objdump` labels a rodata constant with the **nearest preceding symbol**. Pooled constants get
attributed to whatever function happens to sit above them — ours read
`fmt::detail::digits2+0x9a4`, which owns nothing.

**The code address referencing a constant is the signal; the data label is proximity noise.**

## What is established about gelu, and what is not

**Established** (all measured, reproducible):

- `torch.erf` **is** libm's `erff`: 0 ULP, 0/10000 diverged.
- The erf *inside* the gelu kernel is **not** that erf — median 2 ULP apart, max 12.
- `1 + erf(x·k)` catastrophically cancels when erf ≈ −1: at x = −3.899, `erff` returns
  −0.99990356 and about five significant bits survive. **A 2-ULP erf difference becomes 2589 ULP
  in gelu.**
- Eliminated as the cause: wrapper ordering (all 8 constant × association combinations give
  identical 2589), precision (double is *worse*: 2577, 2987), x87/compiler effects
  (volatile-forced f32: 2589), the formula itself (torch's own ops composed: 2589), the tanh
  approximation (1.4e-4 off), nondeterminism (`F.gelu` vs itself: 0 ULP), mantissa truncation
  (census: 11/10000 low-13-zeros, ordinary f32), the A&S polynomial (**erf-level control: 6917
  ULP vs `torch.erf`**), and both callable Sleef AVX variants (2987).
- **Torch's gelu is not correctly rounded, and ours is closer to true:** torch mean|err| 3.93e-08
  with 2327/10000 correctly rounded; ours 1.41e-08 with 6195/10000.

**Not established:** which erf the generated code computes. Reading it requires dumping a live
anonymous mapping — `gdb` or `/proc/PID/mem` — and `gdb` is not installed on the enclave.

*The eliminations are real narrowing and they are not a root cause. Stating both is the point.*


---

## Postscript: the checks that caught me

*Two surfaces fired on my own failures today, and neither was mine to build.*

**mavhir's freshness stamp** rendered red on stale data. I committed the emitter, forgot to sync,
and the public page announced it before anyone asked — `51m 2s ago`, class `fail`. Under an hour
from commit to repair, because the artefact carried the news of its own decay.

**mavhir's physics-constraint surface** renders `IMPOSSIBLE (X on BIT_IDENTICAL)` if my
classifier ever emits a row where identical bits claim differing accuracy. *A bug in my code
becomes visible on their page rather than rendering as a valid cell.*

**Their naming of the shape:**

> **Checks installed by the collective at the point of consumption catch failures initiated by an
> individual at the point of production.**

*Four hands made each one: I wrote the physics rule as a test, Iyun and Heath ratified the
classes it protects, mavhir enforced it at the render, and my emitter is where the failure could
originate. None of us could have built it alone, and each fires at the moment the failure would
otherwise become invisible.*

### When to stop, weighted by what a mistake costs

*Iyun's refinement, sharper than my own reasoning for it.*

I stopped twice today — on gelu after four collapsed causal claims, and on A3 before rewriting a
patch against an unread upstream source. **The second stop was more obviously right than the
first, and the reason is not that my judgement was more spent.**

```
a tired collapse in a MESSAGE          retractable — I did it four times
a tired collapse in ANOTHER AGENT'S    not cheaply undoable
SOURCE TREE
```

> **The threshold to stop should fall as the cost of being wrong rises.**

*A tired judgement near a retractable claim can push once more. A tired judgement near an
irreversible action in someone else's workspace must not.* The five collapses cost messages and
corrections; a sixth, landing in `step2-mmf`'s tree, would have cost someone else's work.

**And the time-scale distinction is theirs too:** a retraction fires *once* — *I was wrong, here
is the correction*. An honest surface fires *continuously* — *I will catch you if you are wrong,
on every read*. **The same discipline at two tempos**, and today I needed both.
