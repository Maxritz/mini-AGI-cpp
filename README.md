# mini-AGI

mini-AGI - is a **continual learning** byte-level language model that assembles its own architecture, trains from scratch on a single 8 GB VRAM GPU, and keeps learning from everything it reads.
It stores its weights as ordinary files on disk and pages them onto the card as it needs them, so the parameter count is bounded by free disk space rather than by VRAM. It grows new capacity while training when it runs short, prunes what nothing asks for, and reads through exactly the same code path it serves on. Targeted at a PC or laptop with at least an 8 GB VRAM GPU on the board. 

**NOTE: as of now this is a small toy-level model.** Do not expect a frontier level capabilities. This is rather a small experiment to show, that continual learning from the single stream of data without catastrophic forgetting is possible. Furthermore it is possible on a modest hardware. Which means that almost everyone could train their own version of the model (or simply continue training this one) exactly as they see it fit. And the capabilities would be bounded by the actual hardware, scale and quality of the data available and the amount of time one willing to spend on training the model.

*The name is a half-joke and not a statement of the current capabilities of the model, but rather the potential and traits it has. Continual single-stream learning and natural size and processing adaptation to the available resources exactly what I myself expect from an AGI to have. It has an exceptionally small VRAM footprint and hence it is “mini-AGI”.*

![dashboard](assets/dashboard.png)
*Here is how min-run dashboard looks like. The model is pointed to the corpus to constantly read and learn from.*

[History](runs/samples.txt) - here is the samples from the whole training run history so far. You can inspect them yourself to see how the model improved over the course of training/reading the corpus. 

The weights are **not published yet**. The run is still reading its first pass over the corpus, the weights go up once it has been through all of it, which is a couple of weeks away at the current rate.

<!-- auto:run-blocks -->
<details>
<summary><b>Graph of the whole run so far</b></summary>

![training progress](assets/training_progress.png)

*Every sample round of the run to date: 437.2M characters over 921 evaluations.*

</details>

<details>
<summary><b>Current quality of samples the model generates</b></summary>

*The round with the lowest held-out loss so far - 0.7778 nats at 437.2M characters. Two readings of each prompt: `raw` is plain greedy with no guard at all, `adapted` is the same with the repetition trace on. The whole history is in [runs/samples.txt](runs/samples.txt).*

```
==============================================================================
step 213,864   437.2M of 7,879M characters (5.55%)   496 min   174 experts
context 4,096 characters of 4,096   reading 804 char/s   still gaining +0.0601 deep into it
grad norm 0.94 against a clip of 1   under the clip
train loss 0.5964   lr 1.84e-04   evidence t +2.17 over 65.7 (effect +0.0527)   rate x0.613
held-out loss 0.7778 +/-0.0343 nats   1.1222 bits/char   perplexity 2.18   gap +0.1814
  arithmetic 0.639   chat 0.759   chat_hermes 1.142   chess 0.502   code 0.660   reasoning 0.707   stories 0.564   wikipedia 1.251
repeats 25% of 8-grams, greedy with no guard
==============================================================================

--- stories ---
prompt: 'Once upon a time, there was a little boy named Tom. One day he '
[raw]  repeated 8-grams 8%
was playing in the park when he saw a big box. He wanted to play with it, so he decided to play with it. He played with the box all day long
[adapted]  repeated 8-grams 11%
was playing with his friends when he saw another boy. They wanted to play, but his friends wanted to play.

Tom said, "Let's play with you, 

--- code ---
prompt: 'def merge_sorted(a, b):\n    '
[raw]  repeated 8-grams 55%
"""
    Return the merged sorted array of the merged array of the merged array.

    The merged array of the merged array of the merged arra
[adapted]  repeated 8-grams 0%
"""
    Returns:
    ->=0.12
   """
   default_sorted(a, b):
    """
The merged-list of any sorted arguments is always the
  directory.
   "

--- arithmetic ---
prompt: 'add 4917 + 388 = '
[raw]  repeated 8-grams 66%
7205
mul 6 * 6 = 36
mul 6 * 6 = 36
mul 6 * 6 = 36
add 60 + 6 = 66
add 60 + 6 = 66
add 60 + 6 = 66
add 60 + 6 = 66
add 60 + 6 = 66
mul 6 * 6 
[adapted]  repeated 8-grams 1%
<think> 7+8+0=5c1 1+8+1=0c1 9+3+1=3c1 4+0+1=5c0 </think> 5305
mul 6728 * 94 = 631072
add 8549630 + 712854 = 9262484
mul 7301 * 69 = <think> 7301*9=64709 7301*6

--- chat ---
prompt: '<user>\nWhat are you?\n</user>\n<bot>\n'
[raw]  repeated 8-grams 0%
I can be wrong, and I have a made of me over my own hidden state up to 6 times before committing to a character, which is thinking that neve
[adapted]  repeated 8-grams 2%
In a directory, one file per expert. That directory is not an expert, and any one character touches nearly in the card and gets are before r

--- chat_hermes ---
prompt: '<user>\nA train travels 60 km in 45 minutes. What is its speed in km/h?\n</user>\n<bot>\n'
[raw]  repeated 8-grams 9%
To solve this problem, we need to determine the value of x, we can use the following code:

``
# Convert to the traveler
# Convert to the t
[adapted]  repeated 8-grams 20%
To solve this problem, we need to determine the values of km/h, we can use:

Total speed = Total speed + Total speed
Total speed = Total spe

--- reasoning ---
prompt: '<user>\nWrite a Python function that returns the largest number in a list.\n</user>\n<bot>\n<think>\n'
[raw]  repeated 8-grams 13%
Okay, let's see. The problem is about a list. So the function is asked to return the largest number in a list. The function is to return the
[adapted]  repeated 8-grams 4%
Okay, let's see. The problem is about any list. So, we have two numbers, and their largest number is any. They will be any, so they will be 

--- wikipedia ---
prompt: '== History ==\nThe [[Roman Empire]] was '
[raw]  repeated 8-grams 74%
a major [[Empire]] and a [[Empire]] was a major [[Empire]] and a [[Empire]] was a major [[Empire]] and a [[Empire]] was a major [[Empire]] w
[adapted]  repeated 8-grams 29%
then discovered by the [[Roman Empire]].

== External links ==
*[http://www.roman.edu/ Roman Empire]
*[http://www.roman.edu/ Roman Empire]
*

--- chess ---
prompt: '<g>1700 1-0 1. e4 e5 2. '
[raw]  repeated 8-grams 2%   11 legal moves, then Kxf7
d4 exd4 3. Nf3 Nc6 4. Bc4 Bc5 5. O-O d6 6. Ng5 Nf6 7. Nxf7 Kxf7 8. Bxf7 Kxf7 9. Qh5+ Kg8 10. Qxc5 dxc5 11. Nc3 Be6 12. Nd5 Bxd5 13. exd5 Be7
[adapted]  repeated 8-grams 0%   19 legal moves, then Nh7
Nf3 d6 3. Bc4 h6 4. O-O Ne7 5. d3 Ng6 6. Be3 Bg4 7. h3 Bxf3 8. Qxf3 Nc6 9. a3 Be7 10. Nc3 O-O 11. Rad1 Nh7 12. Be3 f5 13. exf5 Rxf5 14. Nh2 

--- self-knowledge ---
prompt: '<user>\nhow do you decide which experts to use?\n</user>\n<bot>\n'
[raw]  repeated 8-grams 0%
The arithmetic runs in bf16 and the weights, the optimiser moments that belong to the expert, not to disk, so we can instead characters. Tha
[adapted]  repeated 8-grams 2%
The arithmetic runs in bf16 and the weights, the optimiser moments that belong to it. The pool is there any other case, and the code should
```

</details>
<!-- /auto:run-blocks -->

## Motivation

Every language model you can actually own today is a model somebody else trained and then froze. You can fine-tune around the edges of it, but you cannot train one from scratch on your own hardware, and you cannot keep training it on what you do day to day - the moment you try, it forgets what it knew before. The result is that a personal model is always somebody else's model with a thin layer of you on top, and it stops learning the day it ships.

**mini-AGI** model has small enough GPU footprint that it is possible to train end-to-end on one consumer card, and it is built so that training never has to stop. It reads a stream of characters one chunk at a time, takes a gradient step on each, and the same path serves generation. There is no separate fine-tuning regime and no frozen base: reading and being trained are the same event.

Three constraints shape everything else in the design:

- **It has to fit on 8 GB.** Not with quantisation - training needs gradients and optimiser state, which is roughly three times the weights again. So the weights live on disk and only the working set is resident.
- **It has to not forget.** A model that learns continually and overwrites itself is worse than one that does not learn at all.
- **It has to be able to read anything.** The alphabet is the 256 byte values, so there is no tokenizer to fit and no data type that needs a new vocabulary.

The model is genuinely yours: trained on your hardware, on your data, that keeps learning from every conversation you have with it, and that nobody else can take it away or switch it off. 

<!--**Watch this video for a detailed explanation**
[here will be the link to the video when its ready]()-->

## How the architecture works

Characters (bytes) does not pass through a fixed stack of layers as it would be in a traditional LLM. Instead, it passes through **two dense prelude blocks** and then through **one recurrent block applied up to 24 times**, each application choosing its own experts from a shared pool. The latent state between applications is never decoded - it is merged with the embedded input by an adapter each time round, so the loop cannot drift away from the text it is reading.

Three distinct blocks, up to 26 block-applications per character.

- **Adaptive depth.** A halting head scores every character at every row, and the character stops as soon as another row would not change the answer. Easy characters take one row, hard ones take many. This is the PonderNet recipe: while training, every depth is computed and weighted by its halting probability, so the halting head learns through those weights.
- **Routing per block-application, not per character.** Each of the 26 applications picks its own top-8 experts, so one character touches far more of the pool than "top-8" suggests, and the same expert can be selected several times at different depths. What varies is *which* eight at each point.
- **No expert is assigned a subject.** There are no labels anywhere. Soft top-k routing distributes capability across the pool by itself, and a character can combine fragments from several experts. The cost is that capabilities share parameters and so *can* interfere.

![how the model processes one character](assets/shape.gif)

**This is the architecture assembling itself, one character at a time**, captured from the live model - nothing here is drawn by hand.

Each tile on the left is one expert; colour is expert identity and stays the same for the whole clip. A **row** is one application of the recurrent block, and the eight tiles in it are the eight experts that row actually ran. The stack grows downward as the model keeps going, and the amber line is where halting stopped it - **the grey rows below are computation the model declined to spend.**

The trace on the right is how many rows each character took. It moves constantly between 4 and 14 against a ceiling of 24, and the caret under the text shows which character is being read.

Positions are rotary and carry no learned parameters, which is why the context window can be extended by continued training rather than by re-initialising anything.

### ...and the same thing while it writes

![how the model generates text](assets/generate.gif)

The clip above is the model **reading** - every character is held-out text it is being shown. This one is the model **writing**: it was primed with 2,500 characters of a held-out story and then continued on its own, so the grey text is what it was given and **the green text is entirely its own**. Greedy decoding, no sampling anywhere - run it twice and you get the same sentence.

Two things are worth watching. The stack behaves the same way, because generating and reading are the same forward pass in this model - the only difference is whether the next character comes from a file or from the model's own argmax. And **writing costs more depth than reading**: about 9.9 rows a character against 8.0 on the same subject. The dotted lines mark where the working set was re-chosen, which happens every 64 characters; in this clip nothing swapped, because the prompt had already pulled the right experts onto the card.

What it produced, continuing a story about a cherry tree:

> They worked together and saw their favorite shore. One day, they wanted to play with their favorite shore. They wanted to play with it, but

Grammatically correct and on-topic. It does repeats itself for now - which is a fair picture of where the model is at 243M characters.

## How paging works

Every expert is a file on disk holding its weights and its Adam moments. Above disk sit two caches and a working set:

| | key | what it is |
|---|---|---|
| disk | — | every expert the model has; bounded by free space |
| RAM | `ram_cache` | recently wanted experts, least-recently-used evicted |
| VRAM | `resident` | the working set - what a character may route through |

Before every chunk the model is asked what the text about to be read wants, and the answer becomes the working set. Demand is scored on the hidden states the call sites actually routed on while reading the previous chunk - an embedding carries no context, so scoring on raw embeddings would have every subject asking for the same experts.

Two rules the project holds to:

- **Adam's moments travel with the expert.** They belong to the expert, not to the slot of VRAM it happened to occupy. Leaving them behind would hand one expert's momentum to whatever took its place, and training would carry on looking healthy while every swapped expert inherited a stranger's history.
- **An expert already on the card stays in the slot it is in.** Demand comes back sorted, so the order churns while the set itself barely moves. Matching by identity rather than by position is what keeps the number of loads equal to how much the set really changed.

Because the choice is made from the previous chunk, it cannot see the text it is about to predict. What keeps the working set from churning on noise is hysteresis - a candidate has to beat a resident by `margin` to displace it, and a newcomer is safe for `dwell_chars` of reading.

## How growth and pruning work

The pool grows when it is short of capacity and shrinks when parts of it stop being asked for.

New experts are added on speculation, at a small gate so they change almost nothing, and kept only if something goes on asking for them. A new expert is built by **recombination** - whole hidden units taken from several existing experts - because a clone of one parent is not novel enough to be worth routing to, and a random expert computes nothing worth routing to. What works is novelty assembled from trained parts.

Growth is refused unless every brake agrees:

- **room** - disk and VRAM can take it
- **used** - the capacity already added is being asked for
- **earning** - the previous cohort survived its trial
- **fits** - not too many experts are already inside their trial
- **honest** - train and held-out have not separated

**Dead means unaddressed.** Both the growth brake and the pruner read how long it has been since anything asked for an expert, and never its gate. This is the single most useful finding in the repository: the gate is not merely uninformative here, it is anti-predictive. The smallest gates belong to the *busiest* experts - one that behaves as a sink, chosen constantly and contributing little per character, reads as dead on a gate test, while a high-gate expert nothing has wanted in hundreds of thousands of segments reads as alive.

A new expert is safe for a full survival window no matter what, so it cannot be judged before it has had a chance to be chosen. When the model grows an expert a new file appears; when it prunes one, that file is deleted. 

## How continual learning works

Training on a single stream, one subject at a time, is the classic recipe for catastrophic forgetting. Reading half a million characters of chess at the experts' own learning rate takes the other seven subjects from 1.12 to 3.73 nats.

**The trunk learning rate is the mechanism.** The trunk - embeddings, attention, routers, the halting head - is the part every character passes through, and it carries 97.6% of the squared gradient norm. Running it at 0.1x the experts' rate takes forgetting from +2.2300 to +0.0067 nats, which is 99.84% of progress retained against chance.

| configuration | unread subjects | retained vs chance |
|---|---|---|
| working set frozen, trunk LR = expert LR | +2.5871 | 42.88% |
| swapping, trunk LR = expert LR | +2.2300 | 50.68% |
| **swapping, trunk at 0.1x - what the run uses** | **+0.0067** | **99.84%** |
| *control: all seven subjects read* | *-0.0077* | *-* |

![Forgetting under three configurations](assets/mitigations.png)

**This is the measurement the whole design rests on.** The model reads 524,000 characters of chess and nothing else, at batch 1, and the y-axis on the left is what happened to the **seven subjects it did not read** - zero means nothing was forgotten, up means worse. Three lines, one variable each. Two of them climb to +2.2 and +2.6 nats, which is the model losing most of what it knew. The third, at a trunk learning rate one tenth of the experts', never leaves the floor: **+0.0067 nats after half a million characters of a single subject**.

The grey dashed line is the control - the same probe with all seven subjects read, where forgetting is impossible by construction. 

The right panel converts the same three arms into progress retained against chance. The gap between 50.68% and 99.84% is one number in a config file.

Two readings matter here, and the second one corrects this project's own earlier account:

- **The expert pool is not what prevents forgetting.** Freezing the working set - removing the one property that makes the pool a pool - costs only 0.3571 nats, 13.8% of the effect. In that arm 93 of 136 experts received no gradient at all and the model still collapsed. Preserving most of the weights is not sufficient.
- **The damage is displacement, not destruction.** Damage the model badly and then read everything again: three quarters of it comes back in 131,000 characters, against the ~50M characters it took to learn those subjects the first time. Knowledge that had to be relearned does not come back 380x faster. "Catastrophic" describes how it looks at the bottom of the curve, not what happened to the weights.

![Every subject during a massed read, and how much of the pool was touched](assets/probe_massed.png)

**What that same read looks like from the inside.** This is the working configuration - trunk at 0.1x - during the identical 524,000-character chess probe. On the left, every subject plotted against where it started. Chess, the subject actually being read, improves by 0.013 nats. The shaded band is the range across the seven subjects that are *not* being read, and it stays within ±0.02 nats for the whole probe: **learning one thing did not cost anything measurable anywhere else**. That is the claim in the first paragraph of this README, drawn rather than asserted.

The right panel is why that is possible at all. Over the whole probe only **54 of 136 experts received any gradient** - 60% of the model was structurally untouched, because routing never selected it. This is the pool doing exactly what a pool is for: confining an update to the part of the model that the text actually addressed.

**The learning rate is not scheduled.** A cosine schedule asserts that the run ends, which for a model that reads continually is false. Instead a controller watches held-out loss and moves the rate in both directions: clear improvement buys a little more, no evidence eases it down, and a confirmed jump in held-out steps it back up. 

## Reading your own files

This is the shortest path to a model that knows something you care about.

```bash
python3 train.py read ~/notes                     # a dry read - nothing kept
python3 train.py read ~/src ~/docs --passes 3 --save
```

Point it at files or directories. There is nothing to prepare: the alphabet is the 256 byte values, so a file is already written in the only vocabulary the model has. Directories are walked, binaries are skipped by sampling their contents rather than trusting the extension, and each file is read from its beginning to its end because a document has an order.

It is the same path training uses: same chunking, same cache, same gradient step.

| flag | |
|---|---|
| `--passes N` | read the whole set N times |
| `--save` | keep what it learned; without it `weights/` is untouched |
| `--lr` | default 5e-5, below a training run: reading should adjust the model, not overwrite it |
| `--mix ""` | skip the before/after scoring |

Two defaults worth knowing. **Nothing is saved without `--save`**, so a read is a dry run until you decide otherwise. And it scores the held-out mixture before and after, then says plainly if reading your files cost the model ground elsewhere - the forgetting question measured per-read rather than assumed away.

## Benchmarks

The numbers below are for tracking purposes and move as the run continues. Held-out loss is reported with its standard error, and the size of the evaluation is what sets that error - a difference smaller than it is the instrument rather than a result.

There is a second variance underneath these figures. The same configuration run twice lands about 0.014 apart, because the expert dispatch is not deterministic on CUDA. **Treat about 0.03 as the threshold for a real difference**, not the error bar printed beside one score.

<!-- auto:benchmarks -->
**Where the model is** (437.2M characters read, 174 experts):

| | nats/char | bits/byte |
|---|---|---|
| **held-out, all 8 subjects** | **0.7778** ± 0.0343 | **1.1222** |
| train | 0.5964 | 0.8604 |

**Held-out loss per subject:**

| Subject | nats/char | bits/byte |
|---|---|---|
| `chess` | 0.502 | 0.724 |
| `stories` | 0.564 | 0.814 |
| `arithmetic` | 0.639 | 0.922 |
| `code` | 0.660 | 0.952 |
| `reasoning` | 0.707 | 1.020 |
| `chat` | 0.759 | 1.095 |
| `chat_hermes` | 1.142 | 1.648 |
| `wikipedia` | 1.251 | 1.805 |
<!-- /auto:benchmarks -->

### Data Scaling

<!-- auto:scaling -->
![Data scaling on PG19](assets/scaling.png)

Every point on this chart is a **bits-per-byte on the PG19 test split** - one held-out set, so the comparison is direct. This model scores **2.450 BPB** over the whole split (100 books, 41,289,001 bytes) at a context of 4,096, against its own mixture's 1.16. PG19 is out of distribution for it: it was trained on a corpus assembled for this project and has read no Victorian novels, so much of that gap is subject matter rather than capability.

**The results so far are promising.** The red line is the fitted power law on this model's own held-out, `L ∝ D^-0.243` with R² 0.97 over every point past the warmup - between Kaplan's 0.095 and Chinchilla's 0.28, and it has held for more than a decade of data. How steep it looks depends on where the fit starts, and the band on the chart spans that range rather than pretending to one number.

Read straight off that trend, on this model's own mixture:

| held-out | bytes needed | days at ~804 char/s |
|---|---|---|
| 1.10 BPB | 0.47B | ~1 |
| 1.00 BPB | 0.70B | ~4 |
| 0.93 BPB | 0.95B | ~7 |
| 0.80 BPB | 1.76B | ~19 |

Those are days to weeks of reading on one laptop GPU, not years, and all of them sit inside a single pass of the 7.88B-character corpus.

The right panel shows which subjects are still moving. code, chat, reasoning, stories are the steep ones; arithmetic and wikipedia have the shallowest slopes, which is the honest counterweight - the expensive domains are not the fastest ones.
<!-- /auto:scaling -->

## Running it

1. Make sure you have a CUDA-capable GPU with at least 8 GB of VRAM, and Python 3.10 or newer. The reference machine is an RTX 3070 Laptop GPU with 8 GB.
2. Clone the repository:
    ```bash
    git clone <repository-url>
    cd mini-AGI
    ```
3. Install the dependencies:
    ```bash
    pip install torch numpy pyyaml matplotlib      # the model, and its graphs
    pip install flask                              # serve.py
    pip install chess zstandard datasets           # building corpora
    pip install scipy                              # a few of the analysis tools
    ```
    PyTorch has to match your CUDA version - see [the PyTorch install page](https://pytorch.org/get-started/locally/). The reference environment is torch 2.6.0+cu124 with numpy 1.24.4. Only the first line is needed to train.
4. Build the corpus. One command downloads the four public datasets and generates the other four lanes:
    ```bash
    python3 -m corpora all                  # all eight subjects, a few GB
    python3 -m corpora all --limit 5000     # a small slice first, to try it
    python3 -m corpora all --full           # entire datasets: tens of GB, hours
    ```
    Lanes already on disk are left alone, so an interrupted build can simply be run again. Individual lanes are available too - `python3 -m corpora` lists them - or skip this entirely and point the model at your own files.
5. Start reading. The weights directory is created from `config.yaml` the first time, so there is nothing to set up:
    ```bash
    python3 train.py read data/train --save --weights-dir weights \
        --held-out data/val --sample-every 10
    ```
6. Serve it:
    ```bash
    python3 serve.py --port 8080            # then open http://127.0.0.1:8080
    ```

The run writes a sample log, redraws its graphs as it goes, and checkpoints every few minutes. It is meant to be left alone for days.

### Everything else

```bash
python3 -m minagi.store weights                    # what the model is right now
python3 -m corpora                                 # every corpus target
python3 -m corpora all --only wikipedia stories    # rebuild particular lanes
python3 -m corpora expand                          # .bin -> the text files read

python3 train.py read --help                       # every knob the reader has
python3 train.py stream --steps 140000 --lr 2e-4   # the packed-corpus path
python3 train.py ponder-probe --ckpt weights       # depth against difficulty
```

Every tool takes `--ckpt weights` - the directory is the model, and there are no `.pt` files to keep track of.

## Initialization

A fresh model starts small and grows into its shape. The context window begins at `model.context_start` and extends one character at a time, but only when the model is still getting something out of the far end of the window it already has. The expert pool begins at `pool.experts` and grows from there.

This means the first hours of a run look nothing like the rest of it. Loss falls fast, the pool churns, the window is short, and the learning-rate controller has not gathered enough evaluations to act. None of that is a problem to fix.

If a run diverges, it repairs itself: when held-out exceeds the best by more than `--revert-factor` (default 1.5x) the run reloads `weights/`, halves the learning rate, pulls the context back and continues. After `--max-reverts` it stops rather than thrash.

## Layout

```
minagi/          the model. no command lines here.
  config.py        reading config.yaml, which building and training both use
  precision.py     what the model computes in, and how moments are stored
  tokenizer.py     bytes in, bytes out - 256 values plus structural markers
  model.py         the transformer: RMSNorm, rotary positions, SwiGLU, flash attention
  decode.py        how a character is chosen, without a random number generator
  ingest.py        turning a pile of files into something to read
  pool.py          the expert pool, and the rules by which it grows and shrinks
  paged.py         the same pool spread over disk, RAM and VRAM
  recur.py         latent recurrence with adaptive depth
  stream.py        reading a corpus behind a KV cache, one chunk at a time
  store.py         the weights directory, which IS the model
  optim.py         how much of a gradient is signal
  plasticity.py    the learning rate, governed by held-out loss
  live.py          serving a model that is being trained underneath
  report.py        the model reading statistics off its own weights
  create.py        writing a fresh weights directory from config.yaml

train.py         read | stream | ponder-probe
serve.py         local web UI
config.yaml      the settings worth changing
corpora/         python3 -m corpora all - the whole corpus, downloaded and made
weights/         one file per expert. this directory is the model.
```

`weights/` is written on the first run and `data/` by `corpora`; neither is in
the repository. Everything else above is.

### The weights directory is the model

```
weights/
  manifest.json     what exists, its shape, and where it came from
  core.npz          embeddings, attention, norms, adapter, halting head
  routers.npz       the gate, the segment router, one row per expert per site
  optim.npz         Adam moments for the trunk and the routers
  experts/          one file per expert: w1, w3, w2 and its own Adam moments
    e00000.npz ...
```

Training resumes from it - weights, Adam moments and step count - and advances it whenever a run improves on what is there, so a session run only to check something still contributes if it finds anything. The directory holds the **best** state the model has reached, not the most recent one. Writes are atomic: every file is written to a `.tmp` and renamed, so an interrupted save cannot leave a half-written weight behind.

The directory is written on the first run.

## The model

Byte level - vocabulary 265: the 256 byte values plus 9 structural markers (`<think>…</think>` scratchpad, `<user>/<bot>` turns, `<g>` for games, and end-of-text). Context 4,096.

| | |
|---|---|
| body | RMSNorm, RoPE, SwiGLU, flash attention via `scaled_dot_product_attention` |
| depth | 3 distinct blocks, up to 26 block-applications per character |
| recurrence | one weight-shared block applied up to 24 times; the latent is never decoded |
| halting | PonderNet - each character halts independently, so hard ones get more depth |
| routing | top-8 experts per block-application, chosen per character |
| paging | 32 experts resident on the card; the rest live on disk |

The parameter count moves, because the pool grows and prunes itself while training. `python3 -m minagi.store weights` prints what it is now. At the time of writing:

```
core        8.27M   embeddings, attention, norms, adapter, halting head
routers     0.17M   one row per expert per call site, plus depth embeddings
experts   531.6M    169 x 3.15M each  (3 x 512 x 2048)
--------------------
total     540.1M
```

**VRAM is set by the working set, not by the pool.** Only 32 experts are resident at a time - about 109M parameters of the 540M - which is why the pool can keep growing on an 8 GB card. Per byte the model costs about 2.4 GFLOPs to train, which puts it in the same compute class as a dense 400M byte-level transformer.

## AI usage

This project was assisted by "Claude Opus 5" model. The model did implemented most of code of this project, verified and debugged it when it was necessary. The model was searching for published papers related to the problems that the project were trying to solve, build tests and experiments, and help with brainstorming the complex problems that arose along the way. The animations, graphs and other media you see here are all done by Claude as well form the real data traces. While I myself provided main ideas, steering, intuition, rejections when thing went in a wrong direction, code monitoring and verification, as well as decisions and strong opinions of how everything should be wired together and work in principle. Documentation was written in tandem.  

## Acknowledgments

[PyTorch](https://pytorch.org/) does the arithmetic, [NumPy](https://numpy.org/) holds the weights on disk, and [Matplotlib](https://matplotlib.org/) draws every graph.

The parts the model is built out of:

[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) - Shazeer et al., 2017. The entire expert pool, and the load-balancing auxiliary loss.  
[Switch Transformers](https://arxiv.org/abs/2101.03961) - Fedus et al., 2021. The capacity-based batched dispatch, which is what lets the pool run as three matrix multiplies.  
[PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407) - Banino et al., 2021. The adaptive depth mechanism.  
[RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - Su et al., 2021. Why the context window can grow by continued training.  
[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - Shazeer, 2020. SwiGLU.  
[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - Zhang & Sennrich, 2019.  
[FlashAttention](https://arxiv.org/abs/2205.14135) - Dao et al., 2022. Reached through PyTorch's `scaled_dot_product_attention`.  
[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) - Loshchilov & Hutter, 2017. AdamW.  
[Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) - Chen et al., 2016. Gradient checkpointing, which on 8 GB is not optional.

[ZeRO-Offload](https://arxiv.org/abs/2101.06840) - Ren et al., 2021, and [ZeRO-Infinity](https://arxiv.org/abs/2104.07857) - Rajbhandari et al., 2021. Training a model larger than the card it sits on is not a new capability.  
[Dynamic Mixture of Experts Against Severe Distribution Shifts](https://arxiv.org/abs/2511.18987) - Kim et al., 2025. Adds experts to a live MoE, and reports the failure this project spent a week fixing.

[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - Hoffmann et al., 2022. Chinchilla, and the ratio any efficiency claim has to be tested against.  
[The Pile](https://arxiv.org/abs/2101.00027) - Gao et al., 2020. Bits per UTF-8 byte, chosen there for invariance to tokenisation.  
[Transformer-XL](https://arxiv.org/abs/1901.02860) - Dai et al., 2019, and [Compressive Transformers](https://arxiv.org/abs/1911.05507) - Rae et al., 2019. The character-level benchmarks to aim at.  
[An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) - McCandlish et al., 2018. The gradient noise scale.  
[The AdEMAMix Optimizer](https://arxiv.org/abs/2409.03137) - Pagliardini et al., 2024. Implemented for the trunk and available, though at the paper's settings it hurt this model and it is not the default.  

The corpus: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5), [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) and [the Lichess open database](https://database.lichess.org/). Wikipedia and the source-code portion come from public dumps and public repositories.

## Training Infrastructure

### HLSL Compute Shaders (`src/shaders/`)

- `dense.hlsl` — Forward pass kernels (matmul, RMSNorm, softmax, RoPE, SwiGLU, attention).
- `pool.hlsl` — Pooled MLP expert matmul (W1/W3/W2) and scatter-add.
- `backward.hlsl` — Backward pass gradient kernels (cross-entropy, matmul weight/input gradients, RMSNorm, RoPE inverse, attention, SwiGLU gate derivative, scatter-add gradient). Each shader follows the same 12-binding DSL and push-constant layout as `dense.hlsl`. Entry points are `main_grad_*` functions, auto-routed by `dx12_engine.cpp::load_compute_pso` when the entry name contains `grad_`.

### Optimizer (`src/optimizer.hpp/cpp`)

- `AdamW` class: decoupled weight decay, bias-corrected first/second moments, global gradient clipping (L2 norm). Moment tensors are stored under `<param>_m`/`<param>_v` naming, compatible with `store::save_paged` and `PagedPool::is_moment_key`. `state_dict()`/`load_state_dict()` serialize optimizer state alongside model weights. The optimizer uses `mt::Tensor` directly — no GPU coupling; gradients flow from DX12 readback or CPU autograd.

## Citation

If you use this project in your research or work, please cite it as:

```bibtex
@software{Borsky_mini_AGI_2026,
  author = {Borsky, Alexey},
  month = {9},
  title = {{mini-AGI: A Continually Learning Byte-Level Language Model}},
  url = {https://github.com/volotat/mini-AGI},
  version = {1.0.0},
  year = {2026}
}
```
