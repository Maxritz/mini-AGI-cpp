#!/usr/bin/env python3
"""Golden generator for Wave F: paged pool + pondering + greedy decode.

Runs the REAL minagi with torch, drives the exact golden run of task_waveF.txt
section 8 (phase A chunk reading 170/170/156, phase B replay + 32-char decode),
and writes tests/golden_wavef/{ids.txt,trace.log,logits.bin,summaries.bin,weights/}.

Seeds 1234..1253 are tried until every determinism assert passes (the golden
rejects tie-prone seeds). Determinism is double-checked by re-running the whole
pipeline with the same seed and requiring bit-identical buffers.
"""

import dataclasses
import os
import shutil
import struct
import sys
import tempfile
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from minagi.decode import pick_next
from minagi.paged import PagedPool
from minagi.pool import PooledMLP
from minagi.recur import RecurConfig, RecurCoder
from minagi import store

torch.set_num_threads(1)
warnings.filterwarnings("ignore", module="minagi.recur")

DATA_PHRASE = ("The quick brown fox jumps over the lazy dog. 0123456789\n"
               "Pack my box with five dozen liquor jugs.\n").encode()

V = 256 + 9                 # 265: 256 bytes + 9 control tokens
N = 496
SEG_LENS = [170, 170, 156]
M = 32
D = 24
DW = dict(strength=2.5, decay=0.88, window=64)
RESIDENT, RAM, EXPLORE = 3, 4, 0.15
SEEDS = range(1234, 1254)


def make_cfg():
    return RecurConfig(
        vocab_size=265, n_layer=8, n_head=3, d_model=24, block=1024, d_ff=48,
        rope_theta=10000.0, tie_embeddings=True,
        use_pool=True, pool_experts=6, pool_d_ff=32, pool_depth=1,
        pool_top_k=2, pool_capacity_factor=1.5, pool_max=6, pool_aux=0.01,
        n_prelude=1, n_recur=2, n_coda=0, max_steps=5, min_steps=1,
        train_steps_mean=0.0, bptt_window=4, ponder_beta=0.01,
        halt_prior=0.4, halt_thresh=0.9,
    )


def adapt_margin(logits_row, prev):
    """top-1 vs top-2 gap of the ADAPTED logits (section 7 applies before argmax)."""
    lg = logits_row.float().clone().reshape(-1)
    n = min(prev.shape[1], DW["window"])
    tail = prev[0, -n:]
    w = (DW["decay"] ** torch.arange(n - 1, -1, -1, dtype=torch.float32))
    trace = torch.zeros(V, dtype=torch.float32)
    trace.index_add_(0, tail, w)
    lg = lg - DW["strength"] * trace
    sv = torch.topk(lg, 2).values
    return float(sv[0] - sv[1])


def seg_scores(pool):
    with torch.no_grad():
        lg = pool.segment_router(pool.summary.unsqueeze(0))[0, :6].clone()
        lg = lg + 0.5 * pool.gate.detach().abs().log1p()
    return lg.float()


def pipeline(seed, weights_dir):
    trace = []
    tl = trace.append

    torch.manual_seed(seed)
    cfg = make_cfg()
    model = RecurCoder(cfg).eval()

    edir = os.path.join(weights_dir, "experts")
    os.makedirs(edir, exist_ok=True)
    for i, ex in enumerate(model.pool.experts):
        np.savez(os.path.join(edir, "e%05d.npz" % i),
                 w1=ex.w1.weight.detach().numpy(),
                 w3=ex.w3.weight.detach().numpy(),
                 w2=ex.w2.weight.detach().numpy())

    pool = PagedPool(edir, cfg.d_model, cfg.pool_d_ff, cfg.pool_experts,
                     resident=RESIDENT, ram_capacity=RAM, device="cpu",
                     read_only=True)
    with torch.no_grad():
        pool.gate.copy_(1.0 + 0.02 * torch.arange(cfg.pool_experts,
                                                  dtype=torch.float32))
    pool.explore = EXPLORE
    for m in model.modules():
        if isinstance(m, PooledMLP):
            m._pool[0] = pool
    model.pool = pool
    pool.attach_sites(model)

    tl("MODEL vocab=265 d_model=24 n_head=3 d_ff=48 block=1024 rope_theta=10000.0"
       " n_prelude=1 n_recur=2 n_coda=0 max_steps=5 min_steps=1 halt_prior=0.4"
       " halt_thresh=0.9 pool_experts=6 pool_d_ff=32 pool_top_k=2"
       " pool_capacity_factor=1.5 pool_max=6 pool_resident=3 pool_ram=4"
       " explore=0.15 margin=0.10 dwell=4 seg_lens=170,170,156 N=496 M=32"
       " read_only=1")

    crows = []          # (global idx, halt_row) in reading order
    recs = []           # per-position (halt_row, f32[V] logits)
    summaries = []

    def emit_tiers(seg):
        t = pool.tiers
        tl("TIERS %d %d %d %d %d" % (seg, t.reads, t.hits, t.evictions,
                                     t.writebacks))

    # ---------- phase A: chunked reading, one forward per segment ----------
    corpus = (DATA_PHRASE * ((N // len(DATA_PHRASE)) + 1))[:N]
    caches = model.empty_caches()
    pos = 0
    for seg, L in enumerate(SEG_LENS):
        sw = torch.sort(seg_scores(pool).detach().clone(),
                        descending=True).values
        assert float(sw[2] - sw[3]) > 1e-7, f"SEG{seg} top-3 margin"
        model.begin_segment()
        tl("SEG %d %s" % (seg, ",".join(str(s) for s in pool.slots)))
        emit_tiers(seg)
        chunk = torch.tensor(list(corpus[pos:pos + L]), dtype=torch.long
                             ).unsqueeze(0)
        hal, extra = model(chunk, caches=caches, pos_offset=pos, collect=True)
        steps = extra["steps"][0].long().tolist()
        for j in range(L):
            hr = steps[j]
            crows.append((pos + j, hr))
            recs.append((hr, hal[0, j, :].detach().float().numpy()))
            tl("C %d %d" % (pos + j, hr))
        summaries.append(pool.summary.clone().float().numpy())
        pos += L

    # ---------- phase B: replay + 32-char decode ----------
    sw = torch.sort(seg_scores(pool).detach().clone(), descending=True).values
    assert float(sw[2] - sw[3]) > 1e-7, "SEG3 top-3 margin"
    model.begin_segment()
    tl("SEG 3 %s" % ",".join(str(s) for s in pool.slots))
    emit_tiers(3)

    replay = torch.tensor(list(corpus), dtype=torch.long).unsqueeze(0)
    fre = model.empty_caches()
    out0, _ = model(replay, caches=fre, pos_offset=0)
    summaries.append(pool.summary.clone().float().numpy())

    prev = replay
    offset = N
    cur = int(prev[0, N - 1].item())
    gen = []
    for t in range(M):
        o, _ = model(torch.tensor([[cur]], dtype=torch.long),
                     caches=fre, pos_offset=offset)
        row = o[:, -1, :]
        assert adapt_margin(row, prev) > 1e-3, f"decode #{t} top-1/top-2 margin"
        nxt = pick_next(row, prev, temperature=0.0,
                        adapt_strength=DW["strength"],
                        adapt_decay=DW["decay"],
                        adapt_window=DW["window"])
        assert nxt is not None and 0 <= int(nxt) < V, "pick_next out of range"
        gen.append(int(nxt))
        prev = torch.cat(
            [prev, torch.tensor([[gen[-1]]], dtype=torch.long)], 1)
        cur = gen[-1]
        offset += 1
        summaries.append(pool.summary.clone().float().numpy())

    assert len(summaries) == 3 + 1 + M, "summaries: 3 segs + replay + 32 decode"
    assert int(pool.segments) == 4, "segments must be 4 after the run"

    hr = np.array([r[0] for r in recs], dtype=np.int32)
    lrows = np.stack([r[1] for r in recs]).astype(np.float32)

    store.save(model, path=weights_dir, step=0, val=0.0, opt=None,
               cfg=dataclasses.asdict(cfg), verbose=True,
               extra={"read_only": True, "pool_ram": 4})

    return {"crows": crows, "hr": hr, "logits": lrows,
            "summaries": np.asarray(summaries, dtype=np.float32),
            "gen": gen, "trace": trace}


def same(a, b):
    return (a["crows"] == b["crows"]
            and np.array_equal(a["hr"], b["hr"])
            and np.array_equal(a["logits"], b["logits"])
            and np.array_equal(a["summaries"], b["summaries"])
            and a["gen"] == b["gen"])


def write_golden(outdir, res):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "ids.txt"), "w", newline="\n") as f:
        f.write("496 32\n")
        for b in (DATA_PHRASE * ((N // len(DATA_PHRASE)) + 1))[:N]:
            f.write("%d\n" % b)
        for i in res["gen"]:
            f.write("%d\n" % i)
    with open(os.path.join(outdir, "trace.log"), "w", newline="\n") as f:
        f.write("\n".join(res["trace"]) + "\n")
    with open(os.path.join(outdir, "logits.bin"), "wb") as f:
        f.write(struct.pack("<II", N, V))
        for (_, hr), row in zip(res["crows"], res["logits"]):
            f.write(struct.pack("<I", hr))
            f.write(row.astype("<f4").tobytes())
    with open(os.path.join(outdir, "summaries.bin"), "wb") as f:
        f.write(struct.pack("<I", len(res["summaries"])))
        for s in res["summaries"]:
            f.write(s.astype("<f4").tobytes())


def main():
    outdir = os.path.join(REPO, "tests", "golden_wavef")
    for seed in SEEDS:
        with tempfile.TemporaryDirectory() as a, \
                tempfile.TemporaryDirectory() as b:
            try:
                r1 = pipeline(seed, os.path.join(a, "weights"))
                r2 = pipeline(seed, os.path.join(b, "weights"))
            except AssertionError as e:
                print(f"seed {seed}: assert failed ({e}); next")
                continue
            if not same(r1, r2):
                print(f"seed {seed}: nondeterministic; next")
                continue
            write_golden(outdir, r1)
            shutil.copytree(os.path.join(a, "weights"),
                            os.path.join(outdir, "weights"))
            print(f"seed {seed}: golden written to {outdir}")
            print(f"  gen ids: {r1['gen']}")
            print(f"  sections: 4; summaries: {len(r1['summaries'])}")
            return 0
    print("no seed passed all asserts")
    return 1


if __name__ == "__main__":
    sys.exit(main())