#!/usr/bin/env python3
"""
Generate Reverse-kNN groundtruth for large datasets using GPU streaming.

Unlike gen_rknn_gt.py, this version never loads the full base to GPU.
Base stays in CPU RAM; query and reference chunks are streamed to GPU
with a running top-k merge for the kth-NN stage.

Designed for datasets that exceed GPU memory (e.g., 10M x 1024 = 40GB).

Output format (same as gen_rknn_gt.py):
  [nq:      uint32]
  [offsets:  (nq+1) * uint32]
  [ids:     total * uint32]
  [dists:   total * float32]

Usage:
  python scripts/gen_rknn_gt_large.py \
      --base /data2/msmarco/msmarco10M_base.fbin \
      --query /data2/msmarco/msmarco_query.fbin \
      --k 10 \
      --output /data2/msmarco/msmarco10M_rknn_gt_k10.bin \
      --chunk_query 1024 --chunk_ref 65536
"""

import argparse
import struct
import time
import numpy as np
import torch


def load_fbin(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = struct.unpack("II", f.read(8))
        data = np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)
    print(f"  Loaded {path}: n={n}, d={d}")
    return data


def compute_kth_nn_dists(base: np.ndarray, k: int,
                         chunk_query: int, chunk_ref: int,
                         device: torch.device) -> np.ndarray:
    """
    For each base vector, find squared L2 distance to its k-th NN.

    Loop order: OUTER = reference chunk, INNER = query chunk.
    Each ref chunk is loaded once; each query chunk is reloaded per ref
    iteration. Since chunk_ref is typically much larger than chunk_query,
    the expensive ref upload dominates transfer cost, so keeping it in
    the outer loop cuts total PCIe traffic by (n/chunk_query)x vs the
    reverse ordering.

    Running top-(k+1) buffer lives on GPU as a single (n, topk_k)
    tensor so all queries share it across r iterations. For n=10M,
    topk_k=11, fp32, this is 440 MB — small on a 24 GB GPU.

    base_norms are uploaded once upfront (n × 4 bytes) instead of being
    re-uploaded per chunk.
    """
    n, d = base.shape
    topk_k = k + 1  # +1 to account for self (dist=0)

    # Precompute norms on CPU then upload once.
    base_norms_np = np.einsum("ij,ij->i", base, base)  # (n,)
    all_norms = torch.from_numpy(base_norms_np).to(device)  # 40 MB for 10M

    # Global running top-k buffer on GPU, shared across all r iterations.
    all_buf = torch.full((n, topk_k), float("inf"), device=device)

    total_pairs = ((n + chunk_query - 1) // chunk_query) * ((n + chunk_ref - 1) // chunk_ref)
    pair_count = 0
    t0 = time.time()

    for r_start in range(0, n, chunk_ref):                       # OUTER: ref
        r_end = min(r_start + chunk_ref, n)
        r_chunk = torch.from_numpy(base[r_start:r_end]).to(device)   # (rs, d) — loaded once per outer
        r_norms = all_norms[r_start:r_end]                            # GPU view, no transfer

        for q_start in range(0, n, chunk_query):                 # INNER: query
            q_end = min(q_start + chunk_query, n)
            qs = q_end - q_start

            q_chunk = torch.from_numpy(base[q_start:q_end]).to(device)  # (qs, d)
            q_norms = all_norms[q_start:q_end]                            # GPU view

            # Squared L2 in-place: dists = ||q||^2 + ||r||^2 - 2 q·r
            dists = q_chunk @ r_chunk.T                   # (qs, rs)
            dists.mul_(-2.0)
            dists.add_(q_norms.unsqueeze(1))
            dists.add_(r_norms.unsqueeze(0))
            dists.clamp_(min=0.0)

            # Extract top-k smallest from this ref chunk
            actual_k = min(topk_k, dists.shape[1])
            chunk_topk, _ = torch.topk(dists, actual_k, dim=1, largest=False, sorted=False)
            del dists

            # Merge with running buffer for this query slice
            if actual_k < topk_k:
                pad = torch.full((qs, topk_k - actual_k), float("inf"), device=device)
                chunk_topk = torch.cat([chunk_topk, pad], dim=1)
            merged = torch.cat([all_buf[q_start:q_end], chunk_topk], dim=1)  # (qs, 2*topk_k)
            del chunk_topk
            new_topk, _ = torch.topk(merged, topk_k, dim=1, largest=False, sorted=True)
            del merged
            all_buf[q_start:q_end] = new_topk

            pair_count += 1
            if pair_count % 50 == 0 or pair_count == total_pairs:
                elapsed = time.time() - t0
                eta = elapsed / pair_count * (total_pairs - pair_count)
                print(f"\r  kth-NN: {pair_count}/{total_pairs} chunks  "
                      f"(r {r_end}/{n}, q {q_end}/{n})  "
                      f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s   ", end="", flush=True)

        del r_chunk  # free ~1 GB before next outer iter

    # buf[:, 0] = self (dist 0); buf[:, k] = k-th NN (excluding self)
    kth_dists = all_buf[:, min(k, topk_k - 1)].cpu().numpy()
    elapsed = time.time() - t0
    print(f"\n  kth-NN done in {elapsed:.1f}s")
    return kth_dists


def compute_rknn(base: np.ndarray, queries: np.ndarray,
                 kth_dists: np.ndarray,
                 chunk_query: int, chunk_ref: int,
                 device: torch.device):
    """
    For each query q, find all base vectors x where dist(x,q) <= kth_dist[x].

    Loop order: OUTER = base chunk (loaded once per b-iter),
    INNER = query chunk. Queries, their norms, base_norms, and kth_dists
    are all uploaded to GPU once upfront — only b_chunk is streamed per
    outer iter. Same optimization rationale as compute_kth_nn_dists: the
    large ref tensor dominates PCIe, so keeping it outer cuts transfer
    by (nq/chunk_query)x vs the reverse ordering. For SIFT with 10k
    queries + 10M base this is ~40x fewer GB moved.

    Per-query hit lists live across all base iterations; memory cost is
    ~n_hits × 12 bytes total (well under 10 MB even for nq=10k).
    """
    n_base = base.shape[0]
    nq = queries.shape[0]

    # One-time uploads: queries (4-28 MB), norms + kth (each n_base × 4 bytes ~40 MB @ 10M)
    q_all = torch.from_numpy(queries).to(device)             # (nq, d)
    q_norms_all = (q_all * q_all).sum(dim=1)                  # (nq,)
    base_norms_np = np.einsum("ij,ij->i", base, base)
    all_b_norms = torch.from_numpy(base_norms_np).to(device)
    all_b_kth = torch.from_numpy(kth_dists).to(device)

    # Per-query hit accumulators (alive across all base iterations)
    hit_ids = [[] for _ in range(nq)]
    hit_dists = [[] for _ in range(nq)]

    n_b_chunks = (n_base + chunk_ref - 1) // chunk_ref
    t0 = time.time()

    for bi, b_start in enumerate(range(0, n_base, chunk_ref)):   # OUTER: base
        b_end = min(b_start + chunk_ref, n_base)
        b_chunk = torch.from_numpy(base[b_start:b_end]).to(device)   # (rs, d) — once per outer
        b_norms = all_b_norms[b_start:b_end]
        b_kth = all_b_kth[b_start:b_end]

        for q_start in range(0, nq, chunk_query):                # INNER: query
            q_end = min(q_start + chunk_query, nq)
            q_chunk = q_all[q_start:q_end]                        # GPU view, no transfer
            q_norms = q_norms_all[q_start:q_end]                   # GPU view

            d2 = q_chunk @ b_chunk.T                               # (qs, rs)
            d2.mul_(-2.0)
            d2.add_(q_norms.unsqueeze(1))
            d2.add_(b_norms.unsqueeze(0))
            d2.clamp_(min=0.0)

            mask = d2 <= b_kth.unsqueeze(0)
            qi_indices, bj_indices = torch.where(mask)
            d2_hits = d2[qi_indices, bj_indices]
            del d2

            if qi_indices.numel() > 0:
                qi_np = qi_indices.cpu().numpy()
                bj_np = bj_indices.cpu().numpy() + b_start
                d2_np = d2_hits.cpu().numpy()

                for qi_local, bj_global, dist_val in zip(qi_np, bj_np, d2_np):
                    qi_global = q_start + int(qi_local)
                    hit_ids[qi_global].append(bj_global)
                    hit_dists[qi_global].append(dist_val)

        del b_chunk  # free ~2 GB before next outer iter
        elapsed = time.time() - t0
        print(f"\r  RkNN: base chunks {bi + 1}/{n_b_chunks}  "
              f"(b {b_end}/{n_base})  elapsed {elapsed:.1f}s", end="", flush=True)

    # Serialize per-query results after all base chunks processed.
    offsets = np.zeros(nq + 1, dtype=np.uint32)
    all_ids = []
    all_dists = []
    for i in range(nq):
        ids_arr = np.array(hit_ids[i], dtype=np.uint32)
        dists_arr = np.array(hit_dists[i], dtype=np.float32)
        if len(ids_arr) > 0:
            order = np.argsort(dists_arr)
            ids_arr = ids_arr[order]
            dists_arr = dists_arr[order]
        all_ids.append(ids_arr)
        all_dists.append(dists_arr)
        offsets[i + 1] = offsets[i] + len(ids_arr)

    print()
    return offsets, all_ids, all_dists


def save_rknn_gt(path: str, nq: int, offsets: np.ndarray,
                 all_ids: list, all_dists: list):
    total = int(offsets[nq])
    with open(path, "wb") as f:
        f.write(struct.pack("I", nq))
        f.write(offsets.tobytes())
        for ids in all_ids:
            if len(ids) > 0:
                f.write(ids.tobytes())
        for dists in all_dists:
            if len(dists) > 0:
                f.write(dists.tobytes())
    print(f"  Saved {path}: nq={nq}, total_results={total}, "
          f"avg_results={total / nq:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate RkNN groundtruth for large datasets (GPU streaming)")
    parser.add_argument("--base", required=True, help="Base vectors (.fbin)")
    parser.add_argument("--query", required=True, help="Query vectors (.fbin)")
    parser.add_argument("--k", type=int, required=True, help="k for RkNN")
    parser.add_argument("--output", required=True, help="Output groundtruth file")
    parser.add_argument("--chunk_query", type=int, default=8192,
                        help="Query-side chunk size for kth-NN (default: 8192)")
    parser.add_argument("--chunk_ref", type=int, default=262144,
                        help="Reference-side chunk size for kth-NN (default: 262144)")
    parser.add_argument("--chunk_rknn_base", type=int, default=524288,
                        help="Base chunk size for RkNN stage (default: 524288)")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")

    # 1. Load data to CPU
    print("[1] Loading vectors to CPU RAM")
    base = load_fbin(args.base)
    queries = load_fbin(args.query)
    assert base.shape[1] == queries.shape[1], "Dimension mismatch"
    print(f"  Base memory: {base.nbytes / 1e9:.1f} GB")

    # 2. Compute k-th NN distance for every base vector (streaming)
    print(f"[2] Computing k-th NN distances (k={args.k}, streaming)")
    print(f"  chunk_query={args.chunk_query}, chunk_ref={args.chunk_ref}")
    kth_dists = compute_kth_nn_dists(
        base, args.k, args.chunk_query, args.chunk_ref, device)
    print(f"  kth_dist stats: min={kth_dists.min():.4f}, "
          f"max={kth_dists.max():.4f}, mean={kth_dists.mean():.4f}")

    # 3. Compute RkNN for each query (streaming)
    print(f"[3] Computing RkNN groundtruth (streaming)")
    offsets, all_ids, all_dists = compute_rknn(
        base, queries, kth_dists,
        chunk_query=min(256, queries.shape[0]),
        chunk_ref=args.chunk_rknn_base,
        device=device)

    counts = np.diff(offsets)
    print(f"  Result counts: min={counts.min()}, max={counts.max()}, "
          f"mean={counts.mean():.1f}, median={np.median(counts):.0f}")

    # 4. Save
    print(f"[4] Saving to {args.output}")
    save_rknn_gt(args.output, len(queries), offsets, all_ids, all_dists)
    print("Done.")


if __name__ == "__main__":
    main()
