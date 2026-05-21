#!/usr/bin/env python3
"""
Generate Reverse-kNN groundtruth using PyTorch GPU.

RkNN(q, k) = { x in D | q is among x's k nearest neighbors in D }
            = { x in D | dist(x, q) <= dist(x, x_k) }

where x_k is x's k-th nearest neighbor in D (excluding x itself).

Output format (rknn_gt):
  [nq:      uint32]
  [offsets:  (nq+1) * uint32]   # offsets[i] = start index for query i
  [ids:     total * uint32]     # concatenated result IDs
  [dists:   total * float32]    # concatenated distances

Usage:
  python scripts/gen_rknn_gt.py \
      --base /data2/sift/sift_base.fbin \
      --query /data2/sift/sift_query.fbin \
      --k 10 \
      --output /data2/sift/sift_rknn_gt.bin \
      --chunk_base 2048 \
      --chunk_query 512
"""

import argparse
import struct
import numpy as np
import torch
from pathlib import Path


def load_fbin(path: str) -> np.ndarray:
    """Load fbin format: [n: uint32][d: uint32][n*d float32]."""
    with open(path, "rb") as f:
        n, d = struct.unpack("II", f.read(8))
        data = np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)
    print(f"  Loaded {path}: n={n}, d={d}")
    return data


def compute_kth_nn_dists(base: np.ndarray, k: int, chunk_size: int,
                         device: torch.device) -> np.ndarray:
    """
    For each base vector, find the distance to its k-th nearest neighbor
    among all other base vectors. Returns array of shape (n,).

    Chunked over rows (query side) to fit in GPU memory.
    Distance: squared L2.
    """
    n, d = base.shape
    # precompute ||b||^2 for all base vectors
    base_t = torch.from_numpy(base).to(device)          # (n, d)
    base_norms = (base_t * base_t).sum(dim=1)            # (n,)

    kth_dists = torch.empty(n, device=device, dtype=torch.float32)

    # k+1 because we exclude self (distance 0)
    topk_k = min(k + 1, n)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = base_t[start:end]                        # (cs, d)
        chunk_norms = base_norms[start:end]              # (cs,)

        # dist^2(chunk[i], base[j]) = ||c_i||^2 - 2*c_i·b_j + ||b_j||^2
        dists = chunk_norms.unsqueeze(1) - 2 * (chunk @ base_t.T) + base_norms.unsqueeze(0)
        dists.clamp_(min=0)

        # top-k smallest distances (including self at dist=0)
        topk_vals, _ = torch.topk(dists, topk_k, dim=1, largest=False, sorted=True)
        # index k is the (k+1)-th smallest, i.e. k-th NN (self is index 0)
        kth_dists[start:end] = topk_vals[:, min(k, topk_k - 1)]

        done = end
        print(f"\r  kth-NN dists: {done}/{n}", end="", flush=True)

    print()
    return kth_dists.cpu().numpy()


def compute_rknn(base: np.ndarray, queries: np.ndarray, kth_dists: np.ndarray,
                 chunk_query: int, chunk_base: int,
                 device: torch.device):
    """
    For each query q, find all base vectors x where dist(x, q) <= kth_dist[x].
    Returns (offsets, all_ids, all_dists).
    """
    n_base = base.shape[0]
    nq = queries.shape[0]

    base_t = torch.from_numpy(base).to(device)
    base_norms = (base_t * base_t).sum(dim=1)
    kth_t = torch.from_numpy(kth_dists).to(device)

    offsets = np.zeros(nq + 1, dtype=np.uint32)
    all_ids = []
    all_dists = []

    for q_start in range(0, nq, chunk_query):
        q_end = min(q_start + chunk_query, nq)
        q_chunk = torch.from_numpy(queries[q_start:q_end]).to(device)  # (cq, d)
        q_norms = (q_chunk * q_chunk).sum(dim=1)                       # (cq,)

        # accumulate results per query in this chunk
        chunk_ids = [[] for _ in range(q_end - q_start)]
        chunk_dists_list = [[] for _ in range(q_end - q_start)]

        for b_start in range(0, n_base, chunk_base):
            b_end = min(b_start + chunk_base, n_base)
            b_chunk = base_t[b_start:b_end]                            # (cb, d)
            b_chunk_norms = base_norms[b_start:b_end]                  # (cb,)
            b_chunk_kth = kth_t[b_start:b_end]                         # (cb,)

            # dist^2(q_i, b_j) = ||q_i||^2 - 2*q_i·b_j + ||b_j||^2
            d2 = q_norms.unsqueeze(1) - 2 * (q_chunk @ b_chunk.T) + b_chunk_norms.unsqueeze(0)
            d2.clamp_(min=0)

            # mask: dist(b_j, q_i) <= kth_dist[b_j]
            # shape (cq, cb), compare each column j against kth[b_start+j]
            mask = d2 <= b_chunk_kth.unsqueeze(0)

            # extract hits
            qi_indices, bj_indices = torch.where(mask)
            d2_hits = d2[qi_indices, bj_indices]

            qi_np = qi_indices.cpu().numpy()
            bj_np = bj_indices.cpu().numpy() + b_start
            d2_np = d2_hits.cpu().numpy()

            for qi_local, bj_global, dist_val in zip(qi_np, bj_np, d2_np):
                chunk_ids[qi_local].append(bj_global)
                chunk_dists_list[qi_local].append(dist_val)

        for i in range(q_end - q_start):
            q_idx = q_start + i
            ids_arr = np.array(chunk_ids[i], dtype=np.uint32)
            dists_arr = np.array(chunk_dists_list[i], dtype=np.float32)
            # sort by distance
            if len(ids_arr) > 0:
                order = np.argsort(dists_arr)
                ids_arr = ids_arr[order]
                dists_arr = dists_arr[order]
            all_ids.append(ids_arr)
            all_dists.append(dists_arr)
            offsets[q_idx + 1] = offsets[q_idx] + len(ids_arr)

        print(f"\r  RkNN queries: {q_end}/{nq}", end="", flush=True)

    print()
    return offsets, all_ids, all_dists


def save_rknn_gt(path: str, nq: int, offsets: np.ndarray,
                 all_ids: list, all_dists: list):
    """Save rknn groundtruth in variable-length format."""
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
    parser = argparse.ArgumentParser(description="Generate RkNN groundtruth (GPU)")
    parser.add_argument("--base", required=True, help="Base vectors (.fbin)")
    parser.add_argument("--query", required=True, help="Query vectors (.fbin)")
    parser.add_argument("--k", type=int, required=True, help="k for RkNN")
    parser.add_argument("--output", required=True, help="Output groundtruth file")
    parser.add_argument("--chunk_base", type=int, default=1024,
                        help="Chunk size for base vectors in kth-NN computation")
    parser.add_argument("--chunk_query", type=int, default=256,
                        help="Chunk size for query vectors in RkNN computation")
    parser.add_argument("--chunk_base_rknn", type=int, default=0,
                        help="Chunk size for base vectors in RkNN step (0 = same as chunk_base)")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")

    # 1. Load data
    print("[1] Loading vectors")
    base = load_fbin(args.base)
    queries = load_fbin(args.query)
    assert base.shape[1] == queries.shape[1], "Dimension mismatch"

    # 2. Compute k-th NN distance for every base vector
    print(f"[2] Computing k-th NN distances (k={args.k})")
    kth_dists = compute_kth_nn_dists(base, args.k, args.chunk_base, device)
    print(f"  kth_dist stats: min={kth_dists.min():.4f}, "
          f"max={kth_dists.max():.4f}, mean={kth_dists.mean():.4f}")

    # 3. Compute RkNN for each query
    print(f"[3] Computing RkNN groundtruth")
    chunk_base_rknn = args.chunk_base_rknn if args.chunk_base_rknn > 0 else args.chunk_base
    offsets, all_ids, all_dists = compute_rknn(
        base, queries, kth_dists, args.chunk_query, chunk_base_rknn, device)

    # stats
    counts = np.diff(offsets)
    print(f"  Result counts: min={counts.min()}, max={counts.max()}, "
          f"mean={counts.mean():.1f}, median={np.median(counts):.0f}")

    # 4. Save
    print(f"[4] Saving to {args.output}")
    save_rknn_gt(args.output, len(queries), offsets, all_ids, all_dists)
    print("Done.")


if __name__ == "__main__":
    main()
