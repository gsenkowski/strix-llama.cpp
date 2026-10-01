#!/usr/bin/env python3
"""Create a reduced vocabulary and a reduced MTP draft head from token statistics.

The statistics come from `--token-stats` of the inference engine (CSV: token_id,prompt_count,generated_count); several
files, or directories of files, are summed up and treated as one source.

The tokens are taken in order of decreasing count until they make up the target coverage of the counted tokens. Control,
user-defined and unknown tokens and the special tokens (BOS, EOS, ...) are always part of the reduced vocabulary.

The output is a copy of the input GGUF with two additions, so the engine needs only this one file:

  * the tensor blk.<N>.nextn.draft_head.weight: the rows of the LM head that belong to the reduced vocabulary. The rows are
    copied as they are, so a quantized head stays bit-exact and nothing is dequantized or requantized.
  * the key <arch>.nextn_draft_vocab_ids: the token id of each row, which maps a token chosen from the reduced head back to
    the original vocabulary.

A <output>.vocab.csv with the reduced vocabulary (reduced_id,token_id,text) is written next to it; it is accepted by
mtp_vocab_check.py as well.

The MTP block of the input must score with the LM head of the model (output.weight, or token_embd.weight when tied),
which is the case for qwen35 and qwen35moe. Only numpy and the `gguf` package are needed.

  mtp_vocab_reduce.py stats/ --gguf mtp-model.gguf --coverage 95 -o mtp-model-95.gguf
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from gguf import GGUFValueType, GGUFWriter

import mtp_vocab_common as common

DRAFT_HEAD_NAME = "blk.{bid}.nextn.draft_head.weight"
SUPPORTED_ARCHS = ("qwen35", "qwen35moe")


def select_tokens(counts: np.ndarray, mandatory: set[int], coverage: float, min_count: int, max_size: int,
                  round_to: int) -> np.ndarray:
    """Sorted token ids of the reduced vocabulary."""
    n_vocab = len(counts)
    total = int(counts.sum())
    must = np.zeros(n_vocab, dtype=bool)
    must[list(mandatory)] = True
    if max_size and int(must.sum()) > max_size:
        common.die(f"--max-size {max_size} is smaller than the {int(must.sum())} special tokens that are always kept")

    need = max(0, math.ceil(coverage / 100.0 * total - 1e-9) - int(counts[must].sum()))

    cand = np.flatnonzero(~must)
    # most frequent first, lower token id first on ties
    cand = cand[np.lexsort((cand, -counts[cand].astype(np.int64)))]
    cand_counts = counts[cand].astype(np.uint64)

    k = 0
    if need > 0:
        k = int(np.searchsorted(np.cumsum(cand_counts), need, side="left")) + 1
        k = min(k, len(cand))
    if min_count > 0:
        k = min(k, int((cand_counts >= min_count).sum()))
    if max_size:
        k = min(k, max_size - int(must.sum()))
    if round_to > 1:
        size = int(must.sum()) + k
        k += min(-size % round_to, len(cand) - k)
        if max_size and int(must.sum()) + k > max_size:
            k = max_size - int(must.sum())
    keep = must.copy()
    keep[cand[:k]] = True
    return np.flatnonzero(keep)


def find_head(reader, arch: str):
    names = {t.name: t for t in reader.tensors}
    n_block = int(reader.get_field(f"{arch}.block_count").contents())
    n_nextn = int(reader.get_field(f"{arch}.nextn_predict_layers").contents()) if reader.get_field(f"{arch}.nextn_predict_layers") else 0
    if n_nextn <= 0:
        common.die("the GGUF has no MTP block (nextn_predict_layers is 0)")
    bid = n_block - n_nextn
    if any(n.startswith(f"blk.{b}.nextn.shared_head_head") for n in names for b in range(bid, n_block)):
        common.die("the MTP block has its own LM head (nextn.shared_head_head), which is not supported")
    if DRAFT_HEAD_NAME.format(bid=bid) in names:
        common.die("the GGUF already has a reduced draft head; use the original GGUF as input")
    head = names.get("output.weight") or names.get("token_embd.weight")
    if head is None:
        common.die("the GGUF has no LM head (output.weight / token_embd.weight); it was probably converted with "
                   "--mtp-shared-embd. Use a GGUF that contains the LM head")
    return head, bid


def copy_gguf(reader, writer: GGUFWriter, extra_tensors) -> None:
    skip = {"general.architecture"}
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key in skip:
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        value = field.contents()
        if vtype == GGUFValueType.ARRAY and len(value) == 0:
            # the gguf writer cannot store an empty array; a missing key reads the same
            print(f"note: dropping the empty array {key}")
            continue
        writer.add_key_value(key, value, vtype, sub)
    writer.data_alignment = reader.alignment
    for t in reader.tensors:
        writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
    for name, data, dtype in extra_tensors:
        writer.add_tensor(name, data, raw_dtype=dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stats", nargs="+", help="statistics CSV files and/or directories containing them (summed up)")
    ap.add_argument("--gguf", "-m", required=True, help="GGUF with the original MTP draft head and the original vocabulary")
    ap.add_argument("--vocab", "-v", help="GGUF to take the original vocabulary from, if --gguf has none")
    ap.add_argument("--coverage", "-c", type=float, required=True, help="target coverage in percent of the counted tokens, e.g. 95")
    ap.add_argument("--source", "-s", choices=common.SOURCES, default="generated",
                    help="which counts to cover: the generated tokens (default), the prompt tokens or both")
    ap.add_argument("--min-count", type=int, default=0, help="never add a token that occurred fewer times than this")
    ap.add_argument("--max-size", type=int, default=0, help="upper limit of the reduced vocabulary size, special tokens included (0 = none)")
    ap.add_argument("--round-to", type=int, default=1, help="round the size of the reduced vocabulary up to a multiple of N with the next most frequent tokens")
    ap.add_argument("--output", "-o", required=True, help="output GGUF")
    ap.add_argument("--no-vocab-csv", action="store_true", help="do not write the .vocab.csv next to the output")
    ap.add_argument("--force", "-f", action="store_true", help="overwrite the output")
    args = ap.parse_args()

    if not 0.0 < args.coverage <= 100.0:
        common.die("--coverage must be in (0, 100]")
    if args.min_count < 0 or args.max_size < 0 or args.round_to < 1:
        common.die("--min-count and --max-size must not be negative, --round-to must be at least 1")
    out = Path(args.output)
    if out.exists() and not args.force:
        common.die(f"{out} exists (use --force to overwrite)")
    if out.resolve() == Path(args.gguf).resolve():
        common.die("the output must not be the input")

    reader = common.open_gguf(args.gguf)
    arch = common.reader_arch(reader)
    if arch not in SUPPORTED_ARCHS:
        print(f"warning: architecture '{arch}': the engine uses the reduced head only for {', '.join(SUPPORTED_ARCHS)}")

    vocab = common.load_vocab(common.open_gguf(args.vocab) if args.vocab else reader)
    if vocab is None:
        common.die("no vocabulary (tokenizer.ggml.tokens) in the input; pass the model GGUF with --vocab")

    head, bid = find_head(reader, arch)
    n_embd, n_rows = int(head.shape[0]), int(head.shape[1])
    if n_rows != vocab.n_vocab:
        common.die(f"the LM head has {n_rows} rows, but the vocabulary has {vocab.n_vocab} tokens")

    stats_all, files = common.load_stats(args.stats)
    stats_all = common.pad_stats(stats_all, vocab.n_vocab)
    counts = stats_all[args.source]
    if int(counts.sum()) == 0:
        common.die(f"the statistics contain no {args.source} tokens")

    ids = select_tokens(counts, vocab.mandatory_ids(), args.coverage, args.min_count, args.max_size, args.round_to)

    # report
    in_red = np.zeros(vocab.n_vocab, dtype=bool)
    in_red[ids] = True
    print(f"statistics files   : {len(files)}")
    print(f"original vocabulary: {vocab.n_vocab} tokens, LM head [{n_embd} x {n_rows}] {head.tensor_type.name}")
    print(f"reduced vocabulary : {len(ids)} tokens ({100.0 * len(ids) / vocab.n_vocab:.2f} % of the vocabulary, "
          f"{len(vocab.mandatory_ids())} always kept)")
    for src in common.SOURCES:
        c = stats_all[src]
        tot = int(c.sum())
        got = 100.0 * int(c[in_red].sum()) / tot if tot else float("nan")
        mark = "  <- target" if src == args.source else ""
        print(f"coverage {src:9}: {got:8.4f} % of the tokens{mark}")
    achieved = 100.0 * int(counts[in_red].sum()) / int(counts.sum())
    if achieved + 1e-9 < args.coverage:
        print(f"warning: the target coverage of {args.coverage} % was not reached ({achieved:.4f} %): limited by --max-size / --min-count or the statistics")

    # reduced head: the rows as they are, whatever the quantization
    rows = np.ascontiguousarray(head.data[ids])
    name = DRAFT_HEAD_NAME.format(bid=bid)

    writer = GGUFWriter(str(out), arch)
    writer.add_array(f"{arch}.nextn_draft_vocab_ids", [int(i) for i in ids])
    copy_gguf(reader, writer, [(name, rows, head.tensor_type)])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MiB), reduced head: {name} [{n_embd} x {len(ids)}]")

    if not args.no_vocab_csv:
        csv_path = out.with_suffix(".vocab.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["reduced_id", "token_id", "text"])
            for r, t in enumerate(ids):
                w.writerow([r, int(t), vocab.text(int(t))])
        print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
