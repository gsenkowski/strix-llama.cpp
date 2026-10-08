#!/usr/bin/env python3
"""Check token statistics against a reduced MTP draft vocabulary.

Statistics come from `--token-stats` of the inference engine (CSV: token_id,prompt_count,generated_count). Several files,
or directories of files, are summed up and treated as one source. The reduced vocabulary is either a reduced draft head
GGUF (written by mtp_vocab_reduce.py) or its .vocab.csv.

For the prompt tokens, the generated tokens and both together the tool reports

  * by token id: how many of the token ids that occurred are not in the reduced vocabulary,
  * by occurrence: which share of all the tokens is not in the reduced vocabulary, i.e. what a draft head restricted to
    the reduced vocabulary can never propose, and
  * the most frequent tokens that are not in the reduced vocabulary.

Only numpy and the `gguf` package are needed.

  mtp_vocab_check.py stats/ --reduced mtp-reduced.gguf --vocab model.gguf
"""

from __future__ import annotations

import argparse

import numpy as np

import mtp_vocab_common as common


def pct(part: float, whole: float) -> str:
    return f"{100.0 * part / whole:7.3f} %" if whole > 0 else "      - "


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stats", nargs="+", help="statistics CSV files and/or directories containing them (summed up)")
    ap.add_argument("--reduced", "-r", required=True, help="reduced vocabulary: reduced draft head GGUF or .vocab.csv")
    ap.add_argument("--vocab", "-v", help="GGUF with the original vocabulary, to show the text of tokens and to check the sizes")
    ap.add_argument("--top", "-t", type=int, default=10, help="show the N most frequent tokens outside the reduced vocabulary (default: 10, 0 = off)")
    ap.add_argument("--top-source", choices=common.SOURCES, default="generated",
                    help="which counts rank the tokens outside the reduced vocabulary (default: generated)")
    args = ap.parse_args()

    stats, files = common.load_stats(args.stats)
    reduced = common.load_reduced_ids(args.reduced)

    vocab = common.load_vocab(common.open_gguf(args.vocab)) if args.vocab else None
    if args.vocab and vocab is None:
        common.die(f"{args.vocab} has no tokenizer.ggml.tokens")
    n_vocab = vocab.n_vocab if vocab else max(len(stats["combined"]), int(reduced.max()) + 1 if len(reduced) else 0)
    stats = common.pad_stats(stats, n_vocab)
    if len(reduced) and (reduced.min() < 0 or reduced.max() >= n_vocab):
        common.die("the reduced vocabulary contains token ids outside the vocabulary")

    in_reduced = np.zeros(n_vocab, dtype=bool)
    in_reduced[reduced] = True

    print(f"statistics files   : {len(files)}")
    print(f"vocabulary         : {n_vocab} tokens")
    print(f"reduced vocabulary : {int(in_reduced.sum())} tokens ({pct(in_reduced.sum(), n_vocab).strip()} of the vocabulary)")
    print()

    header = f"{'':10}  {'token ids':>10}  {'not in reduced':>14}  {'by token id':>11}   {'tokens':>14}  {'not in reduced':>14}  {'by occurrence':>13}"
    print(header)
    for src in common.SOURCES:
        counts = stats[src]
        occurred = counts > 0
        n_ids = int(occurred.sum())
        n_ids_out = int((occurred & ~in_reduced).sum())
        total = int(counts.sum())
        total_out = int(counts[~in_reduced].sum())
        print(f"{src:10}  {n_ids:>10}  {n_ids_out:>14}  {pct(n_ids_out, n_ids):>11}   {total:>14}  {total_out:>14}  {pct(total_out, total):>13}")
    print()
    print("by token id    : share of the token ids that occurred which are not in the reduced vocabulary")
    print("by occurrence  : share of all the tokens (weighted by count) which are not in the reduced vocabulary")

    if args.top > 0:
        counts = stats[args.top_source]
        missing = np.flatnonzero(~in_reduced & (counts > 0))
        # most frequent first, lower token id first on ties
        missing = missing[np.lexsort((missing, -counts[missing].astype(np.int64)))][: args.top]
        total = int(counts.sum())
        print()
        print(f"top {len(missing)} tokens not in the reduced vocabulary ({args.top_source} counts)")
        print(f"{'token id':>10}  {'count':>12}  {'share':>9}  text")
        for t in missing:
            text = repr(vocab.text(int(t))) if vocab else ""
            print(f"{int(t):>10}  {int(counts[t]):>12}  {pct(int(counts[t]), total):>9}  {text}")


if __name__ == "__main__":
    main()
