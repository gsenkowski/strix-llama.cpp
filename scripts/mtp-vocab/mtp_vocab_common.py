"""Shared helpers of the MTP vocabulary tools. Needs only numpy and the `gguf` package, not the inference engine."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from gguf import GGUFReader

STATS_HEADER = ["token_id", "prompt_count", "generated_count"]
SOURCES = ("prompt", "generated", "combined")

# key suffix of the reduced vocabulary stored in a GGUF: <arch>.nextn_draft_vocab_ids
IDS_KEY_SUFFIX = ".nextn_draft_vocab_ids"

# tokenizer.ggml.token_type values that are always part of a reduced vocabulary
TOKEN_TYPE_UNKNOWN = 2
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_USER_DEFINED = 4


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------------------------------------------------
# statistics files written by `--token-stats`
# ---------------------------------------------------------------------------------------------------------------------

def expand_stat_paths(paths: Iterable[str]) -> list[Path]:
    """Statistics files; a directory stands for all the .csv files in it."""
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(p.glob("*.csv")))
        elif p.is_file():
            files.append(p)
        else:
            die(f"statistics file not found: {p}")
    if not files:
        die("no statistics files given")
    return files


def load_stats(paths: Iterable[str]) -> tuple[dict[str, np.ndarray], list[Path]]:
    """Sum the counts of all files. Returns {'prompt', 'generated', 'combined'} -> uint64 array indexed by token id."""
    files = expand_stat_paths(paths)
    ids_all: list[np.ndarray] = []
    rows_all: list[np.ndarray] = []
    for f in files:
        with open(f, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                continue  # empty file
            if [h.strip() for h in header] != STATS_HEADER:
                die(f"{f}: unexpected header {header}, expected {STATS_HEADER}")
            rows = [r for r in reader if r]
        if not rows:
            continue
        try:
            arr = np.array(rows, dtype=np.int64)
        except ValueError as e:
            die(f"{f}: malformed row ({e})")
        if arr.ndim != 2 or arr.shape[1] != 3 or (arr < 0).any():
            die(f"{f}: malformed or negative values")
        ids_all.append(arr[:, 0])
        rows_all.append(arr[:, 1:])

    n = int(max((a.max() for a in ids_all), default=-1)) + 1
    prompt = np.zeros(n, dtype=np.uint64)
    gen = np.zeros(n, dtype=np.uint64)
    for ids, rows in zip(ids_all, rows_all):
        np.add.at(prompt, ids, rows[:, 0].astype(np.uint64))
        np.add.at(gen, ids, rows[:, 1].astype(np.uint64))
    return {"prompt": prompt, "generated": gen, "combined": prompt + gen}, files


def pad_stats(stats: dict[str, np.ndarray], n_vocab: int) -> dict[str, np.ndarray]:
    """Resize the count arrays to the vocabulary size; a token id outside the vocabulary is an error."""
    out = {}
    for k, v in stats.items():
        if len(v) > n_vocab:
            die(f"the statistics contain token id {len(v) - 1}, but the vocabulary has only {n_vocab} tokens "
                f"(statistics of a different model?)")
        out[k] = np.pad(v, (0, n_vocab - len(v)))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# GGUF vocabulary
# ---------------------------------------------------------------------------------------------------------------------

class VocabInfo:
    def __init__(self, tokens: list[str], types: list[int] | None, special_ids: dict[str, int]):
        self.tokens = tokens
        self.types = types
        self.special_ids = special_ids

    @property
    def n_vocab(self) -> int:
        return len(self.tokens)

    def text(self, token_id: int) -> str:
        return self.tokens[token_id] if 0 <= token_id < len(self.tokens) else ""

    def mandatory_ids(self) -> set[int]:
        """Tokens that must stay in every reduced vocabulary: control, user-defined, unknown and the special tokens."""
        ids: set[int] = set(self.special_ids.values())
        if self.types is not None:
            for i, t in enumerate(self.types):
                if t in (TOKEN_TYPE_CONTROL, TOKEN_TYPE_USER_DEFINED, TOKEN_TYPE_UNKNOWN):
                    ids.add(i)
        return {i for i in ids if 0 <= i < self.n_vocab}


def _field(reader: GGUFReader, key: str):
    f = reader.get_field(key)
    return None if f is None else f.contents()


def load_vocab(reader: GGUFReader) -> VocabInfo | None:
    tokens = _field(reader, "tokenizer.ggml.tokens")
    if tokens is None:
        return None
    types = _field(reader, "tokenizer.ggml.token_type")
    special: dict[str, int] = {}
    for key in reader.fields:
        if key.startswith("tokenizer.ggml.") and key.endswith("_token_id"):
            v = _field(reader, key)
            if isinstance(v, (int, np.integer)) and int(v) >= 0:
                special[key] = int(v)
    return VocabInfo(list(tokens), None if types is None else [int(t) for t in types], special)


def open_gguf(path: str) -> GGUFReader:
    if not Path(path).is_file():
        die(f"file not found: {path}")
    return GGUFReader(path, "r")


def reader_arch(reader: GGUFReader) -> str:
    arch = _field(reader, "general.architecture")
    if not isinstance(arch, str):
        die("the GGUF has no general.architecture")
    return arch


# ---------------------------------------------------------------------------------------------------------------------
# reduced vocabulary files
# ---------------------------------------------------------------------------------------------------------------------

def load_reduced_ids(path: str) -> np.ndarray:
    """Token ids of a reduced vocabulary, from a reduced GGUF or from the .vocab.csv written by mtp_vocab_reduce.py."""
    p = Path(path)
    if not p.is_file():
        die(f"reduced vocabulary not found: {p}")
    if p.suffix.lower() == ".gguf":
        reader = GGUFReader(str(p), "r")
        for key in reader.fields:
            if key.endswith(IDS_KEY_SUFFIX):
                return np.array(reader.get_field(key).contents(), dtype=np.int64)
        die(f"{p} does not contain a reduced vocabulary ({IDS_KEY_SUFFIX})")
    with open(p, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    if not rows:
        die(f"{p} is empty")
    header = [h.strip() for h in rows[0]]
    if "token_id" in header:
        col = header.index("token_id")
        rows = rows[1:]
    else:  # no header: one token id per line, or reduced_id,token_id
        col = 1 if len(rows[0]) > 1 else 0
    try:
        return np.array([int(r[col]) for r in rows], dtype=np.int64)
    except (ValueError, IndexError):
        die(f"{p}: cannot read token ids (expected a reduced_id,token_id,... CSV)")
