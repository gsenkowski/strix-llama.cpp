# Reduced vocabulary for MTP draft heads

An MTP draft head scores every token of the vocabulary (a 250k x n_embd matrix multiplication for each drafted token).
Most tasks use a small part of the vocabulary, so a draft head restricted to the tokens that actually occur is much
cheaper. A token the reduced head cannot propose can never be drafted, which costs acceptance rate; verification is
unchanged, so the output does not change.

The workflow has three steps. The tools here need only `numpy` and the `gguf` package, not the inference engine.

## 1. Record token statistics

```
llama-server -m model.gguf ... --token-stats --token-stats-dir stats/
```

`--token-stats` also works for `llama-completion` (and `llama-cli`, which runs the server). It counts the tokens of the
prompts (prefill) and the tokens that came out of verification (decode), so the counts do not depend on the draft head or
on speculative decoding. One file per session, `token_stats_<UTC time>_<pid>_<random>.csv`:

```
token_id,prompt_count,generated_count
```

Tokens that did not occur are omitted. The file holds the cumulative counts of the session and is rewritten after an
inference when at least 10 minutes have passed since the last save, and once more at exit.

## 2. Check a vocabulary against statistics

```
./mtp_vocab_check.py stats/ --reduced mtp-reduced.gguf --vocab model.gguf
```

Several files, or directories of files, are summed up as one source. `--reduced` takes a reduced draft head GGUF or the
`.vocab.csv` next to it. For the prompt, generated and combined counts it prints the share of token ids and the share of
tokens (weighted by count) that are not in the reduced vocabulary, and with `--top N` (default 10) the most frequent
tokens that are missing. `--vocab` adds the token text.

## 3. Create a reduced draft head

```
./mtp_vocab_reduce.py stats/ --gguf mtp-model.gguf --coverage 95 -o mtp-model-95.gguf
```

The most frequent tokens are taken until they make up `--coverage` percent of the generated tokens (`--source prompt` or
`combined` to cover other counts). Control, user-defined and unknown tokens and the special tokens (BOS, EOS, ...) are
always kept. `--max-size N` caps the size, `--min-count C` never adds a token seen fewer than C times, `--round-to N`
rounds the size up to a multiple of N.

The input is the GGUF with the MTP block, e.g. the file written by `convert_hf_to_gguf.py --mtp`, which must contain the
LM head. The output is a copy of it with

* the tensor `blk.<N>.nextn.draft_head.weight`: the rows of the LM head of the reduced vocabulary. The rows are copied
  as they are, so a quantized head is not requantized.
* the key `<arch>.nextn_draft_vocab_ids`: the original token id of each row (the mapping back to the full vocabulary).

and a `.vocab.csv` (`reduced_id,token_id,text`). Use the output instead of the original MTP GGUF. The engine drafts with
the reduced head by default; `--spec-draft-mtp-vocab -1` drafts over the full vocabulary with the same file. The full head
stays in the file, so the file is larger than the original.

Supported architectures: `qwen35`, `qwen35moe`, whose MTP block scores with the LM head of the model.

## Tests

```
python3 -m unittest test_mtp_vocab.py
```

`tests/test-mtp-draft-head` checks the engine side on a reduced GGUF (`test-mtp-draft-head -m mtp-model-95.gguf`).
