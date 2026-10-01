#!/usr/bin/env python3
"""Tests of the MTP vocabulary tools on a synthetic GGUF. Run: python3 -m unittest test_mtp_vocab.py"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf import GGUFReader, GGUFWriter
from gguf.quants import quantize

import mtp_vocab_common as common
import mtp_vocab_reduce as reduce_tool

HERE = Path(__file__).parent
N_VOCAB, N_EMBD = 64, 32


def make_gguf(path: Path, head_type: Q) -> np.ndarray:
    rng = np.random.default_rng(0)
    head = rng.standard_normal((N_VOCAB, N_EMBD)).astype(np.float32)
    w = GGUFWriter(str(path), "qwen35")
    w.add_block_count(3)
    w.add_uint32("qwen35.nextn_predict_layers", 1)
    types = [1] * N_VOCAB
    types[0] = types[1] = 3
    types[2] = 4
    w.add_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(N_VOCAB)])
    w.add_array("tokenizer.ggml.token_type", types)
    w.add_uint32("tokenizer.ggml.eos_token_id", 1)
    if head_type == Q.F32:
        w.add_tensor("output.weight", head)
    else:
        w.add_tensor("output.weight", quantize(head, head_type), raw_dtype=head_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return head


def write_stats(path: Path, prompt: np.ndarray, gen: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("token_id,prompt_count,generated_count\n")
        for t in range(len(gen)):
            if prompt[t] or gen[t]:
                f.write(f"{t},{prompt[t]},{gen[t]}\n")


class Tools(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        p = 1 / np.arange(1, N_VOCAB + 1)
        self.gen = np.random.default_rng(1).multinomial(10000, p / p.sum())
        self.prompt = np.random.default_rng(2).multinomial(3000, p[::-1] / p.sum())
        (self.tmp / "stats").mkdir()
        # two files that are used as one source
        write_stats(self.tmp / "stats/a.csv", self.prompt // 2, self.gen // 2)
        write_stats(self.tmp / "stats/b.csv", self.prompt - self.prompt // 2, self.gen - self.gen // 2)

    def tearDown(self):
        self._tmp.cleanup()

    def test_stats_are_summed(self):
        stats, files = common.load_stats([str(self.tmp / "stats")])
        self.assertEqual(len(files), 2)
        np.testing.assert_array_equal(stats["generated"][: len(self.gen)], self.gen)
        np.testing.assert_array_equal(stats["combined"][: len(self.gen)], self.gen + self.prompt)

    def test_selection_reaches_coverage_with_fewest_tokens(self):
        mandatory = {0, 1, 2}
        for cov in (50, 80, 95, 100):
            ids = reduce_tool.select_tokens(self.gen.astype(np.uint64), mandatory, cov, 0, 0, 1)
            got = self.gen[ids].sum() / self.gen.sum() * 100
            self.assertGreaterEqual(got + 1e-9, cov)
            self.assertTrue(mandatory <= set(ids.tolist()))
            self.assertEqual(list(ids), sorted(ids))
            if len(ids) > len(mandatory):
                # dropping the least frequent non-mandatory token would fall below the target
                rest = [i for i in ids if i not in mandatory]
                worst = min(rest, key=lambda i: (self.gen[i], -i))
                self.assertLess((self.gen[ids].sum() - self.gen[worst]) / self.gen.sum() * 100, cov)

    def test_limits(self):
        counts = self.gen.astype(np.uint64)
        ids = reduce_tool.select_tokens(counts, {0, 1, 2}, 100, 0, 20, 1)
        self.assertEqual(len(ids), 20)
        ids = reduce_tool.select_tokens(counts, {0, 1, 2}, 100, 100, 0, 1)
        self.assertTrue(all(counts[i] >= 100 or i < 3 for i in ids))
        ids = reduce_tool.select_tokens(counts, {0, 1, 2}, 50, 0, 0, 16)
        self.assertEqual(len(ids) % 16, 0)

    def run_tool(self, script, *args):
        r = subprocess.run([sys.executable, str(HERE / script), *map(str, args)], capture_output=True, text=True, cwd=HERE)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_reduce_f32_and_quantized_heads(self):
        for head_type in (Q.F32, Q.Q8_0):
            src = self.tmp / f"mtp-{head_type.name}.gguf"
            out = self.tmp / f"red-{head_type.name}.gguf"
            make_gguf(src, head_type)
            self.run_tool("mtp_vocab_reduce.py", self.tmp / "stats", "-m", src, "-c", 90, "-o", out)

            r_in, r_out = GGUFReader(str(src)), GGUFReader(str(out))
            ids = np.array(r_out.get_field("qwen35.nextn_draft_vocab_ids").contents())
            self.assertTrue((np.diff(ids) > 0).all())
            t_in = next(t for t in r_in.tensors if t.name == "output.weight")
            t_out = {t.name: t for t in r_out.tensors}
            # the reduced head is the selected rows of the original, bit for bit
            red = t_out["blk.2.nextn.draft_head.weight"]
            self.assertEqual(red.tensor_type, head_type)
            self.assertEqual([int(x) for x in red.shape], [N_EMBD, len(ids)])
            np.testing.assert_array_equal(np.asarray(red.data), np.asarray(t_in.data)[ids])
            # everything of the input is still there, unchanged
            self.assertEqual(set(t_out) - {"blk.2.nextn.draft_head.weight"}, {t.name for t in r_in.tensors})
            np.testing.assert_array_equal(np.asarray(t_out["output.weight"].data), np.asarray(t_in.data))
            self.assertEqual(r_out.get_field("tokenizer.ggml.eos_token_id").contents(), 1)
            self.assertEqual(len(r_out.get_field("tokenizer.ggml.tokens").contents()), N_VOCAB)

            # the vocab.csv holds the same mapping
            np.testing.assert_array_equal(common.load_reduced_ids(str(out.with_suffix(".vocab.csv"))), ids)
            np.testing.assert_array_equal(common.load_reduced_ids(str(out)), ids)

    def test_check_tool(self):
        src, out = self.tmp / "mtp.gguf", self.tmp / "red.gguf"
        make_gguf(src, Q.F32)
        self.run_tool("mtp_vocab_reduce.py", self.tmp / "stats", "-m", src, "-c", 90, "-o", out)
        ids = common.load_reduced_ids(str(out))
        text = self.run_tool("mtp_vocab_check.py", self.tmp / "stats", "-r", out, "-v", src, "--top", 3)
        missing = np.ones(N_VOCAB, dtype=bool)
        missing[ids] = False
        line = next(l for l in text.splitlines() if l.startswith("generated"))
        self.assertIn(f"{self.gen[missing].sum()}", line)
        self.assertIn(f"{missing[self.gen > 0].sum()}", line)
        self.assertEqual(len([l for l in text.split("top 3")[1].splitlines() if l.strip().startswith(tuple("0123456789"))]), 3)


if __name__ == "__main__":
    unittest.main()
