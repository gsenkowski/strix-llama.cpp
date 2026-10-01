// Regression test for the reduced MTP draft head stored in a GGUF (tensor blk.N.nextn.draft_head.weight plus the key
// <arch>.nextn_draft_vocab_ids, written by scripts/mtp-vocab/mtp_vocab_reduce.py).
//
// Needs a qwen35 or qwen35moe model with an MTP head whose MTP block uses the model LM head, converted with
// mtp_vocab_reduce.py:
//
//   test-mtp-draft-head -m mtp-reduced.gguf
//
// A context with mtp_draft_vocab = 0 must use the reduced head: -inf exactly at the token ids that are not in
// <arch>.nextn_draft_vocab_ids, and finite logits that match those of a full-vocabulary context (mtp_draft_vocab = -1)
// fed the same inputs. The argmax of the reduced context must therefore be the full-vocabulary argmax whenever that
// token is in the reduced vocabulary, mapped back to its original token id.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../src/llama-ext.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static llama_context_ptr make_ctx(llama_model * model, int32_t mtp_draft_vocab) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 256;
    cparams.n_batch         = 16;
    cparams.n_ubatch        = 16;
    cparams.n_seq_max       = 1;
    cparams.ctx_type        = LLAMA_CONTEXT_TYPE_MTP;
    cparams.mtp_draft_vocab = mtp_draft_vocab;
    cparams.n_rs_seq        = 0;
    llama_context_ptr ctx(llama_init_from_model(model, cparams));
    if (ctx) {
        llama_set_embeddings_nextn(ctx.get(), true, /*masked*/ true);
    }
    return ctx;
}

// the token ids of the reduced vocabulary, from the key <arch>.nextn_draft_vocab_ids of the GGUF
static std::vector<int32_t> read_reduced_ids(const std::string & path) {
    std::vector<int32_t> ids;
    gguf_init_params ip = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), ip);
    if (!gctx) {
        return ids;
    }
    for (int64_t k = 0; k < gguf_get_n_kv(gctx); ++k) {
        const std::string key = gguf_get_key(gctx, k);
        const std::string suffix = ".nextn_draft_vocab_ids";
        if (key.size() > suffix.size() && key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0 &&
                gguf_get_kv_type(gctx, k) == GGUF_TYPE_ARRAY) {
            const int32_t * data = (const int32_t *) gguf_get_arr_data(gctx, k);
            ids.assign(data, data + gguf_get_arr_n(gctx, k));
        }
    }
    gguf_free(gctx);
    return ids;
}

// decode one token with a pseudo-random hidden state at position pos; return the draft logits
static bool decode_step(llama_model * model, llama_context * ctx, llama_pos pos, std::vector<float> & out) {
    const int32_t n_embd  = llama_model_n_embd_out(model);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_batch batch = llama_batch_init(1, n_embd, 1);
    batch.token = (llama_token *) malloc(sizeof(llama_token));

    std::mt19937 rng(1000 + pos);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int32_t i = 0; i < n_embd; ++i) {
        batch.embd[i] = dist(rng);
    }
    batch.token[0]     = (llama_token) ((pos * 7919 + 13) % n_vocab);
    batch.pos[0]       = pos;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]    = 1;
    batch.n_tokens     = 1;

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        fprintf(stderr, "llama_decode failed: %d\n", rc);
        return false;
    }
    const float * logits = llama_get_logits_ith(ctx, -1);
    out.assign(logits, logits + n_vocab);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    ggml_backend_load_all();

    const std::vector<int32_t> reduced = read_reduced_ids(params.model.path);
    if (reduced.empty()) {
        fprintf(stderr, "%s has no reduced draft vocabulary, skipping\n", params.model.path.c_str());
        return 0;
    }

    llama_model_params mparams = common_model_params_to_llama(params);
    mparams.load_mtp = true;
    llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mparams));
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    if (llama_model_n_layer_nextn(model.get()) == 0) {
        fprintf(stderr, "model has no MTP layers, skipping\n");
        return 0;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    std::vector<uint8_t> in_reduced((size_t) n_vocab, 0);
    for (int32_t id : reduced) {
        if (id < 0 || id >= n_vocab) {
            fprintf(stderr, "FAIL: reduced vocabulary id %d is outside the vocabulary\n", id);
            return 1;
        }
        in_reduced[(size_t) id] = 1;
    }

    llama_context_ptr ctx_red  = make_ctx(model.get(), 0);  // the reduced head of the GGUF
    llama_context_ptr ctx_full = make_ctx(model.get(), -1); // the full vocabulary
    if (!ctx_red || !ctx_full) {
        fprintf(stderr, "failed to create contexts\n");
        return 1;
    }

    bool ok = true;
    for (int step = 0; step < 4; ++step) {
        std::vector<float> red, full;
        if (!decode_step(model.get(), ctx_red.get(), step, red) || !decode_step(model.get(), ctx_full.get(), step, full)) {
            return 1;
        }
        size_t n_bad_inf = 0, n_bad_fin = 0, n_bad_val = 0;
        double max_diff = 0.0;
        for (int32_t t = 0; t < n_vocab; ++t) {
            if (in_reduced[(size_t) t]) {
                if (!std::isfinite(red[(size_t) t])) {
                    n_bad_inf++;
                    continue;
                }
                const double d = std::fabs((double) red[(size_t) t] - (double) full[(size_t) t]);
                max_diff = std::max(max_diff, d);
                if (d > 1e-2 * std::max(1.0, std::fabs((double) full[(size_t) t]))) {
                    n_bad_val++;
                }
            } else if (!(std::isinf(red[(size_t) t]) && red[(size_t) t] < 0)) {
                n_bad_fin++;
            }
            if (!std::isfinite(full[(size_t) t])) {
                n_bad_val++;
            }
        }
        // the best token of the reduced head is the best token of the full vocabulary restricted to the reduced one
        int32_t best_red = -1, best_full_in_red = -1;
        for (int32_t t = 0; t < n_vocab; ++t) {
            if (best_red < 0 || red[(size_t) t] > red[(size_t) best_red]) {
                best_red = t;
            }
            if (in_reduced[(size_t) t] && (best_full_in_red < 0 || full[(size_t) t] > full[(size_t) best_full_in_red])) {
                best_full_in_red = t;
            }
        }
        const bool same_best = best_red == best_full_in_red;
        const bool step_ok = n_bad_inf == 0 && n_bad_fin == 0 && n_bad_val == 0 && same_best;
        fprintf(stderr, "step %d: %s draftable %zu/%d, non-finite inside %zu, finite outside %zu, mismatches vs full vocab %zu "
                "(max |diff| %.3g), argmax %d (full vocab restricted: %d)\n", step, step_ok ? "OK  " : "FAIL",
                reduced.size(), n_vocab, n_bad_inf, n_bad_fin, n_bad_val, max_diff, best_red, best_full_in_red);
        ok &= step_ok;
    }

    fprintf(stderr, "%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
