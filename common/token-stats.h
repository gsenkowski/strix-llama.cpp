#pragma once

#include "llama.h"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

// Counts how often each token of the vocabulary occurs in the prompts (prefill) and in the verified output (decode) of a
// session and saves the counts as CSV (token_id,prompt_count,generated_count; tokens that never occurred are omitted).
//
// Only tokens that are final are counted - the prompt and the tokens that came out of verification - so the result does
// not depend on the draft head or on speculative decoding at all.
//
// Each session writes to its own file in the output directory. The file is rewritten with the cumulative counts, at most
// once per interval: maybe_save() is meant to be called once after each finished inference and checks the timer, save()
// writes unconditionally (final flush).
class common_token_stats {
public:
    common_token_stats(int32_t n_vocab, const std::string & dir, int64_t interval_s = 600);

    void add_prompt(const llama_token * tokens, size_t n);
    void add_generated(llama_token token);

    // save if the interval has elapsed since the last save (or since the start) and there is something new
    void maybe_save();

    // save now if there is something new
    void save();

    const std::string & path() const { return path_; }

private:
    bool save_locked();

    using clock = std::chrono::steady_clock;

    std::mutex                mtx_;
    std::vector<uint64_t>     n_prompt_;
    std::vector<uint64_t>     n_gen_;
    std::string               path_;
    clock::duration           interval_;
    clock::time_point         t_last_save_;
    bool                      dirty_ = false;
};
