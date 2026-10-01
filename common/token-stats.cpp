#include "token-stats.h"

#include "log.h"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <random>
#include <system_error>

#if defined(_WIN32)
#   include <process.h>
#   define TOKEN_STATS_GETPID _getpid
#else
#   include <unistd.h>
#   define TOKEN_STATS_GETPID getpid
#endif

// token_stats_<UTC date and time>_<pid>_<random>.csv - unique per session, even for sessions started in the same second
static std::string token_stats_file_name() {
    const std::time_t now = std::time(nullptr);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &now);
#else
    gmtime_r(&now, &tm_utc);
#endif
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%dT%H%M%SZ", &tm_utc);

    std::random_device rd;
    char name[96];
    std::snprintf(name, sizeof(name), "token_stats_%s_%d_%06x.csv", ts, (int) TOKEN_STATS_GETPID(), (unsigned) (rd() & 0xffffff));
    return name;
}

common_token_stats::common_token_stats(int32_t n_vocab, const std::string & dir, int64_t interval_s) :
    n_prompt_((size_t) std::max(n_vocab, 0), 0),
    n_gen_   ((size_t) std::max(n_vocab, 0), 0),
    interval_(std::chrono::seconds(interval_s)),
    t_last_save_(clock::now()) {
    const std::filesystem::path d = dir.empty() ? std::filesystem::path(".") : std::filesystem::path(dir);
    std::error_code ec;
    std::filesystem::create_directories(d, ec);
    if (ec) {
        LOG_WRN("%s: cannot create '%s': %s\n", __func__, d.string().c_str(), ec.message().c_str());
    }
    path_ = (d / token_stats_file_name()).string();
    LOG_INF("%s: recording token statistics to '%s'\n", __func__, path_.c_str());
}

void common_token_stats::add_prompt(const llama_token * tokens, size_t n) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (size_t i = 0; i < n; ++i) {
        const llama_token t = tokens[i];
        if (t >= 0 && (size_t) t < n_prompt_.size()) {
            n_prompt_[(size_t) t]++;
        }
    }
    dirty_ = dirty_ || n > 0;
}

void common_token_stats::add_generated(llama_token token) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (token >= 0 && (size_t) token < n_gen_.size()) {
        n_gen_[(size_t) token]++;
        dirty_ = true;
    }
}

void common_token_stats::maybe_save() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (clock::now() - t_last_save_ >= interval_) {
        save_locked();
    }
}

void common_token_stats::save() {
    std::lock_guard<std::mutex> lock(mtx_);
    save_locked();
}

bool common_token_stats::save_locked() {
    t_last_save_ = clock::now();
    if (!dirty_) {
        return true;
    }

    // write next to the target and rename, so that a crash never leaves a half-written file behind
    const std::string tmp = path_ + ".tmp";
    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        LOG_WRN("%s: cannot write '%s'\n", __func__, tmp.c_str());
        return false;
    }
    std::fputs("token_id,prompt_count,generated_count\n", f);
    for (size_t t = 0; t < n_gen_.size(); ++t) {
        if (n_prompt_[t] != 0 || n_gen_[t] != 0) {
            std::fprintf(f, "%zu,%llu,%llu\n", t, (unsigned long long) n_prompt_[t], (unsigned long long) n_gen_[t]);
        }
    }
    const bool ok = std::fclose(f) == 0;

    std::error_code ec;
    if (ok) {
        std::filesystem::rename(tmp, path_, ec);
    }
    if (!ok || ec) {
        LOG_WRN("%s: failed to save '%s'\n", __func__, path_.c_str());
        std::filesystem::remove(tmp, ec);
        return false;
    }
    dirty_ = false;
    return true;
}
