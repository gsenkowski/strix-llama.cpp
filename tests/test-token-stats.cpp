// Tests of common_token_stats (--token-stats): CSV content, one unique file per session, the save interval.

#include "token-stats.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

namespace fs = std::filesystem;

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static size_t n_files(const fs::path & dir) {
    size_t n = 0;
    for (const auto & e : fs::directory_iterator(dir)) {
        GGML_UNUSED(e);
        n++;
    }
    return n;
}

int main() {
    const fs::path dir = fs::temp_directory_path() / ("test-token-stats-" + std::to_string(std::rand()));
    fs::remove_all(dir);

    // counts and CSV: tokens that did not occur are omitted, out-of-range ids are ignored
    {
        common_token_stats stats(16, (dir / "a/b").string(), 3600);
        const llama_token prompt[] = { 3, 3, 5 };
        stats.add_prompt(prompt, 3);
        stats.add_generated(5);
        stats.add_generated(7);
        stats.add_generated(7);
        stats.add_generated(-1);
        stats.add_generated(16);

        // the interval has not elapsed: nothing is written
        stats.maybe_save();
        CHECK(!fs::exists(stats.path()));

        stats.save();
        CHECK(fs::exists(stats.path()));
        CHECK(read_file(stats.path()) ==
            "token_id,prompt_count,generated_count\n"
            "3,2,0\n"
            "5,1,1\n"
            "7,0,2\n");

        // cumulative: later counts are added to the same file
        stats.add_generated(3);
        stats.save();
        CHECK(read_file(stats.path()) ==
            "token_id,prompt_count,generated_count\n"
            "3,2,1\n"
            "5,1,1\n"
            "7,0,2\n");
        CHECK(n_files(dir / "a/b") == 1); // no temporary file left behind
    }

    // an interval of 0 saves on every maybe_save(), and only if there is something new
    {
        common_token_stats stats(16, (dir / "c").string(), 0);
        stats.maybe_save();
        CHECK(!fs::exists(stats.path()));
        stats.add_generated(1);
        stats.maybe_save();
        CHECK(fs::exists(stats.path()));
        fs::remove(stats.path());
        stats.maybe_save(); // nothing new
        CHECK(!fs::exists(stats.path()));
    }

    // sessions started at the same time write different files
    {
        std::set<std::string> paths;
        for (int i = 0; i < 20; ++i) {
            common_token_stats stats(16, (dir / "d").string(), 3600);
            paths.insert(stats.path());
        }
        CHECK(paths.size() == 20);
    }

    fs::remove_all(dir);
    fprintf(stderr, "PASS\n");
    return 0;
}
