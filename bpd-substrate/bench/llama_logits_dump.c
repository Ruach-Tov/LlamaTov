/*
 * llama_logits_dump.c — Reference logits extractor for Level-3 harness.
 *
 * Companion tool to bench/bit_identical_whole_model.py. Loads a GGUF via
 * libllama, tokenizes a prompt, runs decode+generate, dumps per-step
 * logits to .npy-like binary files that the Python harness reads.
 *
 * Design note: this exists because llama_model_default_params() and
 * llama_context_default_params() return structs BY VALUE, which is
 * painful to call correctly from ctypes across ABI boundaries. A small
 * C companion sidesteps the by-value struct problem entirely.
 *
 * Build:
 *   gcc -O2 -o /tmp/llama_logits_dump bench/llama_logits_dump.c \
 *       -I external/llama.cpp/include \
 *       -L external/llama.cpp/build/bin \
 *       -Wl,-rpath,$(pwd)/external/llama.cpp/build/bin \
 *       -lllama
 *
 * Usage:
 *   llama_logits_dump --gguf <path> --tokens <csv> --n-generate <N> \
 *                     --out-prefix <prefix>
 *
 * Output: <prefix>_step0.bin, <prefix>_step1.bin, ...
 * Each file: int32 vocab_size + vocab_size × float32 logits.
 *
 * Scope (per scout-boundary): measures only, modifies nothing.
 */

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"


static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --gguf <path> --tokens <csv> --n-generate <N> --out-prefix <prefix>\n"
        "  --gguf         Path to GGUF model file (required)\n"
        "  --tokens       Comma-separated pre-tokenized input (required)\n"
        "  --n-generate   Number of tokens to generate (default: 4)\n"
        "  --out-prefix   Path prefix for output files (required)\n"
        "\n"
        "Output: <prefix>_step0.bin, <prefix>_step1.bin, ...\n"
        "  Each: int32 vocab_size + vocab_size × float32 logits\n",
        argv0);
}


static int parse_tokens_csv(const char *csv, int32_t **out_toks, int *out_n) {
    /* Count commas, allocate, parse. Returns 0 on success. */
    int n = 1;
    for (const char *p = csv; *p; p++) if (*p == ',') n++;
    int32_t *toks = calloc(n, sizeof(int32_t));
    if (!toks) return -1;
    char *dup = strdup(csv);
    if (!dup) { free(toks); return -1; }
    int i = 0;
    for (char *tok = strtok(dup, ","); tok && i < n; tok = strtok(NULL, ",")) {
        toks[i++] = (int32_t)strtol(tok, NULL, 10);
    }
    free(dup);
    *out_toks = toks;
    *out_n = i;
    return 0;
}


static int write_logits_step(const char *prefix, int step,
                             int32_t vocab_size, const float *logits) {
    char path[1024];
    snprintf(path, sizeof(path), "%s_step%d.bin", prefix, step);
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "  [ref] fopen(%s): %s\n", path, strerror(errno));
        return -1;
    }
    if (fwrite(&vocab_size, sizeof(int32_t), 1, f) != 1 ||
        fwrite(logits, sizeof(float), vocab_size, f) != (size_t)vocab_size) {
        fprintf(stderr, "  [ref] fwrite(%s): %s\n", path, strerror(errno));
        fclose(f);
        return -1;
    }
    fclose(f);
    fprintf(stdout, "  [ref] wrote %s (vocab=%d)\n", path, vocab_size);
    return 0;
}


int main(int argc, char **argv) {
    const char *gguf_path = NULL;
    const char *tokens_csv = NULL;
    const char *out_prefix = NULL;
    int n_generate = 4;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gguf") == 0 && i + 1 < argc) {
            gguf_path = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            tokens_csv = argv[++i];
        } else if (strcmp(argv[i], "--n-generate") == 0 && i + 1 < argc) {
            n_generate = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--out-prefix") == 0 && i + 1 < argc) {
            out_prefix = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    if (!gguf_path || !tokens_csv || !out_prefix) {
        fprintf(stderr, "missing required arg\n");
        usage(argv[0]);
        return 2;
    }

    /* Parse tokens */
    int32_t *prompt_toks = NULL;
    int n_prompt = 0;
    if (parse_tokens_csv(tokens_csv, &prompt_toks, &n_prompt) != 0) {
        fprintf(stderr, "parse_tokens_csv failed\n");
        return 1;
    }
    fprintf(stdout, "  [ref] parsed %d prompt tokens\n", n_prompt);

    /* llama.cpp init */
    llama_backend_init();

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;  /* CPU-only for bit-identity comparison */
    mp.use_mmap = true;

    struct llama_model *model = llama_model_load_from_file(gguf_path, mp);
    if (!model) {
        fprintf(stderr, "  [ref] llama_model_load_from_file(%s) failed\n", gguf_path);
        llama_backend_free();
        free(prompt_toks);
        return 1;
    }

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_prompt + n_generate + 16;  /* small headroom */
    cp.n_batch = cp.n_ctx;
    cp.n_ubatch = cp.n_ctx;
    cp.n_threads = 1;   /* deterministic single-thread for bit-identity */
    cp.n_threads_batch = 1;

    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "  [ref] llama_init_from_model failed\n");
        llama_model_free(model);
        llama_backend_free();
        free(prompt_toks);
        return 1;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    fprintf(stdout, "  [ref] loaded model: n_vocab=%d\n", n_vocab);

    /* Prefill: submit prompt tokens */
    struct llama_batch batch = llama_batch_get_one(prompt_toks, n_prompt);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "  [ref] prefill llama_decode failed\n");
        goto cleanup;
    }

    /* Generate + dump logits per step */
    int32_t last_token = prompt_toks[n_prompt - 1];
    int rc = 0;
    for (int step = 0; step < n_generate; step++) {
        /* Get last-position logits */
        const float *logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            fprintf(stderr, "  [ref] llama_get_logits_ith(-1) returned NULL at step %d\n", step);
            rc = 1;
            break;
        }

        /* Dump */
        if (write_logits_step(out_prefix, step, n_vocab, logits) != 0) {
            rc = 1;
            break;
        }

        /* Greedy argmax for next token */
        int32_t argmax = 0;
        float best = logits[0];
        for (int32_t v = 1; v < n_vocab; v++) {
            if (logits[v] > best) { best = logits[v]; argmax = v; }
        }
        last_token = argmax;
        fprintf(stdout, "  [ref] step %d: argmax=%d (logit=%.6f)\n", step, argmax, best);

        /* Submit next single-token batch */
        struct llama_batch next_batch = llama_batch_get_one(&last_token, 1);
        if (llama_decode(ctx, next_batch) != 0) {
            fprintf(stderr, "  [ref] step %d llama_decode failed\n", step);
            rc = 1;
            break;
        }
    }

cleanup:
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    free(prompt_toks);
    return rc;
}
