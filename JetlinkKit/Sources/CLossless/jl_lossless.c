// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.

#include "jl_lossless.h"

#include <stdlib.h>

#include "vendor/zstd.h"

struct jl_lossless {
  ZSTD_DCtx *dctx;
  uint8_t *errors;
  size_t capacity;
};

jl_lossless *jl_lossless_create(void) {
  jl_lossless *ctx = calloc(1, sizeof(*ctx));
  if (ctx == NULL) return NULL;
  ctx->dctx = ZSTD_createDCtx();
  if (ctx->dctx == NULL) {
    free(ctx);
    return NULL;
  }
  return ctx;
}

void jl_lossless_free(jl_lossless *ctx) {
  if (ctx == NULL) return;
  ZSTD_freeDCtx(ctx->dctx);
  free(ctx->errors);
  free(ctx);
}

// fold's inverse (jetlink.lossless): 0, 1, 2, 3, 4 ... -> 0, -1, 1, -2, 2 ...
static inline int unfold(int z) { return (z & 1) ? -((z + 1) >> 1) : (z >> 1); }

static inline int med(int a, int b, int c) {
  int mx = a > b ? a : b, mn = a < b ? a : b;
  return c >= mx ? mn : c <= mn ? mx : a + b - c;
}

void jl_lossless_unmed(const uint8_t *errors, int h, int w, uint8_t *out) {
  // the first row predicts from the left, the first pixel from 0
  int a = 0;
  for (int j = 0; j < w; j++) {
    a = (a + unfold(errors[j])) & 255;
    out[j] = (uint8_t)a;
  }
  for (int i = 1; i < h; i++) {
    const uint8_t *e = errors + (size_t)i * w;
    const uint8_t *up = out + (size_t)(i - 1) * w;
    uint8_t *row = out + (size_t)i * w;
    // the first column predicts from above
    a = (up[0] + unfold(e[0])) & 255;
    row[0] = (uint8_t)a;
    for (int j = 1; j < w; j++) {
      a = (med(a, up[j], up[j - 1]) + unfold(e[j])) & 255;
      row[j] = (uint8_t)a;
    }
  }
}

int jl_lossless_plane(jl_lossless *ctx, const void *src, size_t n, int h, int w, uint8_t *out) {
  size_t size = (size_t)h * (size_t)w;
  if (ctx->capacity < size) {
    uint8_t *grown = realloc(ctx->errors, size);
    if (grown == NULL) return -1;
    ctx->errors = grown;
    ctx->capacity = size;
  }
  size_t got = ZSTD_decompressDCtx(ctx->dctx, ctx->errors, size, src, n);
  if (ZSTD_isError(got) || got != size) return -1;
  jl_lossless_unmed(ctx->errors, h, w, out);
  return 0;
}
