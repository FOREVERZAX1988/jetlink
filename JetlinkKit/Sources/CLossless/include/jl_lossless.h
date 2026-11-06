// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// A lossless frame's planes back to pixels (jetlink.lossless): each plane is
// its MED errors packed alone with zstd. Unpacking is zstd's; the MED inverse
// is sequential, each pixel from its decoded left, upper and upper-left
// neighbours, zero outside the plane.

#ifndef JL_LOSSLESS_H
#define JL_LOSSLESS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// A plane's decoder: a zstd context and the errors' scratch. One per thread.
typedef struct jl_lossless jl_lossless;

jl_lossless *jl_lossless_create(void);
void jl_lossless_free(jl_lossless *ctx);

/// One plane: `n` packed bytes at `src` back to its h*w pixels at `out`.
/// 0 when the plane is whole, -1 when it does not unpack to exactly h*w bytes.
int jl_lossless_plane(jl_lossless *ctx, const void *src, size_t n, int h, int w, uint8_t *out);

/// The MED inverse alone: folded errors (h*w) to pixels.
void jl_lossless_unmed(const uint8_t *errors, int h, int w, uint8_t *out);

#ifdef __cplusplus
}
#endif

#endif
