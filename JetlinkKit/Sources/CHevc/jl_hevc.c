// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// An HEVC access unit back to the six warped planes (jl_hevc.h).
//
// The bitstream half is plain C and portable: where the NAL units are in an
// Annex-B access unit, whether one of them starts a sequence, and how an
// NV12 frame unpacks into the six planes the model was built for. Those are
// what the Python's jetlink.hevc does, and a test can check them without a
// decoder.
//
// The decoder half is libavcodec, opened at run time (JL_HEVC_DLOPEN, as
// COrt does with JL_ORT_DLOPEN), so a build needs no ffmpeg headers and a
// host without libavcodec loaded reports no decoder. That is what keeps the
// Python from offering 'hevc' to a server that cannot take it.
//
// One access unit in, one NV12 frame out: the frame path is strictly in
// order and no reorder buffer is left on.

#include "jl_hevc.h"

#include <stdlib.h>
#include <string.h>

#if defined(JL_HEVC_DLOPEN)
#define JL_HEVC_AV 1
#endif

// -- the bitstream ------------------------------------------------------------

size_t jl_hevc_nals(const uint8_t *au, size_t n, jl_nal *out, size_t max) {
  if (au == NULL || out == NULL || max == 0) return 0;
  size_t found = 0;
  size_t i = 0;
  while (i + 3 < n && found < max) {
    size_t header;
    if (au[i] == 0 && au[i + 1] == 0 && au[i + 2] == 1) {
      header = i + 3;
    } else if (au[i] == 0 && au[i + 1] == 0 && au[i + 2] == 0 && au[i + 3] == 1) {
      header = i + 4;
    } else {
      i++;
      continue;
    }
    // the next start code ends this one
    size_t end = n;
    for (size_t j = header; j + 3 < n; j++) {
      if ((au[j] == 0 && au[j + 1] == 0 && au[j + 2] == 1) ||
          (au[j] == 0 && au[j + 1] == 0 && au[j + 2] == 0 && au[j + 3] == 1)) {
        end = j;
        break;
      }
    }
    if (header < end) {
      out[found].data = au + header;
      out[found].size = end - header;
      out[found].type = (au[header] >> 1) & 0x3F;
      found++;
    }
    i = end > header ? end : header;
  }
  return found;
}

int jl_hevc_is_keyframe(const uint8_t *au, size_t n) {
  jl_nal nals[64];
  size_t count = jl_hevc_nals(au, n, nals, 64);
  for (size_t i = 0; i < count; i++) {
    // 16..21 are IRAP, the ones a decoder can start from. Within them, 16
    // (BLA), 18 (reserved), 19 (IDR) and 20 (CRA) are; 17 and 21 are not.
    int type = nals[i].type;
    if (type == 16 || type == 18 || type == 19 || type == 20) return 1;
  }
  return 0;
}

size_t jl_hevc_frame_bytes(int width, int height) {
  // the NV12 frame's own size: a luma of width x height and a chroma of
  // half both ways, interleaved
  if (width <= 0 || height <= 0 || (width & 1) || (height & 1)) return 0;
  return (size_t)width * (size_t)height + (size_t)width * (size_t)height / 2;
}

// The six warped planes (2, 6, H, W) out of an NV12 frame: the luma's four
// phases and the chroma's two planes.
//
// The luma is twice as wide and twice as tall as one plane, and the four
// phases are its (even, even), (even, odd), (odd, even) and (odd, odd)
// pixels: a warp that put four sub-sampled images side by side, and this
// puts them back where each belongs. The chroma is half resolution in both
// directions and interleaved U and V, so it is one plane's worth of each.
//
// `width` and `height` are the NV12 frame's own, so one plane is
// (height / 2, width / 2). The Python is `jetlink.hevc.from_nv12`.
void jl_hevc_unpack_nv12(const uint8_t *nv12, int width, int height, uint8_t *out) {
  if (nv12 == NULL || out == NULL) return;
  size_t w = (size_t)width / 2;    // one plane's width
  size_t h = (size_t)height / 2;   // and its height
  if (w == 0 || h == 0) return;
  size_t plane = w * h;
  // the luma, as the (2H, 2W) it is inside the frame
  for (size_t dy = 0; dy < 2; dy++) {
    for (size_t dx = 0; dx < 2; dx++) {
      uint8_t *dst = out + ((dy * 2 + dx) * plane);
      for (size_t y = 0; y < h; y++) {
        const uint8_t *src = nv12 + (2 * y + dy) * (2 * w) + dx;
        for (size_t x = 0; x < w; x++) {
          dst[y * w + x] = src[2 * x];
        }
      }
    }
  }
  // the chroma, interleaved U and V
  const uint8_t *chroma = nv12 + 4 * plane;
  for (size_t y = 0; y < h; y++) {
    const uint8_t *row = chroma + y * w * 2;
    uint8_t *u = out + 4 * plane + y * w;
    uint8_t *v = out + 5 * plane + y * w;
    for (size_t x = 0; x < w; x++) {
      u[x] = row[2 * x];
      v[x] = row[2 * x + 1];
    }
  }
}

// -- the decoder --------------------------------------------------------------

#ifdef JL_HEVC_AV

#include <dlfcn.h>

// libavcodec's few entry points, named as its headers name them. The library
// is opened at run time, so the build needs none of libavcodec's own
// headers and a host without it simply has no decoder.
//
// The types below are libavcodec's, laid out as its headers lay them out:
// the packet and the frame are plain structs whose fields this code reads
// and writes. Only the fields used here are named.
typedef struct jl_av_packet {
  uint8_t *data;
  int size;
  int stream_index;
  int flags;
  int64_t pts;
  int64_t dts;
} jl_av_packet;

typedef struct jl_av_frame {
  uint8_t *data[8];
  int linesize[8];
  uint8_t **extended_data;
  int width;
  int height;
  int format;
  int key_frame;
} jl_av_frame;

typedef struct jl_av_codec_ctx jl_av_codec_ctx;
typedef struct jl_av_codec jl_av_codec;

typedef struct {
  void *handle;
  jl_av_codec *(*find_by_name)(const char *);
  jl_av_codec_ctx *(*alloc_context)(const jl_av_codec *);
  int (*open)(jl_av_codec_ctx *, const jl_av_codec *);
  int (*send_packet)(jl_av_codec_ctx *, const jl_av_packet *);
  int (*receive_frame)(jl_av_codec_ctx *, jl_av_frame *);
  void (*free_context)(jl_av_codec_ctx *);
  void (*free_frame)(jl_av_frame *);
  const char *(*name)(const jl_av_codec *);
} jl_av_api;

struct jl_hevc {
  int width;
  int height;
  size_t frame_bytes;
  int broken;   // fed a non-keyframe with no reference: waiting for an IRAP
  jl_av_api api;
  jl_av_codec *codec;
  jl_av_codec_ctx *context;
  jl_av_frame frame;
  uint8_t *nv12;   // the frame, NV12, copied out of the decoder's
  int opened;
};

// The library, opened once and kept: dlopen on every frame would cost more
// than the decode. NULL when there is no library, or no decoder in it.
static jl_av_api *jl_av(void) {
  static jl_av_api api;
  static int tried;
  if (tried) return api.handle != NULL ? &api : NULL;
  tried = 1;
  void *handle = dlopen("libavcodec.so", RTLD_NOW | RTLD_LOCAL);
  if (handle == NULL) handle = dlopen("libavcodec.dylib", RTLD_NOW | RTLD_LOCAL);
  if (handle == NULL) return NULL;
  api.handle = handle;
  api.find_by_name = (jl_av_codec *(*)(const char *))dlsym(handle, "avcodec_find_decoder_by_name");
  api.alloc_context = (jl_av_codec_ctx *(*)(const jl_av_codec *))dlsym(handle, "avcodec_alloc_context3");
  api.open = (int (*)(jl_av_codec_ctx *, const jl_av_codec *))dlsym(handle, "avcodec_open2");
  api.send_packet = (int (*)(jl_av_codec_ctx *, const jl_av_packet *))dlsym(handle, "avcodec_send_packet");
  api.receive_frame = (int (*)(jl_av_codec_ctx *, jl_av_frame *))dlsym(handle, "avcodec_receive_frame");
  api.free_context = (void (*)(jl_av_codec_ctx *))dlsym(handle, "avcodec_free_context");
  api.free_frame = (void (*)(jl_av_frame *))dlsym(handle, "av_frame_free");
  api.name = (const char *(*)(const jl_av_codec *))dlsym(handle, "avcodec_get_name");
  if (api.find_by_name == NULL || api.alloc_context == NULL || api.open == NULL ||
      api.send_packet == NULL || api.receive_frame == NULL) {
    dlclose(handle);
    api.handle = NULL;
    return NULL;
  }
  return &api;
}

jl_hevc *jl_hevc_create(int width, int height) {
  size_t bytes = jl_hevc_frame_bytes(width, height);
  if (bytes == 0) return NULL;
  jl_av_api *api = jl_av();
  if (api == NULL) return NULL;
  struct jl_hevc *ctx = calloc(1, sizeof(*ctx));
  if (ctx == NULL) return NULL;
  ctx->width = width;
  ctx->height = height;
  ctx->frame_bytes = bytes;
  ctx->api = *api;
  ctx->nv12 = calloc(1, bytes);
  if (ctx->nv12 == NULL) {
    free(ctx);
    return NULL;
  }
  ctx->codec = api->find_by_name("hevc");
  if (ctx->codec == NULL) {
    free(ctx->nv12);
    free(ctx);
    return NULL;
  }
  return ctx;
}

void jl_hevc_free(jl_hevc *ctx) {
  if (ctx == NULL) return;
  if (ctx->context != NULL && ctx->api.free_context != NULL) ctx->api.free_context(ctx->context);
  if (ctx->api.free_frame != NULL) ctx->api.free_frame(&ctx->frame);
  free(ctx->nv12);
  free(ctx);
}

void jl_hevc_flush(jl_hevc *ctx) {
  if (ctx == NULL) return;
  ctx->broken = 1;
  // the decoder's own reference pictures are dropped too, so a P frame after
  // a gap is refused rather than decoded against frames that are gone
  if (ctx->context != NULL && ctx->api.send_packet != NULL) {
    ctx->api.send_packet(ctx->context, NULL);
  }
}

int jl_hevc_available(void) {
  return jl_av() != NULL ? 1 : 0;
}

const char *jl_hevc_backend(void) {
  return jl_av() != NULL ? "libavcodec" : "";
}

// The decoder, opened on the first access unit: the width and height come
// from the bitstream's own parameter sets, and the frame path is in order,
// so there is no reconfiguration to do.
static int jl_open(struct jl_hevc *ctx) {
  if (ctx->opened) return 0;
  if (ctx->context != NULL) {
    ctx->api.free_context(ctx->context);
    ctx->context = NULL;
  }
  ctx->context = ctx->api.alloc_context(ctx->codec);
  if (ctx->context == NULL) return -1;
  if (ctx->api.open(ctx->context, ctx->codec) != 0) {
    ctx->api.free_context(ctx->context);
    ctx->context = NULL;
    return -1;
  }
  ctx->opened = 1;
  return 0;
}

// The frame out of the decoder's, as NV12: the luma plane and the chroma
// plane, each with its own stride, into one buffer of the frame's size.
static int jl_copy_nv12(struct jl_hevc *ctx) {
  jl_av_frame *frame = &ctx->frame;
  if (frame->data[0] == NULL || frame->data[1] == NULL) return -1;
  size_t luma = (size_t)ctx->width * (size_t)ctx->height;
  for (int row = 0; row < ctx->height; row++) {
    memcpy(ctx->nv12 + (size_t)row * (size_t)ctx->width, frame->data[0] + (size_t)row * frame->linesize[0],
           (size_t)ctx->width);
  }
  // the chroma is half resolution in both directions and interleaved: the
  // decoder's plane 1 is the U and V bytes, in the order the encoder took
  for (int row = 0; row < ctx->height / 2; row++) {
    memcpy(ctx->nv12 + luma + (size_t)row * (size_t)ctx->width, frame->data[1] + (size_t)row * frame->linesize[1],
           (size_t)ctx->width);
  }
  return 0;
}

int jl_hevc_decode(jl_hevc *ctx, const void *src, size_t n, uint8_t *out, int *keyframe) {
  if (ctx == NULL || src == NULL || out == NULL || n == 0) return -3;
  int irap = jl_hevc_is_keyframe(src, n);
  if (keyframe != NULL) *keyframe = irap;
  if (irap) {
    ctx->broken = 0;
  } else if (ctx->broken) {
    return -2;   // the stream owes an IDR it has not had
  }
  if (jl_open(ctx) != 0) return -3;
  jl_av_packet packet;
  memset(&packet, 0, sizeof(packet));
  packet.data = (uint8_t *)src;
  packet.size = (int)n;
  if (ctx->api.send_packet(ctx->context, &packet) != 0) return -3;
  if (ctx->api.free_frame != NULL) ctx->api.free_frame(&ctx->frame);
  memset(&ctx->frame, 0, sizeof(ctx->frame));
  if (ctx->api.receive_frame(ctx->context, &ctx->frame) != 0) return -3;
  if (ctx->frame.width != ctx->width || ctx->frame.height != ctx->height) return -3;
  if (jl_copy_nv12(ctx) != 0) return -3;
  memcpy(out, ctx->nv12, ctx->frame_bytes);
  return 1;
}

#else

// No decoder on this build: a server that cannot decode says so rather than
// taking a frame it would have to throw away.
struct jl_hevc {
  int width;
  int height;
  size_t frame_bytes;
  int broken;
};

jl_hevc *jl_hevc_create(int width, int height) {
  (void)width;
  (void)height;
  return NULL;
}

void jl_hevc_free(jl_hevc *ctx) { (void)ctx; }

void jl_hevc_flush(jl_hevc *ctx) { (void)ctx; }

int jl_hevc_available(void) { return 0; }

const char *jl_hevc_backend(void) { return ""; }

int jl_hevc_decode(jl_hevc *ctx, const void *src, size_t n, uint8_t *out, int *keyframe) {
  (void)ctx;
  (void)src;
  (void)n;
  (void)out;
  if (keyframe != NULL) *keyframe = 0;
  return -3;
}

#endif
