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

// Three decoder backends, in preference order:
//   JL_HEVC_MEDIACODEC  Android's own MediaCodec — the platform's HEVC
//                       hardware decoder (c2.qti.hevc.decoder on a Snapdragon).
//                       Needed on a phone that also runs the large model:
//                       FFmpeg's software decoder pinned the app's GPU at 99%.
//   JL_HEVC_DLOPEN      libavcodec opened at run time (Linux/Jetson, where
//                       ffmpeg is installed and no GPU shares the frame loop).
//   neither             no decoder: the hello omits hevc and frames stay
//                       on the lossless carrier.
#if defined(JL_HEVC_DLOPEN)
#define JL_HEVC_AV 1
#endif
#if defined(JL_HEVC_MEDIACODEC)
#define JL_HEVC_MC 1
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

#ifdef JL_HEVC_MC

// Android MediaCodec: the platform's HEVC decoder, which on a Snapdragon is
// c2.qti.hevc.decoder — hardware, off the GPU the large model is using. The
// comma's encoder prepends VPS/SPS/PPS to every IDR (hevc_venus.py), so the
// decoder needs no out-of-band CSD: each access unit is fed as-is, and a
// flush drops its references so a later P frame is refused until an IDR.
//
// Output is COLOR_FormatYUV420SemiPlanar (NV12): a luma plane followed by
// interleaved chroma, exactly what jl_copy_nv12 below reads.
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaFormat.h>

struct jl_hevc {
  int width;
  int height;
  size_t frame_bytes;
  int broken;
  AMediaCodec *codec;
  uint8_t *nv12;
  int64_t pts;              // microseconds, monotonically increasing per AU
};

static const char *jl_mc_mime(void) { return "video/hevc"; }

jl_hevc *jl_hevc_create(int width, int height) {
  size_t bytes = jl_hevc_frame_bytes(width, height);
  if (bytes == 0) return NULL;

  AMediaCodec *codec = AMediaCodec_createDecoderByType(jl_mc_mime());
  if (codec == NULL) return NULL;

  AMediaFormat *fmt = AMediaFormat_new();
  if (fmt == NULL) { AMediaCodec_delete(codec); return NULL; }
  AMediaFormat_setString(fmt, AMEDIAFORMAT_KEY_MIME, jl_mc_mime());
  AMediaFormat_setInt32(fmt, AMEDIAFORMAT_KEY_WIDTH, width);
  AMediaFormat_setInt32(fmt, AMEDIAFORMAT_KEY_HEIGHT, height);
  // 0x7F000001 = COLOR_FormatYUV420SemiPlanar (NV12), the format openpilot's
  // Venus encoder produces. Named here rather than via a header because the
  // constant lives in MediaFormat.h (Java-facing), not in the NDK C headers.
  AMediaFormat_setInt32(fmt, AMEDIAFORMAT_KEY_COLOR_FORMAT, 0x7F000001);

  media_status_t st = AMediaCodec_configure(codec, fmt, NULL, NULL, 0);
  AMediaFormat_delete(fmt);
  if (st != AMEDIA_OK) { AMediaCodec_delete(codec); return NULL; }
  if (AMediaCodec_start(codec) != AMEDIA_OK) { AMediaCodec_delete(codec); return NULL; }

  struct jl_hevc *ctx = (struct jl_hevc *)calloc(1, sizeof(*ctx));
  if (ctx == NULL) { AMediaCodec_delete(codec); return NULL; }
  ctx->codec = codec;
  ctx->width = width;
  ctx->height = height;
  ctx->frame_bytes = bytes;
  ctx->nv12 = (uint8_t *)calloc(1, bytes);
  if (ctx->nv12 == NULL) { AMediaCodec_delete(codec); free(ctx); return NULL; }
  return ctx;
}

void jl_hevc_free(jl_hevc *ctx) {
  if (ctx == NULL) return;
  if (ctx->codec != NULL) {
    AMediaCodec_stop(ctx->codec);
    AMediaCodec_delete(ctx->codec);
  }
  free(ctx->nv12);
  free(ctx);
}

void jl_hevc_flush(jl_hevc *ctx) {
  if (ctx == NULL || ctx->codec == NULL) return;
  ctx->broken = 1;
  AMediaCodec_flush(ctx->codec);
}

int jl_hevc_available(void) { return 1; }

const char *jl_hevc_backend(void) { return "mediacodec"; }

// The decoder's NV12 out (luma + interleaved chroma, with strides) into one
// tight NV12 buffer. Anonymous namespace so it cannot clash with the
// libavcodec backend's helper of the same name.
static int jl_mc_copy_nv12(struct jl_hevc *ctx, uint8_t *data, size_t n,
                           int32_t offset, int32_t size) {
  if (data == NULL || size <= 0 || (size_t)offset + (size_t)size > n) return -1;
  const uint8_t *frame = data + offset;
  // The codec's stride can exceed width (alignment); the encoder made a
  // tight 512x256 NV12, but read the stride the codec reports by deriving it
  // from the buffer size: luma is width*height, chroma width*height/2.
  size_t luma = (size_t)ctx->width * (size_t)ctx->height;
  if ((size_t)size < luma) return -1;
  memcpy(ctx->nv12, frame, luma);
  memcpy(ctx->nv12 + luma, frame + luma, (size_t)size - luma);
  return 0;
}

int jl_hevc_decode(jl_hevc *ctx, const void *src, size_t n, uint8_t *out, int *keyframe) {
  if (ctx == NULL || src == NULL || out == NULL || n == 0 || ctx->codec == NULL) return -3;
  int irap = jl_hevc_is_keyframe(src, n);
  if (keyframe != NULL) *keyframe = irap;
  if (irap) {
    ctx->broken = 0;
  } else if (ctx->broken) {
    return -2;   // no IDR yet: the codec has no reference picture for this AU
  }

  // Feed the access unit. A positive index is an input buffer slot; the
  // negatives are INFO_TRY_AGAIN_LATER (-1) and a codec that went away.
  //
  // Both waits are short on purpose. This call runs on modeld's frame thread,
  // inside a 50 ms camera period: waiting 20 ms for an input slot and then up
  // to 4 x 20 ms for its frame spent 100 ms here and pushed the reply past the
  // link's 0.2 s deadline (measured 2026-11-16: `link 158 ms`, failure 7).
  // Jetlink is pipelined, so a frame whose picture is not out yet is a frame
  // to publish the previous plan for -- the next call drains it.
  ssize_t in = AMediaCodec_dequeueInputBuffer(ctx->codec, 2000);  // 2 ms
  if (in < 0) return 0;   // no input slot yet: publish the previous plan
  size_t cap = 0;
  uint8_t *buf = AMediaCodec_getInputBuffer(ctx->codec, (size_t)in, &cap);
  if (buf == NULL || n > cap) return -3;
  memcpy(buf, src, n);
  uint32_t flags = irap ? AMEDIACODEC_BUFFER_FLAG_KEY_FRAME : 0;
  if (AMediaCodec_queueInputBuffer(ctx->codec, (size_t)in, 0, n,
                                   (uint64_t)ctx->pts++, flags) != AMEDIA_OK) return -3;

  // Drain at most one decoded frame, without waiting past a few milliseconds.
  // The codec decodes on its own time; what is not out yet is drained by the
  // next access unit, which is exactly the one-frame-old plan a pipelined
  // frame publishes anyway.
  for (int tries = 0; tries < 2; tries++) {
    AMediaCodecBufferInfo info;
    memset(&info, 0, sizeof(info));
    ssize_t ix = AMediaCodec_dequeueOutputBuffer(ctx->codec, &info, 3000);  // 3 ms
    if (ix == AMEDIACODEC_INFO_TRY_AGAIN_LATER) break;
    if (ix == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED ||
        ix == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) continue;
    if (ix < 0) return -3;
    size_t out_cap = 0;
    uint8_t *out_buf = AMediaCodec_getOutputBuffer(ctx->codec, (size_t)ix, &out_cap);
    if (out_buf == NULL) { AMediaCodec_releaseOutputBuffer(ctx->codec, (size_t)ix, 0); return -3; }
    int rc = jl_mc_copy_nv12(ctx, out_buf, out_cap, info.offset, info.size);
    AMediaCodec_releaseOutputBuffer(ctx->codec, (size_t)ix, 0);
    if (rc != 0) return -3;
    memcpy(out, ctx->nv12, ctx->frame_bytes);
    return 1;
  }
  return 0;   // no frame yet: hold the previous plan, not a failure
}

#elif defined(JL_HEVC_AV)

#include <dlfcn.h>

// libavcodec's few entry points, named as its headers name them. The library
// is opened at run time, so the build needs none of libavcodec's own
// headers and a host without it simply has no decoder.
//
// The types below mirror libavcodec's public structs field-for-field, in
// declaration order, up to the last field this code reads. Only that prefix
// is named; the real structs continue past it, which is fine because the API
// takes pointers and only touches the fields we lay out here.
//
// This prefix used to be wrong: the packet was `data,size,stream_index,flags,
// pts,dts` and the frame omitted `nb_samples`. That is not any FFmpeg
// release's ABI — AVPacket starts with `buf,pts,dts`, and AVFrame has
// `nb_samples` between `height` and `format`. The wrong layout went unnoticed
// on Apple (VideoToolbox, never libavcodec) and on hosts with no decoder at
// all (the hello simply omitted hevc). It would have fed send_packet a bogus
// pointer/size and read a bogus format on any real libavcodec. Fixed here so
// the Android APK, which now carries libavcodec, can decode HEVC.
//
// FFmpeg 5.1 through 7.x keep this prefix stable (AV_NUM_DATA_POINTERS = 8).
typedef struct jl_av_packet {
  void *buf;          // AVBufferRef *
  int64_t pts;
  int64_t dts;
  uint8_t *data;
  int size;
  int stream_index;
  int flags;
} jl_av_packet;

typedef struct jl_av_frame {
  uint8_t *data[8];
  int linesize[8];
  uint8_t **extended_data;
  int width;
  int height;
  int nb_samples;
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
