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

// -- a decoder's layout, tightened -------------------------------------------

int jl_hevc_tighten_nv12(const uint8_t *src, size_t n, int width, int height,
                         size_t stride, size_t slice_height, size_t left,
                         size_t chroma_at, uint8_t *out) {
  if (src == NULL || out == NULL || width <= 0 || height < 2) return -1;
  const size_t w = (size_t)width;
  const size_t h = (size_t)height;
  if (stride < w || slice_height < h) return -1;
  const size_t luma = w * h;
  // what this would read, against what src holds: the last luma row and the
  // last chroma row are the far corners
  if ((h - 1) * stride + left + w > n) return -1;
  if (chroma_at + (h / 2 - 1) * stride + w > n) return -1;
  if (stride == w && left == 0 && chroma_at == luma) {
    memcpy(out, src, luma + luma / 2);
    return 0;
  }
  for (size_t y = 0; y < h; y++) {
    memcpy(out + y * w, src + y * stride + left, w);
  }
  const uint8_t *chroma = src + chroma_at;
  for (size_t y = 0; y < h / 2; y++) {
    memcpy(out + luma + y * w, chroma + y * stride, w);
  }
  return 0;
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
#include <android/hardware_buffer.h>
#include <android/native_window.h>
#include <media/NdkImage.h>
#include <media/NdkImageReader.h>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaFormat.h>
#include <time.h>

struct jl_hevc {
  int width;
  int height;
  size_t frame_bytes;
  int broken;
  AMediaCodec *codec;
  // When the codec took a surface, the picture comes out through the reader:
  // three planes with their own row and pixel strides, in YUV_420_888, linear
  // by construction. Both are NULL when it would not take one.
  AImageReader *reader;
  ANativeWindow *window;
  /// Which component the decoder was opened from, for the log line: the same
  /// bytes through a hardware decoder and a software one are different
  /// pictures, and which one answered has to be readable.
  char codec_name[64];
  /// Set when the surface path was tried and gave nothing: the decoder is then
  /// opened again the way this device is known to answer, without one.
  int no_surface;
  /// The patient path belongs to the first access unit of a session, once. It
  /// was keyed on "no picture yet" at first, which is true for as long as the
  /// decoder produces nothing -- so every frame waited up to a second for a
  /// picture that never came, and the host gave up at 1.5 s (measured
  /// 2026-11-21: sessions of 42 and 116 frames that ended in `no answer to 2
  /// frames in 1.5s`, not in a refusal).
  int first_done;
  uint8_t *nv12;
  int64_t pts;              // microseconds, monotonically increasing per AU
  // the codec's own account of its output buffer (jl_hevc_layout_of). Where a
  // field is 0 the codec never said, and the copy falls back to the tight
  // rows the encoder itself made.
  jl_hevc_layout layout;
};

static const char *jl_mc_mime(void) { return "video/hevc"; }

// Configure from the stream's own parameter sets.
//
// Byte-buffer mode has no surface to hand a picture to, so the output layout is
// entirely the codec's business -- and with nothing but width, height and a
// colour format, this phone's decoder inferred the rest: a buffer the size of
// the luma alone (131072 bytes for a 512x256 frame), a picture half made of
// concealment, plane mean 33 where the encoder's was 60. The same access units
// decoded to 59.5/4.9 on a standard decoder, so the stream was never the
// question. Handing over VPS, SPS and PPS as csd-0 and csd-1 is what every other
// Android HEVC client does, and it is the one thing this one did not.
static int jl_mc_configure(struct jl_hevc *ctx, const uint8_t *au, size_t n) {
  jl_nal nals[64];
  size_t count = jl_hevc_nals(au, n, nals, 64);
  const uint8_t *vps = NULL, *sps = NULL, *pps = NULL;
  size_t vps_n = 0, sps_n = 0, pps_n = 0;
  for (size_t i = 0; i < count; i++) {
    switch (nals[i].type) {
      case 32: vps = nals[i].data; vps_n = nals[i].size; break;
      case 33: sps = nals[i].data; sps_n = nals[i].size; break;
      case 34: pps = nals[i].data; pps_n = nals[i].size; break;
      default: break;
    }
  }
  if (sps == NULL || pps == NULL) return -1;

  static const uint8_t sc[4] = {0, 0, 0, 1};
  const size_t csd0_n = (vps != NULL ? vps_n + 4 : 0) + sps_n + 4;
  uint8_t *csd0 = (uint8_t *)malloc(csd0_n);
  uint8_t *csd1 = (uint8_t *)malloc(pps_n + 4);
  if (csd0 == NULL || csd1 == NULL) { free(csd0); free(csd1); return -1; }
  size_t at = 0;
  if (vps != NULL) { memcpy(csd0 + at, sc, 4); at += 4; memcpy(csd0 + at, vps, vps_n); at += vps_n; }
  memcpy(csd0 + at, sc, 4); at += 4;
  memcpy(csd0 + at, sps, sps_n);
  memcpy(csd1, sc, 4);
  memcpy(csd1 + 4, pps, pps_n);

  // The software decoder first, then whatever the platform picks. This is an
  // A/B on the component, not a preference: the same access units decode to
  // 125.0/72.5 here and to 43/68 with no chroma plane on the phone, through
  // both the codec's buffer and a surface, with and without parameter sets.
  // What is left to vary is which decoder answered.
  AMediaCodec *codec = AMediaCodec_createCodecByName("c2.android.hevc.decoder");
  if (codec != NULL) {
    snprintf(ctx->codec_name, sizeof(ctx->codec_name), "%s", "c2.android.hevc.decoder");
  } else {
    codec = AMediaCodec_createDecoderByType(jl_mc_mime());
    if (codec == NULL) { free(csd0); free(csd1); return -1; }
    snprintf(ctx->codec_name, sizeof(ctx->codec_name), "%s", "platform default");
  }

  AMediaFormat *fmt = AMediaFormat_new();
  if (fmt == NULL) { AMediaCodec_delete(codec); free(csd0); free(csd1); return -1; }
  AMediaFormat_setString(fmt, AMEDIAFORMAT_KEY_MIME, jl_mc_mime());
  AMediaFormat_setInt32(fmt, AMEDIAFORMAT_KEY_WIDTH, ctx->width);
  AMediaFormat_setInt32(fmt, AMEDIAFORMAT_KEY_HEIGHT, ctx->height);
  // 0x7F000001 = COLOR_FormatYUV420SemiPlanar (NV12), what openpilot's Venus
  // encoder produces. Named here rather than via a header because the constant
  // lives in MediaFormat.h (Java-facing), not in the NDK C headers.
  AMediaFormat_setInt32(fmt, AMEDIAFORMAT_KEY_COLOR_FORMAT, 0x7F000001);
  AMediaFormat_setBuffer(fmt, "csd-0", csd0, csd0_n);
  AMediaFormat_setBuffer(fmt, "csd-1", csd1, pps_n + 4);
  // Ask the codec not to queue frames ahead. Measured 2026-11-16: a frame's
  // round trip through the phone ran 156-167 ms while the phone's own inference
  // was 27 ms and ping was 17 ms -- the rest was the decoder sitting on two or
  // three frames of its own pipeline. Both keys are hints a codec may ignore.
  AMediaFormat_setInt32(fmt, "low-latency", 1);
  AMediaFormat_setInt32(fmt, "operating-rate", 30);

  // A surface for the codec to render into, and a reader behind it.
  //
  // Byte-buffer output from this decoder was never a picture: a 131072-byte
  // buffer for a 512x256 frame (the luma alone) holding black with a little
  // noise, while the same access units decoded to 59.5/4.9 elsewhere. Handing
  // it a surface takes the layout out of its hands twice over: the queue does
  // the colour conversion, and what comes back is YUV_420_888 with its own row
  // and pixel strides, which AImage reports instead of being guessed at.
  AImageReader *reader = NULL;
  ANativeWindow *window = NULL;
  // CPU_READ_OFTEN, because the picture is read here and nowhere else: without
  // it the queue may hand back buffers only the GPU can touch, and
  // AImage_getPlaneData answers with nothing.
  media_status_t made = ctx->no_surface ? AMEDIA_ERROR_UNKNOWN
                                         : AImageReader_newWithUsage(
                                             ctx->width, ctx->height, AIMAGE_FORMAT_YUV_420_888,
                                             AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, 4, &reader);
  if (made != AMEDIA_OK && !ctx->no_surface) {
    made = AImageReader_new(ctx->width, ctx->height, AIMAGE_FORMAT_YUV_420_888, 4, &reader);
  }
  if (made != AMEDIA_OK || reader == NULL ||
      AImageReader_getWindow(reader, &window) != AMEDIA_OK) {
    // no reader on this device: fall back to the buffer the codec hands over
    if (reader != NULL) AImageReader_delete(reader);
    reader = NULL;
    window = NULL;
  }

  // The parameter sets are the one part that is not a hint: a codec that does
  // not know a key in `fmt` may reject the whole format, so this is also tried
  // stripped down to what a decoder cannot do without.
  AMediaFormat *plain = AMediaFormat_new();
  if (plain != NULL) {
    AMediaFormat_setString(plain, AMEDIAFORMAT_KEY_MIME, jl_mc_mime());
    AMediaFormat_setInt32(plain, AMEDIAFORMAT_KEY_WIDTH, ctx->width);
    AMediaFormat_setInt32(plain, AMEDIAFORMAT_KEY_HEIGHT, ctx->height);
    AMediaFormat_setInt32(plain, AMEDIAFORMAT_KEY_COLOR_FORMAT, 0x7F000001);
    AMediaFormat_setBuffer(plain, "csd-0", csd0, csd0_n);
    AMediaFormat_setBuffer(plain, "csd-1", csd1, pps_n + 4);
  }

  // In order of preference, and the surface is the one that matters: a picture
  // through the reader has a layout the codec cannot get wrong.
  media_status_t st = AMEDIA_ERROR_UNKNOWN;
  if (window != NULL) st = AMediaCodec_configure(codec, fmt, window, NULL, 0);
  if (st != AMEDIA_OK && window != NULL && plain != NULL) {
    st = AMediaCodec_configure(codec, plain, window, NULL, 0);
  }
  if (st != AMEDIA_OK) {
    // a codec that will not take the surface still decodes into a buffer
    if (window != NULL) {
      AImageReader_delete(reader);
      reader = NULL;
      window = NULL;
    }
    st = AMediaCodec_configure(codec, fmt, NULL, NULL, 0);
    if (st != AMEDIA_OK && plain != NULL) {
      st = AMediaCodec_configure(codec, plain, NULL, NULL, 0);
    }
  }
  if (plain != NULL) AMediaFormat_delete(plain);
  AMediaFormat_delete(fmt);
  free(csd0);
  free(csd1);
  if (st != AMEDIA_OK) {
    AMediaCodec_delete(codec);
    if (reader != NULL) AImageReader_delete(reader);
    return -1;
  }
  if (AMediaCodec_start(codec) != AMEDIA_OK) {
    AMediaCodec_delete(codec);
    if (reader != NULL) AImageReader_delete(reader);
    return -1;
  }
  ctx->codec = codec;
  ctx->reader = reader;
  ctx->window = window;
  return 0;
}

jl_hevc *jl_hevc_create(int width, int height) {
  size_t bytes = jl_hevc_frame_bytes(width, height);
  if (bytes == 0) return NULL;
  if (jl_hevc_available() == 0) return NULL;
  struct jl_hevc *ctx = (struct jl_hevc *)calloc(1, sizeof(*ctx));
  if (ctx == NULL) return NULL;
  ctx->width = width;
  ctx->height = height;
  ctx->frame_bytes = bytes;
  ctx->layout.width = width;
  ctx->layout.height = height;
  // nothing decodes until a parameter set has been seen: a P frame with no
  // reference is a frame to conceal, and concealing it published a black picture
  ctx->broken = 1;
  ctx->nv12 = (uint8_t *)calloc(1, bytes);
  if (ctx->nv12 == NULL) { free(ctx); return NULL; }
  return ctx;
}

void jl_hevc_free(jl_hevc *ctx) {
  if (ctx == NULL) return;
  if (ctx->codec != NULL) {
    AMediaCodec_stop(ctx->codec);
    AMediaCodec_delete(ctx->codec);
  }
  // the window is the reader's, so it goes when the reader does
  if (ctx->reader != NULL) AImageReader_delete(ctx->reader);
  free(ctx->nv12);
  free(ctx);
}

void jl_hevc_flush(jl_hevc *ctx) {
  if (ctx == NULL || ctx->codec == NULL) return;
  ctx->broken = 1;
  AMediaCodec_flush(ctx->codec);
  if (ctx->reader != NULL) {
    // frames already rendered are from before the flush
    AImage *stale = NULL;
    while (AImageReader_acquireLatestImage(ctx->reader, &stale) == AMEDIA_OK && stale != NULL) {
      AImage_delete(stale);
      stale = NULL;
    }
  }
}

int jl_hevc_available(void) {
  // Ask the platform, once. This used to answer 1 unconditionally, which is
  // how a hello came to offer a codec the server could not decode: every
  // join then ended in BAD_SHAPE (2026-11-16) because HevcFrame's decoders
  // were nil. A device whose HEVC decoder is missing, or refuses this
  // process, must not be offered at all.
  static int checked = -1;
  if (checked >= 0) return checked;
  AMediaCodec *probe = AMediaCodec_createDecoderByType(jl_mc_mime());
  checked = probe != NULL ? 1 : 0;
  if (probe != NULL) AMediaCodec_delete(probe);
  return checked;
}

const char *jl_hevc_backend(void) { return "mediacodec"; }

// What the codec says about its own output buffer. A hardware decoder aligns
// rows to its own width (64, 128 or 256 bytes, which for a 512-pixel luma may
// or may not land on the picture's edge) and pads the picture up to whole
// macroblocks, and neither is visible in the buffer's size or content.
// The keys are named as strings because the C headers carry only some of them
// (the same reason "low-latency" below is a string).
static void jl_mc_read_layout(struct jl_hevc *ctx) {
  if (ctx->codec == NULL) return;
  AMediaFormat *fmt = AMediaCodec_getOutputFormat(ctx->codec);
  if (fmt == NULL) return;
  int32_t v = 0;
  if (AMediaFormat_getInt32(fmt, "stride", &v) && v > 0) ctx->layout.stride = v;
  if (AMediaFormat_getInt32(fmt, "slice-height", &v) && v > 0) ctx->layout.slice_height = v;
  if (AMediaFormat_getInt32(fmt, "color-format", &v) && v > 0) ctx->layout.color_format = v;
  if (AMediaFormat_getInt32(fmt, "crop-left", &v) && v >= 0) ctx->layout.crop_left = v;
  if (AMediaFormat_getInt32(fmt, "crop-top", &v) && v >= 0) ctx->layout.crop_top = v;
  AMediaFormat_delete(fmt);
}

// The decoder's NV12 out (luma + interleaved chroma, with the strides the
// codec reported) into one tight NV12 buffer.
//
// This used to memcpy `width * height` from the front on the assumption that
// the codec laid its rows tight, and Android's hardware HEVC decoder does not
// have to: it pads each row to its own alignment and the picture to whole
// macroblocks. On a padded buffer that copy neither fails nor truncates -- it
// reads width bytes out of every stride bytes, so the picture shears a little
// every row and the chroma starts at the wrong place entirely. The frame still
// "decodes", which is why the carrier looked healthy while the model answered
// as if the car were parked: the plane means measured downstream (16/42) were
// nothing like the encoder's (65/5).
// One picture out of the reader, as tight NV12.
//
// YUV_420_888 is three planes and neither of them is guaranteed tight: the row
// stride is whatever the queue picked and the chroma may be interleaved (pixel
// stride 2, NV12) or planar (1, I420). Both are read here the same way, from
// the strides the image itself carries -- the numbers the byte-buffer path
// could never get out of this decoder.
static int jl_mc_copy_image(struct jl_hevc *ctx, AImage *img) {
  int32_t iw = 0, ih = 0;
  if (AImage_getWidth(img, &iw) != AMEDIA_OK || AImage_getHeight(img, &ih) != AMEDIA_OK) return -1;
  const int w = ctx->width, h = ctx->height;
  if (iw < w || ih < h) return -1;   // a picture smaller than the model's frame

  uint8_t *plane[3] = {NULL, NULL, NULL};
  int len[3] = {0, 0, 0};
  int32_t row[3] = {0, 0, 0}, pix[3] = {0, 0, 0};
  for (int i = 0; i < 3; i++) {
    if (AImage_getPlaneData(img, i, &plane[i], &len[i]) != AMEDIA_OK) plane[i] = NULL;
    if (AImage_getPlaneRowStride(img, i, &row[i]) != AMEDIA_OK) row[i] = 0;
    if (AImage_getPlanePixelStride(img, i, &pix[i]) != AMEDIA_OK) pix[i] = 0;
  }
  ctx->layout.img_w = iw;
  ctx->layout.img_h = ih;
  ctx->layout.img_row0 = row[0];
  ctx->layout.img_row1 = row[1];
  ctx->layout.img_pix1 = pix[1];
  ctx->layout.img_len0 = len[0];
  ctx->layout.img_len1 = len[1];

  // A plane this code cannot read is not a reason to publish nothing: the luma
  // is four of the model's six planes, and a picture with a guessed chroma is
  // a number to argue with where an empty frame is not. So every read below is
  // clamped to what the plane actually holds, and whatever cannot be read is
  // filled rather than refused.
  const size_t luma = (size_t)w * (size_t)h;
  const size_t cw = (size_t)w / 2, ch = (size_t)h / 2;
  const int y_row = row[0] >= w ? row[0] : w;

  if (plane[0] != NULL) {
    int rows = 0;
    for (int y = 0; y < h; y++) {
      const size_t at = (size_t)y * (size_t)y_row;
      if (at + (size_t)w > (size_t)len[0]) break;
      memcpy(ctx->nv12 + (size_t)y * (size_t)w, plane[0] + at, (size_t)w);
      rows++;
    }
    // what could not be read repeats the last row that could, so a short plane
    // is a smear rather than a black band
    for (int y = rows; y < h; y++) {
      memcpy(ctx->nv12 + (size_t)y * (size_t)w,
             ctx->nv12 + (size_t)(rows > 0 ? rows - 1 : 0) * (size_t)w, (size_t)w);
    }
    if (rows == 0) return -1;   // not one readable row: this is not a picture
  } else {
    return -1;
  }

  uint8_t *dst = ctx->nv12 + luma;
  if (plane[1] != NULL && plane[2] != NULL && pix[1] >= 1 && pix[2] >= 1 &&
      row[1] >= 1 && row[2] >= 1) {
    const int u_row = row[1], v_row = row[2];
    const int u_pix = pix[1], v_pix = pix[2];
    for (size_t y = 0; y < ch; y++) {
      uint8_t *d = dst + y * (size_t)w;
      const size_t u_at = y * (size_t)u_row, v_at = y * (size_t)v_row;
      if (u_at + (cw - 1) * (size_t)u_pix + 1 > (size_t)len[1] ||
          v_at + (cw - 1) * (size_t)v_pix + 1 > (size_t)len[2]) {
        memset(d, 128, (size_t)w);   // past what the plane holds
        continue;
      }
      const uint8_t *ru = plane[1] + u_at;
      const uint8_t *rv = plane[2] + v_at;
      for (size_t x = 0; x < cw; x++) {
        d[2 * x] = ru[x * (size_t)u_pix];
        d[2 * x + 1] = rv[x * (size_t)v_pix];
      }
    }
  } else {
    memset(dst, 128, luma / 2);   // no readable chroma: neutral, not a guess
  }

  ctx->layout.stride = y_row;
  ctx->layout.slice_height = ih;
  ctx->layout.color_format = AIMAGE_FORMAT_YUV_420_888;
  ctx->layout.crop_left = 0;
  ctx->layout.crop_top = 0;
  ctx->layout.copies++;
  ctx->layout.from_image++;
  if (y_row != w) ctx->layout.strided++;
  ctx->layout.head_lo = 0;
  ctx->layout.head_hi = 0;
  for (int i = 0; i < 8; i++) {
    ctx->layout.head_lo |= (unsigned long long)ctx->nv12[i] << (8 * i);
    ctx->layout.head_hi |= (unsigned long long)ctx->nv12[8 + i] << (8 * i);
  }
  return 0;
}

static int jl_mc_copy_nv12(struct jl_hevc *ctx, uint8_t *data, size_t n,
                           int32_t offset, int32_t size) {
  if (data == NULL || size <= 0 || (size_t)offset + (size_t)size > n) {
    ctx->layout.refused++;
    return -1;
  }
  const uint8_t *frame = data + offset;
  ctx->layout.last_size = size;
  ctx->layout.last_offset = offset;
  ctx->layout.out_cap = (int)(n - (size_t)offset);
  // The first 16 bytes of the picture, as they lie in the buffer. A linear
  // NV12 row of a lit scene is a smooth run of neighbouring values; a tiled or
  // compressed buffer is not, and telling those apart from the numbers alone
  // cost an afternoon (2026-11-21: stride=0, size=luma, mean 28, spread 55).
  if (n >= (size_t)offset + 16) {
    memcpy(&ctx->layout.head_lo, frame, 8);
    memcpy(&ctx->layout.head_hi, frame + 8, 8);
  }
  const size_t w = (size_t)ctx->width;
  const size_t h = (size_t)ctx->height;
  const size_t luma = w * h;
  if ((size_t)size < luma) {
    // not even a tight luma: nothing here to copy, whatever it is
    ctx->layout.refused++;
    return -1;
  }
  // What bounds a padded read is the buffer's own capacity, not `size`: a
  // decoder reports `size` as the bytes of the picture it cropped, while the
  // allocation behind it carries the padding. Bounding by `size` refuses a
  // frame this could have laid out -- the first closed loop on 2026-11-21 came
  // back BAD_SHAPE at frame 0 for exactly that.
  const size_t avail = n > (size_t)offset ? n - (size_t)offset : 0;

  // A layout the codec never named is the tight one the encoder made; a stride
  // narrower than the picture cannot be one, so it is not trusted either.
  size_t stride = ctx->layout.stride > 0 && (size_t)ctx->layout.stride >= w
                    ? (size_t)ctx->layout.stride : w;
  size_t slice = ctx->layout.slice_height > 0 && (size_t)ctx->layout.slice_height >= h
                   ? (size_t)ctx->layout.slice_height : h;
  const size_t left = ctx->layout.crop_left > 0 && (size_t)ctx->layout.crop_left < w
                        ? (size_t)ctx->layout.crop_left : 0;
  const size_t top = ctx->layout.crop_top > 0 && (size_t)ctx->layout.crop_top < h
                       ? (size_t)ctx->layout.crop_top : 0;
  const size_t top_off = top * stride;
  const uint8_t *picture = frame + top_off;
  const size_t room = top_off < avail ? avail - top_off : 0;

  // the chroma begins after the luma's slice rows, padding included
  size_t chroma_at = stride * slice;
  if (chroma_at + stride * (h / 2) > avail) {
    // a codec that pads rows but starts the chroma right after the picture
    const size_t alt = stride * h;
    if (alt + stride * (h / 2) <= avail) {
      chroma_at = alt;
      slice = h;
    }
  }
  const size_t chroma_rel = chroma_at > top_off ? chroma_at - top_off : 0;
  if (chroma_at > top_off && chroma_rel + (h / 2 - 1) * stride + w <= room &&
      jl_hevc_tighten_nv12(picture, room, ctx->width, ctx->height, stride, slice,
                           left, chroma_rel, ctx->nv12) == 0) {
    ctx->layout.copies++;
    if (stride != w || slice != h || left != 0) ctx->layout.strided++;
    return 0;
  }

  // Nothing fitted the layout the codec named, so the buffer is the tight one
  // after all -- or its geometry is not one this understands. Take what is
  // there rather than refuse a frame: a sheared picture is bad, no picture is
  // worse (it is BAD_SHAPE at the host, and the whole link hands back).
  if ((size_t)size >= luma + luma / 2) {
    memcpy(ctx->nv12, frame, luma + luma / 2);
  } else {
    memcpy(ctx->nv12, frame, luma);
    memset(ctx->nv12 + luma, 0, luma / 2);
  }
  ctx->layout.copies++;
  ctx->layout.tight_fallback++;
  return 0;
}

const jl_hevc_layout *jl_hevc_layout_of(jl_hevc *ctx) {
  return ctx == NULL ? NULL : &ctx->layout;
}

int jl_hevc_format_string(jl_hevc *ctx, char *out, size_t cap) {
  if (ctx == NULL || out == NULL || cap == 0) return 0;
  out[0] = 0;
  if (ctx->codec == NULL) return 0;
  AMediaFormat *fmt = AMediaCodec_getOutputFormat(ctx->codec);
  size_t at = 0;
  const char *pre = "codec=";
  while (*pre != 0 && at + 1 < cap) out[at++] = *pre++;
  for (int i = 0; ctx->codec_name[i] != 0 && at + 1 < cap; i++) out[at++] = ctx->codec_name[i];
  out[at] = 0;
  if (fmt == NULL) return (int)at;
  const char *text = AMediaFormat_toString(fmt);
  if (text != NULL) {
    if (at + 1 < cap) out[at++] = ' ';
    for (int i = 0; text[i] != 0 && at + 1 < cap; i++) out[at++] = text[i];
    out[at] = 0;
  }
  AMediaFormat_delete(fmt);
  return (int)at;
}

// Take the decoder down and keep nothing: the codec and the queue it renders
// into only make sense together.
static void jl_mc_teardown(struct jl_hevc *ctx) {
  if (ctx->codec != NULL) {
    AMediaCodec_stop(ctx->codec);
    AMediaCodec_delete(ctx->codec);
    ctx->codec = NULL;
  }
  if (ctx->reader != NULL) {
    AImageReader_delete(ctx->reader);
    ctx->reader = NULL;
  }
  ctx->window = NULL;
  ctx->broken = 1;
}

// Feed one access unit and take one picture out of it.
//
// 1 is a frame, 0 is nothing out yet, -3 is a codec that would not even take
// the bytes. Nothing here waits: a frame that is not out yet is the one-frame-old
// plan a pipelined link publishes anyway, and a decoder that has only just been
// configured is covered by the blank-frame grace on the other side of the
// bridge. Waiting here instead -- 0.8 s for a first picture -- was what could
// not fit the link's 0.45 s first-frame budget.
static int jl_mc_take(struct jl_hevc *ctx, const void *src, size_t n, uint8_t *out, int irap) {
  // No access unit waits any more.
  //
  // The 300 ms for an input slot and 700 ms for a picture were what made a join
  // survive a decoder that had not started, back when "nothing yet" was answered
  // as BAD_SHAPE. The blank-frame grace in HevcFrame answers that now, and the
  // wait has become the only thing that cannot meet the link's first-frame
  // budget (0.45 s) with the phone's decoder taking a frame or two to warm up.
  // So a frame is fed and taken in a few milliseconds, and a picture that is not
  // out yet is the next frame's.
  ssize_t in = -1;
  int slot_ms = 2;
  while (slot_ms > 0) {
    in = AMediaCodec_dequeueInputBuffer(ctx->codec, 2000);   // 2 ms
    if (in >= 0) break;
    slot_ms -= 2;
    break;
    struct timespec ts = {0, 2000000};   // 2 ms
    nanosleep(&ts, NULL);
  }
  if (in < 0) { ctx->layout.stage = 5; return 0; }   // no input slot yet
  size_t cap = 0;
  uint8_t *buf = AMediaCodec_getInputBuffer(ctx->codec, (size_t)in, &cap);
  if (buf == NULL || n > cap) { ctx->layout.stage = 6; return -3; }
  memcpy(buf, src, n);
  uint32_t flags = irap ? AMEDIACODEC_BUFFER_FLAG_KEY_FRAME : 0;
  if (AMediaCodec_queueInputBuffer(ctx->codec, (size_t)in, 0, n,
                                   (uint64_t)ctx->pts++, flags) != AMEDIA_OK) {
    ctx->layout.stage = 6;
    return -3;
  }

  int left_ms = 6;
  while (left_ms > 0) {
    AMediaCodecBufferInfo info;
    memset(&info, 0, sizeof(info));
    ssize_t ix = AMediaCodec_dequeueOutputBuffer(ctx->codec, &info, 3000);
    left_ms -= 3;
    if (ix == AMEDIACODEC_INFO_TRY_AGAIN_LATER) continue;
    if (ix == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
      // the one moment a codec says how wide its rows are and how it padded
      // the picture
      jl_mc_read_layout(ctx);
      continue;
    }
    if (ix == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) continue;
    if (ix < 0) { ctx->layout.stage = 2; return -3; }
    if (ctx->layout.stride == 0) {
      // ask again every frame until it answers: a format it has not finished
      // configuring is not a format that has no stride
      jl_mc_read_layout(ctx);
    }
    if (ctx->reader != NULL) {
      // render it, which hands the frame to the reader, then take the newest
      // one: a picture the link is too late for is dropped, not decoded a
      // frame behind
      AMediaCodec_releaseOutputBuffer(ctx->codec, (size_t)ix, true);
      AImage *img = NULL;
      int wait = 12;
      for (; wait > 0 && img == NULL; wait--) {
        if (AImageReader_acquireLatestImage(ctx->reader, &img) != AMEDIA_OK) {
          img = NULL;
          struct timespec ts = {0, 1000000};   // 1 ms
          nanosleep(&ts, NULL);
        }
      }
      if (img == NULL) { ctx->layout.stage = 3; continue; }
      int rc = jl_mc_copy_image(ctx, img);
      AImage_delete(img);
      if (rc != 0) { ctx->layout.stage = 4; return -3; }
      ctx->layout.stage = 0;
      memcpy(out, ctx->nv12, ctx->frame_bytes);
      return 1;
    }
    size_t out_cap = 0;
    uint8_t *out_buf = AMediaCodec_getOutputBuffer(ctx->codec, (size_t)ix, &out_cap);
    if (out_buf == NULL) {
      AMediaCodec_releaseOutputBuffer(ctx->codec, (size_t)ix, 0);
      ctx->layout.stage = 2;
      return -3;
    }
    int rc = jl_mc_copy_nv12(ctx, out_buf, out_cap, info.offset, info.size);
    AMediaCodec_releaseOutputBuffer(ctx->codec, (size_t)ix, 0);
    if (rc != 0) { ctx->layout.stage = 4; return -3; }
    ctx->layout.stage = 0;
    memcpy(out, ctx->nv12, ctx->frame_bytes);
    return 1;
  }
  if (ctx->layout.stage == 0) ctx->layout.stage = 2;
  return 0;
}

int jl_hevc_decode(jl_hevc *ctx, const void *src, size_t n, uint8_t *out, int *keyframe) {
  if (ctx == NULL || src == NULL || out == NULL || n == 0) return -3;
  int irap = jl_hevc_is_keyframe(src, n);
  if (keyframe != NULL) *keyframe = irap;
  if (ctx->codec == NULL) {
    // The decoder is opened from a parameter set, so the first access unit that
    // arrives with one is where it starts. Until then there is nothing to feed
    // it: 0 is "no frame yet", not a failure.
    if (!irap) { ctx->layout.stage = 1; return 0; }   // no parameter set yet
    if (jl_mc_configure(ctx, (const uint8_t *)src, n) != 0) { ctx->layout.stage = 1; return 0; }
  }
  // what this decoder was handed, so a log line can tell a stream that never
  // arrived from a picture it could not make
  ctx->layout.au_bytes = (int)n;
  {
    jl_nal head[1];
    size_t got = jl_hevc_nals((const uint8_t *)src, n, head, 1);
    ctx->layout.au_first_type = got > 0 ? head[0].type : -1;
  }
  if (irap) {
    ctx->broken = 0;
  } else if (ctx->broken) {
    return -2;   // no IDR yet: the codec has no reference picture for this AU
  }

  int rc = jl_mc_take(ctx, src, n, out, irap);

  // A surface that gives no picture on the very first access unit is a surface
  // this decoder will not render into -- and a join that dies at frame 0 is
  // worse than a picture with the wrong layout. Open it again without one and
  // feed the same access unit once more: the buffer path is the one this
  // device is known to answer on, whatever it puts in it.
  if (rc == 0 && ctx->first_done == 0 && ctx->reader != NULL) {
    ctx->layout.stage_first = ctx->layout.stage;   // keep what the surface got to
    jl_mc_teardown(ctx);
    ctx->no_surface = 1;
    if (jl_mc_configure(ctx, (const uint8_t *)src, n) == 0) {
      ctx->layout.stage = 0;
      ctx->broken = 0;
      rc = jl_mc_take(ctx, src, n, out, irap);
    }
  }
  ctx->first_done = 1;   // from here on, frames are answered, not waited for
  return rc;
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
  jl_hevc_layout layout;   // the decoder's row stride, for the same log line
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
  ctx->layout.width = width;
  ctx->layout.height = height;
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

int jl_hevc_format_string(jl_hevc *ctx, char *out, size_t cap) {
  // libavcodec has no MediaFormat to name: the backend is the whole answer
  (void)ctx;
  const char *name = jl_hevc_backend();
  size_t n = 0;
  if (out != NULL && cap > 0) {
    while (name[n] != 0 && n + 1 < cap) { out[n] = name[n]; n++; }
    out[n] = 0;
  }
  return (int)n;
}

const jl_hevc_layout *jl_hevc_layout_of(jl_hevc *ctx) {
  return ctx == NULL ? NULL : &ctx->layout;
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
  // libavcodec hands out each plane with its own line size, which is the same
  // fact the MediaCodec path had to be taught to ask for
  ctx->layout.stride = frame->linesize[0];
  ctx->layout.slice_height = ctx->height;
  if (frame->linesize[0] != ctx->width) ctx->layout.strided++;
  ctx->layout.copies++;
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

int jl_hevc_format_string(jl_hevc *ctx, char *out, size_t cap) {
  (void)ctx;
  if (out != NULL && cap > 0) out[0] = 0;
  return 0;
}

const jl_hevc_layout *jl_hevc_layout_of(jl_hevc *ctx) {
  (void)ctx;
  return NULL;
}

int jl_hevc_decode(jl_hevc *ctx, const void *src, size_t n, uint8_t *out, int *keyframe) {
  (void)ctx;
  (void)src;
  (void)n;
  (void)out;
  if (keyframe != NULL) *keyframe = 0;
  return -3;
}

#endif
