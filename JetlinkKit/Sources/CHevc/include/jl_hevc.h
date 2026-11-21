// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// An HEVC frame back to the warped planes (jetlink.hevc): the comma's
// hardware encoder sent two cameras' Annex-B access units, NV12 at 512x256,
// and they are decoded to NV12 again and un-padded into the (2, 6, H, W)
// planes the model was built for.
//
// The decoder behind it is the platform's: VideoToolbox where Apple has it,
// libavcodec opened at run time elsewhere (JL_HEVC_DLOPEN, as COrt does
// with JL_ORT_DLOPEN), so a build needs no ffmpeg headers. A host with no
// decoder reports none, and that is what keeps the Python from offering
// 'hevc' to a server that cannot take it.
//
// The Python is `jetlink.hevc.from_nv12`.

#ifndef JL_HEVC_H
#define JL_HEVC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// A stream decoder: one camera's HEVC sequence. One per camera, one access
/// unit at a time, in the order they were encoded.
typedef struct jl_hevc jl_hevc;

/// A decoder for `width` x `height` NV12. NULL when the platform has no
/// decoder to open, or it refuses these dimensions.
jl_hevc *jl_hevc_create(int width, int height);
void jl_hevc_free(jl_hevc *ctx);

/// One Annex-B access unit at `src` into the frame's NV12: `width * height
/// * 3 / 2` bytes at `out`.
///
/// Returns 0 when the frame is out, 1 when a frame is ready at `out`, and a
/// negative value when the bitstream cannot be decoded from here: -2 when
/// the stream owes an IDR it has not had (a P frame after a gap, or before
/// the sequence's head) and the caller should wait for the next IRAP, and
/// -3 when the decoder itself failed.
///
/// `keyframe`, when not NULL, says whether the access unit starts a
/// sequence, read off the bitstream and not off the decoder: a caller that
/// lost its place needs to know before the frame, not after.
int jl_hevc_decode(jl_hevc *ctx, const void *src, size_t n, uint8_t *out, int *keyframe);

/// Drops the reference pictures: the next access unit has to be a keyframe
/// for anything to come out. The caller's answer to a gap
/// (jetlink.openpilot.stream).
void jl_hevc_flush(jl_hevc *ctx);

/// Whether this build has a decoder at all, and so may list 'hevc' in its
/// hello's frame_codecs. 0 when there is none: the Python never offers what
/// no server on this platform could take.
int jl_hevc_available(void);

/// The decoder's name, for a log line ("videotoolbox" / "libavcodec").
/// Empty when there is none.
const char *jl_hevc_backend(void);

/// How the decoder laid its output out, for a log line.
///
/// A decoder is free to pad its rows (a stride wider than the picture) and to
/// pad the picture to whole macroblocks (a slice-height taller than it). That
/// padding is readable only from the codec's own output format, never from the
/// buffer, and a copy that assumes tight rows does not fail on a padded buffer
/// -- it shears the picture. The model then sees a scrambled frame and answers
/// as if the camera were still (measured 2026-11-21: the comma's warped plane
/// mean 65/spread 5 arrived at the phone as mean 16/spread 42, on the one
/// carrier that has a decoder). These are the numbers that say which side of
/// the wire that happened on.
typedef struct {
  int width;          ///< the picture, as asked for
  int height;
  int stride;         ///< bytes per luma row; 0 when the codec never said
  int slice_height;   ///< luma rows in the buffer, padding included; 0 unknown
  int color_format;   ///< the codec's color-format; 0 when it never said
  int crop_left;      ///< the picture's origin inside the buffer (0 = none)
  int crop_top;
  int copies;         ///< frames copied
  int strided;        ///< of those, how many needed the row-by-row path
  int tight_fallback; ///< of those, how many fell back to tight rows anyway
  int from_image;     ///< of those, how many came through the image API
  int refused;        ///< buffers that could not hold the picture at all
  int last_size;      ///< the last output buffer's `info.size`
  int last_offset;    ///< and its `info.offset`
  int out_cap;        ///< and the capacity `getOutputBuffer` reported
  // What an AImage said about itself, kept for the log line: the geometry this
  // device would not state in a format and answers only through the queue.
  int img_w;
  int img_h;
  int img_row0;       ///< luma row stride
  int img_row1;       ///< chroma row stride
  int img_pix1;       ///< chroma pixel stride (2 interleaved, 1 planar)
  int img_len0;       ///< the luma plane's own length
  int img_len1;       ///< and the chroma plane's
  int stage_first;    ///< the stage the surface attempt reached before the
                      ///< decoder was reopened without one
  int stage;          ///< how far the last decode got: 0 frame out, 1 no
                      ///< decoder, 2 no output buffer, 3 no readable image,
                      ///< 4 a picture that would not copy
  int au_bytes;       ///< the access unit this decode was fed, in bytes
  int au_first_type;  ///< its first NAL's type (32 VPS on an IDR, 1 a P frame)
  unsigned long long head_lo;  ///< the picture's first 8 bytes, as they lie
  unsigned long long head_hi;  ///< and the next 8: a linear NV12 row of a lit
                               ///< scene is smooth; a tiled/compressed buffer
                               ///< is not
} jl_hevc_layout;

/// The last decode's layout, or NULL without a decoder. Valid until the next
/// `jl_hevc_decode`.
const jl_hevc_layout *jl_hevc_layout_of(jl_hevc *ctx);

/// The codec's own output format as text (`AMediaFormat_toString`), copied
/// into `out`. For a log line: it is the only place a decoder says what it
/// decided the picture is. Returns the length written, 0 when there is none.
int jl_hevc_format_string(jl_hevc *ctx, char *out, size_t cap);

// -- the bitstream, apart from any decoder ------------------------------------
//
// These are what the adapters are built on, and what a test can check
// without a decoder at all: where the NAL units are in an Annex-B access
// unit, and whether one of them starts a sequence.

// A NAL unit in an access unit: where its bytes are, and its type.
typedef struct {
  const uint8_t *data;
  size_t size;
  int type;
} jl_nal;

/// The NAL units of the Annex-B access unit at `au`, up to `max` of them,
/// into `out`. The number found, or 0 when there are none (a buffer with no
/// start code in it is not a bitstream).
size_t jl_hevc_nals(const uint8_t *au, size_t n, jl_nal *out, size_t max);

/// Whether the access unit holds a NAL that starts a sequence: an IRAP, type
/// 16 to 21, of which 16 (BLA), 19 (IDR) and 20 (CRA) are the ones a
/// decoder can start from.
int jl_hevc_is_keyframe(const uint8_t *au, size_t n);

/// The NV12 a frame of `width` x `height` takes, and the six planes it
/// unpacks to. Zero when the geometry is not one NV12 can hold.
size_t jl_hevc_frame_bytes(int width, int height);

/// The six warped planes (2, 6, H, W) out of an NV12 frame: the luma's four
/// phases and the chroma's two planes. `plane_shape` is (H, W) of one plane,
/// which the NV12's size alone cannot say: a wrong guess decodes to a
/// plausible but wrong frame rather than failing.
///
/// `out` is 6 * H * W bytes, the six planes one after another. The Python is
/// `jetlink.hevc.from_nv12`.
void jl_hevc_unpack_nv12(const uint8_t *nv12, int width, int height, uint8_t *out);

/// A decoder's NV12, as it laid it out, into a tight one: `height` luma rows
/// of `width` bytes read every `stride` bytes from `left`, then the chroma's
/// `height / 2` rows from `chroma_at`. `out` is the usual
/// `jl_hevc_frame_bytes(width, height)`.
///
/// This is the step that used to be a plain memcpy on the assumption that
/// every decoder lays its rows tight. Android's does not have to, and reading
/// a padded buffer that way shears the picture instead of failing. Bounds are
/// checked against `n`, the bytes `src` actually holds. Returns 0, or -1 when
/// `src` cannot supply that picture.
int jl_hevc_tighten_nv12(const uint8_t *src, size_t n, int width, int height,
                         size_t stride, size_t slice_height, size_t left,
                         size_t chroma_at, uint8_t *out);

#ifdef __cplusplus
}
#endif

#endif
