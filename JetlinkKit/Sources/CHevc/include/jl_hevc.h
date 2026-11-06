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

#ifdef __cplusplus
}
#endif

#endif
