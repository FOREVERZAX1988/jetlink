/* A standalone check of the portable half of jl_hevc.c: the NAL scan, the
   IRAP test and the NV12 unpack, with the decoder stubbed out. Built and run
   on the comma, which has a compiler and no ffmpeg.

   Deliberately outside Sources/CHevc: SwiftPM compiles every .c file under a
   target's directory, and this one has a main. It stayed in the library once
   and the macOS app failed to link with "duplicate symbol '_main'" -- its own
   main and the library's. From JetlinkKit/:

     gcc -std=c11 -Wall -Wextra -Werror -O2 -I Sources/CHevc/include \
       Scripts/check_hevc.c Sources/CHevc/jl_hevc.c -o /tmp/check_hevc
     /tmp/check_hevc

   Add -DJL_HEVC_DLOPEN to compile the dlopen decoder arm instead of the stub. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "jl_hevc.h"

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } } while (0)

int main(void) {
  /* a three-byte start code: a VPS (32), an SPS (33), an IDR (19) */
  unsigned char au[] = {
    0x00, 0x00, 0x01, 0x40, 0x01, 0x0c, 0x01,
    0x00, 0x00, 0x01, 0x42, 0x01, 0x01,
    0x00, 0x00, 0x01, 0x26, 0x01, 0xaf, 0x08,
  };
  jl_nal nals[8];
  size_t n = jl_hevc_nals(au, sizeof(au), nals, 8);
  printf("nals found: %zu\n", n);
  CHECK(n == 3, "three NAL units");
  if (n == 3) {
    printf("  types: %d %d %d\n", nals[0].type, nals[1].type, nals[2].type);
    CHECK(nals[0].type == 32, "VPS is type 32");
    CHECK(nals[1].type == 33, "SPS is type 33");
    CHECK(nals[2].type == 19, "IDR is type 19");
    CHECK(nals[2].size == 4, "IDR payload is four bytes");
  }
  CHECK(jl_hevc_is_keyframe(au, sizeof(au)) == 1, "an IDR makes it a keyframe");

  /* a P frame alone: not a keyframe */
  unsigned char p[] = { 0x00, 0x00, 0x01, 0x02, 0x01, 0xd0 };
  CHECK(jl_hevc_is_keyframe(p, sizeof(p)) == 0, "a P frame is not a keyframe");

  /* a four-byte start code */
  unsigned char au4[] = { 0x00, 0x00, 0x00, 0x01, 0x26, 0x01 };
  CHECK(jl_hevc_is_keyframe(au4, sizeof(au4)) == 1, "a four-byte start code is found too");

  /* the geometry: an NV12 frame of 512 x 256, as the comma's encoder makes */
  CHECK(jl_hevc_frame_bytes(512, 256) == 512 * 256 * 3 / 2, "a 512x256 frame's bytes");
  CHECK(jl_hevc_frame_bytes(511, 256) == 0, "an odd width is not NV12");
  CHECK(jl_hevc_frame_bytes(512, 0) == 0, "no height is nothing");

  /* the unpack: an NV12 frame of 8 x 4 holds one plane of 4 x 2, so the six
     planes are 8 bytes each. Build a frame whose luma is a known pattern and
     whose chroma a known pair, and see the six planes come back where they
     belong. */
  int fw = 8, fh = 4;             /* the NV12 frame */
  int w = fw / 2, h = fh / 2;     /* one plane: 4 wide, 2 tall */
  size_t frame = jl_hevc_frame_bytes(fw, fh);
  unsigned char *nv = calloc(1, frame);
  for (int y = 0; y < fh; y++)
    for (int x = 0; x < fw; x++)
      nv[(size_t)y * fw + x] = (unsigned char)(y * 10 + x);
  size_t luma = (size_t)fw * fh;
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      nv[luma + ((size_t)y * w + x) * 2] = (unsigned char)(100 + y);
      nv[luma + ((size_t)y * w + x) * 2 + 1] = (unsigned char)(200 + y);
    }
  unsigned char planes[6 * 8];
  memset(planes, 0, sizeof(planes));
  jl_hevc_unpack_nv12(nv, fw, fh, planes);
  /* plane 0 is the luma's even,even pixels */
  CHECK(planes[0] == 0 && planes[1] == 2, "plane 0 is the even,even luma");
  /* plane 1 is even,odd */
  CHECK(planes[8] == 1 && planes[9] == 3, "plane 1 is the even,odd luma");
  /* plane 2 is odd,even */
  CHECK(planes[16] == 10 && planes[17] == 12, "plane 2 is the odd,even luma");
  /* plane 3 is odd,odd */
  CHECK(planes[24] == 11 && planes[25] == 13, "plane 3 is the odd,odd luma");
  /* plane 4 is the chroma's first, plane 5 the second */
  CHECK(planes[32] == 100 && planes[33] == 100, "plane 4 is the first chroma");
  CHECK(planes[40] == 200 && planes[41] == 200, "plane 5 is the second chroma");
  free(nv);

  printf(fails ? "\n%d CHECK(s) failed\n" : "\nall checks passed\n", fails);
  return fails ? 1 : 0;
}
