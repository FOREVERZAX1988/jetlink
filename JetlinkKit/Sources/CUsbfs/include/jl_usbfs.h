// usbdevfs, Linux's user-space USB, as plain C calls: its ioctls are macros
// Swift cannot import. What the Android app's server needs to be the USB host
// for the comma's gadget on a file descriptor the app opened
// (UsbDeviceConnection.getFileDescriptor): bulk URBs submitted, discarded and
// reaped, a wait for a completion that another thread can end, and the speed
// the bus negotiated.
//
// Calls return 0 or an errno. The kernel writes a URB's status, length and
// any IN data only when the URB is reaped, so a URB's memory and its buffer
// must outlive it until then; see UsbfsPipes.swift.
#ifndef JL_USBFS_H
#define JL_USBFS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jl_urb jl_urb;

// A bulk URB on `endpoint` (its address, 0x80 set for IN) over `buffer`.
// `context` comes back from jl_urb_context once reaped. NULL when out of memory.
jl_urb *jl_urb_create(uint8_t endpoint, void *buffer, int length, void *context);
void jl_urb_free(jl_urb *urb);
void *jl_urb_context(const jl_urb *urb);
// After the reap: 0, or a negative errno (-ENOENT or -ECONNRESET when discarded).
int jl_urb_status(const jl_urb *urb);
int jl_urb_actual(const jl_urb *urb);

int jl_usbfs_submit(int fd, jl_urb *urb);
int jl_usbfs_discard(int fd, jl_urb *urb);
// One finished URB without waiting: EAGAIN when there is none, ENODEV when
// the device is gone and nothing is left to reap.
int jl_usbfs_reap(int fd, jl_urb **out);

// An eventfd that ends a jl_usbfs_wait from another thread; -1 on failure.
int jl_usbfs_wake_create(void);
void jl_usbfs_wake(int wake_fd);
void jl_usbfs_wake_close(int wake_fd);
// Waits up to `timeout_ms` (-1: forever) for a URB to reap or a wake.
// 0 either way, ENODEV once the device is gone (reap what is left first).
int jl_usbfs_wait(int fd, int wake_fd, int timeout_ms);

// The bus speed: 1 low, 2 full, 3 high, 5 super, 6 super-plus; or -errno.
int jl_usbfs_speed(int fd);

#ifdef __cplusplus
}
#endif

#endif
