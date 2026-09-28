#include "jl_usbfs.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__

#include <linux/usbdevice_fs.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <unistd.h>

struct jl_urb {
  // First, so the pointer the kernel hands back from a reap is this struct.
  struct usbdevfs_urb urb;
};

jl_urb *jl_urb_create(uint8_t endpoint, void *buffer, int length, void *context) {
  jl_urb *u = calloc(1, sizeof(jl_urb));
  if (u == NULL) {
    return NULL;
  }
  u->urb.type = USBDEVFS_URB_TYPE_BULK;
  u->urb.endpoint = endpoint;
  u->urb.buffer = buffer;
  u->urb.buffer_length = length;
  // No SHORT_NOT_OK and no ZERO_PACKET: a read ends short at the end of a
  // message, and what jetlink sends is padded so it never needs a ZLP.
  u->urb.flags = 0;
  u->urb.usercontext = context;
  return u;
}

void jl_urb_free(jl_urb *urb) {
  free(urb);
}

void *jl_urb_context(const jl_urb *urb) {
  return urb->urb.usercontext;
}

int jl_urb_status(const jl_urb *urb) {
  return urb->urb.status;
}

int jl_urb_actual(const jl_urb *urb) {
  return urb->urb.actual_length;
}

int jl_usbfs_submit(int fd, jl_urb *urb) {
  return ioctl(fd, USBDEVFS_SUBMITURB, &urb->urb) == 0 ? 0 : errno;
}

int jl_usbfs_discard(int fd, jl_urb *urb) {
  return ioctl(fd, USBDEVFS_DISCARDURB, &urb->urb) == 0 ? 0 : errno;
}

int jl_usbfs_reap(int fd, jl_urb **out) {
  struct usbdevfs_urb *done = NULL;
  *out = NULL;
  if (ioctl(fd, USBDEVFS_REAPURBNDELAY, &done) != 0) {
    return errno;
  }
  *out = (jl_urb *)done;
  return 0;
}

int jl_usbfs_wake_create(void) {
  return eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
}

void jl_usbfs_wake(int wake_fd) {
  uint64_t one = 1;
  ssize_t n = write(wake_fd, &one, sizeof(one));
  (void)n;
}

void jl_usbfs_wake_close(int wake_fd) {
  close(wake_fd);
}

int jl_usbfs_wait(int fd, int wake_fd, int timeout_ms) {
  // usbfs reports a URB to reap as writable, and a device gone as HUP.
  struct pollfd fds[2] = {
      {.fd = fd, .events = POLLOUT},
      {.fd = wake_fd, .events = POLLIN},
  };
  int n = poll(fds, 2, timeout_ms);
  if (n < 0) {
    return errno == EINTR ? 0 : errno;
  }
  if (fds[1].revents & POLLIN) {
    uint64_t drained;
    ssize_t r = read(wake_fd, &drained, sizeof(drained));
    (void)r;
  }
  if (fds[0].revents & (POLLHUP | POLLERR | POLLNVAL)) {
    return ENODEV;
  }
  return 0;
}

int jl_usbfs_speed(int fd) {
  int speed = ioctl(fd, USBDEVFS_GET_SPEED);
  return speed >= 0 ? speed : -errno;
}

#else

// Elsewhere there is no usbfs: every call says so. The package builds this
// target only for Linux and Android.
jl_urb *jl_urb_create(uint8_t endpoint, void *buffer, int length, void *context) {
  (void)endpoint, (void)buffer, (void)length, (void)context;
  return NULL;
}
void jl_urb_free(jl_urb *urb) { (void)urb; }
void *jl_urb_context(const jl_urb *urb) { (void)urb; return NULL; }
int jl_urb_status(const jl_urb *urb) { (void)urb; return -ENOSYS; }
int jl_urb_actual(const jl_urb *urb) { (void)urb; return 0; }
int jl_usbfs_submit(int fd, jl_urb *urb) { (void)fd, (void)urb; return ENOSYS; }
int jl_usbfs_discard(int fd, jl_urb *urb) { (void)fd, (void)urb; return ENOSYS; }
int jl_usbfs_reap(int fd, jl_urb **out) { (void)fd; *out = NULL; return ENOSYS; }
int jl_usbfs_wake_create(void) { return -1; }
void jl_usbfs_wake(int wake_fd) { (void)wake_fd; }
void jl_usbfs_wake_close(int wake_fd) { (void)wake_fd; }
int jl_usbfs_wait(int fd, int wake_fd, int timeout_ms) { (void)fd, (void)wake_fd, (void)timeout_ms; return ENOSYS; }
int jl_usbfs_speed(int fd) { (void)fd; return -ENOSYS; }

#endif
