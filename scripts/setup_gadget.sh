#!/usr/bin/env bash
#
# Copyright (c) 2026-, Zeph Leggett.
# This file is part of jetlink and is licensed under the MIT License.
#
# Bring up the jetlink USB gadget on the comma. Run as root, at boot, before
# anything opens the link. The comma is the USB device and the Jetson the host;
# docs/transport.md says why.
#
# The gadget is composite: the FunctionFS vendor interface first, so it stays
# interface 0 for the hosts that open it by number, and a CDC-NCM network
# interface (ECM where the kernel lacks NCM) after it. A Jetson or a Mac uses
# the vendor interface and can ignore the network one. An iPhone can only use
# the network one: it gets 192.168.60.x by DHCP from the dnsmasq this script
# starts on usb0, and dials the comma at 192.168.60.1:5599.
#
# Does not bind the UDC: a FunctionFS gadget cannot attach to a controller until
# its descriptors are written, and whoever opens ep0 writes them and binds. The
# NCM function rides on that bind, and its usb0 exists only from the first bind
# on, so the network part of setup is repeated by whoever binds:
#
#   sudo scripts/setup_gadget.sh            # at boot
#   sudo scripts/setup_gadget.sh --net      # after a bind: usb0 address, DHCP
#   sudo scripts/setup_gadget.sh --check    # what this comma can do
#   sudo scripts/setup_gadget.sh --teardown
#
# On failure the reason is left in $STATUS_FILE as well as on stderr, so the
# openpilot side can say why the link is unavailable. The network part reports
# separately in $NET_STATUS_FILE; it never fails the gadget.
set -euo pipefail

GADGET=/sys/kernel/config/usb_gadget/jetlink
FFS_MOUNT=${FFS_MOUNT:-/dev/ffs-jetlink}
FFS_NAME=jetlink
CONFIGFS=/sys/kernel/config
# tmpfs on purpose: per-boot state, and the comma's flash is precious
STATUS_FILE=${JETLINK_STATUS_FILE:-/dev/shm/jetlink-gadget}
NET_STATUS_FILE=${JETLINK_NET_STATUS_FILE:-/dev/shm/jetlink-net}
# pid.codes test allocation; get a real PID before distributing this
VID=${JETLINK_VID:-0x1209}
PID=${JETLINK_PID:-0x0001}
# NCM first: it batches packets, which a 460 KB frame benefits from. ECM is
# the older class and the fallback.
NET_FUNCTIONS=${JETLINK_NET_FUNCTIONS:-"ncm ecm"}
NET_IF=usb0    # the function name suffix only; see net_ifname for the netdev
COMMA_ADDR=${JETLINK_COMMA_ADDR:-192.168.60.1}
COMMA_PREFIX=24
DHCP_RANGE=${JETLINK_DHCP_RANGE:-192.168.60.2,192.168.60.9,1h}
DNSMASQ_PID=/dev/shm/jetlink-dnsmasq.pid
DNSMASQ_IF=/dev/shm/jetlink-dnsmasq.if
DNSMASQ_LEASES=/dev/shm/jetlink-usb0.leases

status() {
  # best effort: a device with no /dev/shm still gets the stderr line
  { echo "$1" > "$STATUS_FILE" && chmod 0644 "$STATUS_FILE"; } 2>/dev/null || true
}

net_status() {
  { echo "$1" > "$NET_STATUS_FILE" && chmod 0644 "$NET_STATUS_FILE"; } 2>/dev/null || true
}

fail() {
  echo "jetlink: $1" >&2
  status "error: $1"
  exit 1
}

# The network function present in the gadget, if any: "ncm", "ecm" or "".
net_function() {
  local kind
  for kind in $NET_FUNCTIONS; do
    if [[ -d "$GADGET/functions/$kind.$NET_IF" ]]; then
      echo "$kind"
      return 0
    fi
  done
  echo ""
}

dnsmasq_alive() {
  local pid
  pid=$(cat "$DNSMASQ_PID" 2>/dev/null || true)
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

# The interface the kernel gave the network function. The function is named
# ncm.usb0 but the netdev is not usb0: the modem already holds that name on a
# comma, so ours comes up as usb1 or later, and the number can change from one
# bind to the next. u_ether records the real name in the function's ifname,
# readable only while the gadget is bound.
net_ifname() {
  local kind name
  kind=$(net_function)
  [[ -n "$kind" ]] || return 1
  name=$(cat "$GADGET/functions/$kind.$NET_IF/ifname" 2>/dev/null || true)
  [[ -n "$name" && -d "/sys/class/net/$name" ]] || return 1
  echo "$name"
}

# The comma's end of the cable network. Idempotent, and never fatal: the gadget
# is usable by a Jetson or a Mac with no network at all. Called at the end of
# setup, where the netdev usually does not exist yet, and by the owner through
# --net once it has bound the UDC and the interface has appeared.
net_up() {
  local kind dev why
  kind=$(net_function)
  if [[ -z "$kind" ]]; then
    net_status "net: unavailable"
    echo "no network function in the gadget; the cable link is USB only" >&2
    return 0
  fi
  if ! dev=$(net_ifname); then
    net_status "error: no netdev yet; it appears when the owner binds the UDC (then run --net)"
    echo "$kind function present, its netdev not yet created (appears at bind); run --net after binding" >&2
    return 0
  fi
  # NetworkManager manages every device on AGNOS and would DHCP on it itself
  nmcli dev set "$dev" managed no >/dev/null 2>&1 || true
  if ! ip addr replace "$COMMA_ADDR/$COMMA_PREFIX" dev "$dev" 2>/dev/null; then
    why="could not set $COMMA_ADDR/$COMMA_PREFIX on $dev"
    net_status "error: $why"; echo "jetlink: $why" >&2
    return 0
  fi
  if ! ip link set "$dev" up 2>/dev/null; then
    why="could not bring $dev up"
    net_status "error: $why"; echo "jetlink: $why" >&2
    return 0
  fi
  # Steer receive processing onto the big cores: the comma's little cores add
  # milliseconds to a 460 KB frame. Not every kernel exposes it.
  if [[ -w "/sys/class/net/$dev/queues/rx-0/rps_cpus" ]]; then
    echo f0 > "/sys/class/net/$dev/queues/rx-0/rps_cpus" 2>/dev/null || true
  fi
  # DHCP for the phone, and only that: no router (option 3) and no DNS (option 6),
  # so the phone keeps its default route over Wi-Fi. No DNS service (--port=0).
  # dnsmasq binds the interface by name, so a netdev that came back under a new
  # name after a rebind needs a fresh dnsmasq: the running one is on a ghost.
  if dnsmasq_alive && [[ "$(cat "$DNSMASQ_IF" 2>/dev/null || true)" != "$dev" ]]; then
    kill "$(cat "$DNSMASQ_PID")" 2>/dev/null || true
    sleep 0.2
  fi
  if ! dnsmasq_alive; then
    rm -f "$DNSMASQ_PID" 2>/dev/null || true
    if ! dnsmasq --conf-file=/dev/null --bind-interfaces --interface="$dev" \
        --except-interface=lo --port=0 --dhcp-range="$DHCP_RANGE" \
        --dhcp-option=3 --dhcp-option=6 --dhcp-leasefile="$DNSMASQ_LEASES" \
        --pid-file="$DNSMASQ_PID" 2>/dev/null; then
      why="dnsmasq would not start on $dev"
      net_status "error: $why"; echo "jetlink: $why" >&2
      return 0
    fi
    echo "$dev" > "$DNSMASQ_IF"
  fi
  net_status "ok $COMMA_ADDR $dev"
  echo "$dev ($kind) at $COMMA_ADDR/$COMMA_PREFIX, DHCP $DHCP_RANGE"
  return 0
}

net_down() {
  local pid
  pid=$(cat "$DNSMASQ_PID" 2>/dev/null || true)
  if [[ -n "$pid" ]]; then
    kill "$pid" 2>/dev/null || true
  fi
  rm -f "$DNSMASQ_PID" 2>/dev/null || true
}

check() {
  local ok=1 u g port kind found=""
  echo "kernel: $(uname -r)"
  if mountpoint -q "$CONFIGFS" && [[ -d "$CONFIGFS/usb_gadget" ]]; then
    echo "configfs USB gadgets: yes"
  else
    echo "configfs USB gadgets: NO (no $CONFIGFS/usb_gadget)"
    ok=0
  fi
  shopt -s nullglob
  local udcs=(/sys/class/udc/*)
  shopt -u nullglob
  if [[ ${#udcs[@]} -gt 0 ]]; then
    for u in "${udcs[@]}"; do
      # current_speed is the negotiated bus speed: super-speed is USB 3, and a
      # 460 KB frame is ~1 ms there against ~11 ms at high-speed (USB 2). The
      # owner logs it on every configured edge, since nothing in this script
      # runs after enumeration.
      echo "device controller: $(basename "$u"), state $(cat "$u/state" 2>/dev/null || echo unknown), speed $(cat "$u/current_speed" 2>/dev/null || echo unknown)"
    done
  else
    echo "device controller: NONE"
    ok=0
  fi
  for g in "$CONFIGFS"/usb_gadget/*/; do
    [[ -d "$g" ]] || continue
    echo "gadget $(basename "$g"): bound to '$(cat "$g/UDC" 2>/dev/null)'"
  done
  if [[ -r /proc/config.gz ]]; then
    zcat /proc/config.gz | grep -E '^CONFIG_USB_(CONFIGFS_(NCM|ECM|ECM_SUBSET|RNDIS|F_FS)|F_NCM|F_ECM)=' | sed 's/^/kernel option: /' || true
  fi
  # A phone plugged straight into the comma negotiates power and the comma may
  # end up sourcing it, which reboots the comma. Through a hub the comma sinks.
  for port in /sys/class/typec/port*; do
    [[ -d "$port" ]] || continue
    echo "USB-C $(basename "$port"): data role $(cat "$port/data_role" 2>/dev/null), power role $(cat "$port/power_role" 2>/dev/null)"
  done
  if [[ $ok -eq 1 ]]; then
    kind=$(net_function)
    if [[ -n "$kind" ]]; then
      found=" $kind (in the gadget)"
    else
      # a throwaway gadget: tries the functions and removes them, never binds
      local probe="$CONFIGFS/usb_gadget/jetlink-probe"
      mkdir -p "$probe" 2>/dev/null || true
      for kind in $NET_FUNCTIONS; do
        if mkdir "$probe/functions/$kind.probe" 2>/dev/null; then
          found="$found $kind"
          rmdir "$probe/functions/$kind.probe" 2>/dev/null || true
        fi
      done
      rmdir "$probe" 2>/dev/null || true
    fi
    if [[ -n "$found" ]]; then
      echo "network gadget functions:$found"
    else
      echo "network gadget functions: none"
    fi
    if dev=$(net_ifname); then
      echo "$dev: $(cat "/sys/class/net/$dev/operstate" 2>/dev/null || echo unknown), $(ip -4 -o addr show dev "$dev" 2>/dev/null | awk '{print $4}' | tr '\n' ' ')"
    else
      echo "netdev: absent (it appears when the owner binds the UDC)"
    fi
    echo "network status: $(cat "$NET_STATUS_FILE" 2>/dev/null || echo unknown)"
    if dnsmasq_alive; then echo "dnsmasq: running"; else echo "dnsmasq: not running"; fi
    if [[ -n "$found" ]]; then
      echo "RESULT: this comma can present a USB network adapter beside the jetlink link"
      return 0
    fi
    echo "RESULT: this comma has no network gadget function; the link is USB only"
    return 1
  fi
  echo "RESULT: this comma cannot present a USB gadget"
  return 1
}

if [[ "${1:-}" == "--check" ]]; then
  [[ $EUID -eq 0 ]] || { echo "run --check as root (sudo)" >&2; exit 1; }
  check
  exit $?
fi

if [[ "${1:-}" == "--net" ]]; then
  [[ $EUID -eq 0 ]] || fail "setup_gadget.sh --net must run as root"
  [[ -d "$GADGET" ]] || { net_status "error: no gadget"; fail "no gadget at $GADGET; run setup first"; }
  net_up
  exit 0
fi

if [[ "${1:-}" == "--teardown" ]]; then
  net_down
  if [[ -d "$GADGET" ]]; then
    echo "" > "$GADGET/UDC" 2>/dev/null || true
    rm -f "$GADGET/configs/c.1/ffs.$FFS_NAME" 2>/dev/null || true
    for f in "$GADGET"/configs/c.1/*."$NET_IF"; do
      [[ -L "$f" ]] && { rm -f "$f" 2>/dev/null || true; }
    done
    rmdir "$GADGET/configs/c.1/strings/0x409" 2>/dev/null || true
    # a config with a function still linked cannot go, and trying is a
    # configfs error worth not making
    linked=0
    for f in "$GADGET"/configs/c.1/*; do
      [[ -L "$f" ]] && linked=1
    done
    if [[ $linked -eq 0 ]]; then
      rmdir "$GADGET/configs/c.1" 2>/dev/null || true
    else
      echo "jetlink: configs/c.1 still has a function linked; left in place" >&2
    fi
    rmdir "$GADGET/functions/ffs.$FFS_NAME" 2>/dev/null || true
    for f in "$GADGET"/functions/*."$NET_IF"; do
      [[ -d "$f" ]] && { rmdir "$f" 2>/dev/null || true; }
    done
    rmdir "$GADGET/strings/0x409" 2>/dev/null || true
    rmdir "$GADGET" 2>/dev/null || true
  fi
  # A plain umount can block or segfault on a FunctionFS instance whose owner died
  # with endpoints open; lazy-detach unhooks it now and lets the kernel finish.
  umount -l "$FFS_MOUNT" 2>/dev/null || umount "$FFS_MOUNT" 2>/dev/null || true
  rmdir "$FFS_MOUNT" 2>/dev/null || true
  status "error: gadget torn down"
  net_status "error: gadget torn down"
  echo "jetlink gadget torn down"
  exit 0
fi

[[ $EUID -eq 0 ]] || fail "setup_gadget.sh must run as root"

# set -e alone exits without going through fail, leaving last boot's "ok" in
# $STATUS_FILE for the openpilot side to read.
trap 'fail "line $LINENO: $BASH_COMMAND failed"' ERR

# AGNOS builds these into the kernel and ships no /lib/modules, so modprobe is only
# worth trying where a module tree exists; the checks below test for the result.
if [[ -d "/lib/modules/$(uname -r)" ]]; then
  for m in configfs libcomposite usb_f_fs usb_f_ncm usb_f_ecm; do
    modprobe "$m" 2>/dev/null || true
  done
fi

mountpoint -q "$CONFIGFS" || mount -t configfs none "$CONFIGFS" 2>/dev/null || true
mountpoint -q "$CONFIGFS" || fail "no configfs at $CONFIGFS; this kernel cannot configure a USB gadget"

# absent means this AGNOS build has no CONFIG_USB_LIBCOMPOSITE, which nothing
# in userspace can fix
[[ -d "$CONFIGFS/usb_gadget" ]] || fail "kernel has no USB gadget support (CONFIG_USB_LIBCOMPOSITE); jetlink needs an AGNOS build that has it"

# No FunctionFS preflight on purpose: the kernel registers the functionfs
# filesystem only while some ffs.* function exists, so /proc/filesystems never
# lists it on a cold boot. The mkdir, mount and ep0 checks below test it in use.

# a gadget needs a device controller; a machine wired host-only has none
shopt -s nullglob
udcs=("/sys/class/udc"/*)
shopt -u nullglob
[[ ${#udcs[@]} -gt 0 ]] || fail "no USB device controller in /sys/class/udc; this device cannot act as a USB gadget"

# refuse to fight another gadget for the controller rather than unbinding it
for other in "$CONFIGFS"/usb_gadget/*/UDC; do
  if [[ -e "$other" ]]; then
    owner=$(basename "$(dirname "$other")")
    bound=$(cat "$other" 2>/dev/null || true)
    if [[ "$owner" != "jetlink" && -n "$bound" ]]; then
      fail "USB gadget '$owner' already holds the device controller ($bound); tear it down first"
    fi
  fi
done

mkdir -p "$GADGET" || fail "could not create the gadget at $GADGET"
cd "$GADGET"

echo "$VID"   > idVendor
echo "$PID"   > idProduct
# 0x0101: bumped when the gadget became composite, so hosts that cache
# descriptors by VID/PID/bcdDevice (macOS, Windows) fetch the new ones
echo 0x0101   > bcdDevice
echo 0x0320   > bcdUSB            # 3.2: advertise SuperSpeed
# Miscellaneous / Common Class / IAD: the composite device class, which tells a
# host to bind a driver per interface association (the vendor interface for
# jetlink, CDC-NCM for the network) rather than one for the whole device
echo 0xEF     > bDeviceClass
echo 0x02     > bDeviceSubClass
echo 0x01     > bDeviceProtocol

# The device tree carries a serial on some commas and not others; any stable
# string will do, and it also seeds the network interface's MAC addresses.
serial=$({ tr -d '\0' < /proc/device-tree/serial-number; } 2>/dev/null || cat /etc/machine-id 2>/dev/null || echo 0001)
[[ -n "$serial" ]] || serial=0001
mkdir -p strings/0x409
echo "zoompilot"  > strings/0x409/manufacturer
echo "jetlink"    > strings/0x409/product
echo "$serial"    > strings/0x409/serialnumber

mkdir -p configs/c.1/strings/0x409
echo "jetlink inference link" > configs/c.1/strings/0x409/configuration
# self-powered, and as little as the spec allows: the Jetson has its own 12 V feed
echo 0xC0 > configs/c.1/bmAttributes
echo 8    > configs/c.1/MaxPower

# the mkdir instantiates the function and registers functionfs, so a kernel
# genuinely without it fails here
mkdir -p "functions/ffs.$FFS_NAME" ||
  fail "kernel has no ffs gadget function (CONFIG_USB_CONFIGFS_F_FS); jetlink cannot present its endpoints"
# Link once: this is re-run on every deploy, and unlinking a function from a bound
# config force-unbinds the UDC, so an unconditional ln -sf drops a live link.
# Linked before the network function so it is interface 0: hosts that open the
# vendor interface by number depend on that.
[[ -L "configs/c.1/ffs.$FFS_NAME" ]] ||
  ln -s "$GADGET/functions/ffs.$FFS_NAME" "configs/c.1/ffs.$FFS_NAME" ||
  fail "could not link ffs.$FFS_NAME into configs/c.1"

# The network interface, for a phone. Optional: a kernel with neither function
# still serves a Jetson or a Mac over the vendor interface.
net_kind=$(net_function)
if [[ -z "$net_kind" ]]; then
  for kind in $NET_FUNCTIONS; do
    if mkdir "functions/$kind.$NET_IF" 2>/dev/null; then
      net_kind=$kind
      break
    fi
  done
fi
if [[ -n "$net_kind" ]]; then
  # Stable, locally administered MAC addresses from the serial, so a host sees
  # the same adapter every drive. Qualcomm's 4.9 kernel refuses the writes
  # (EINVAL) and picks its own; DHCP makes that harmless.
  hash=$(printf '%s' "$serial" | md5sum | cut -c1-10)
  mac() { printf '%s:%s:%s:%s:%s:%s' "$1" "${hash:0:2}" "${hash:2:2}" "${hash:4:2}" "${hash:6:2}" "${hash:8:2}"; }
  if ! { echo "$(mac 02)" > "functions/$net_kind.$NET_IF/dev_addr" &&
         echo "$(mac 06)" > "functions/$net_kind.$NET_IF/host_addr"; } 2>/dev/null; then
    echo "jetlink: the kernel chose the network interface's MAC addresses itself" >&2
  fi
  # after ffs, so it takes the next interface numbers
  [[ -L "configs/c.1/$net_kind.$NET_IF" ]] ||
    ln -s "$GADGET/functions/$net_kind.$NET_IF" "configs/c.1/$net_kind.$NET_IF" ||
    fail "could not link $net_kind.$NET_IF into configs/c.1"
else
  echo "jetlink: kernel has none of the '$NET_FUNCTIONS' gadget functions; no cable network" >&2
fi

mkdir -p "$FFS_MOUNT"
# Owned by the user openpilot runs as: on a root-only mount modeld and jetlinkd
# cannot open the endpoints, and Path.exists() raises rather than returning False.
FFS_USER="${JETLINK_USER:-comma}"
if id -u "$FFS_USER" >/dev/null 2>&1; then
  FFS_OPTS="uid=$(id -u "$FFS_USER"),gid=$(id -g "$FFS_USER")"
else
  FFS_OPTS=""
fi
mountpoint -q "$FFS_MOUNT" || mount -t functionfs ${FFS_OPTS:+-o "$FFS_OPTS"} "$FFS_NAME" "$FFS_MOUNT" ||
  fail "could not mount functionfs at $FFS_MOUNT"

# the check that proves the chain: ep0 is what a client opens to write the
# descriptors and bind the controller
[[ -e "$FFS_MOUNT/ep0" ]] || fail "functionfs mounted at $FFS_MOUNT but has no ep0"

# the client binds the UDC as the openpilot user, so hand it that one attribute
if [ -n "$FFS_OPTS" ]; then
  chown "$FFS_USER" "$GADGET/UDC" 2>/dev/null || true
fi

status ok
echo "gadget ready at $GADGET"
echo "functionfs mounted at $FFS_MOUNT"
if [[ -n "$net_kind" ]]; then
  echo "network function: $net_kind.$NET_IF (after ffs.$FFS_NAME in configs/c.1)"
fi
echo "available UDCs: $(ls /sys/class/udc | tr '\n' ' ')"
echo "now start the server; it writes the descriptors and binds the UDC"
# usb0 usually does not exist yet (it is created at bind), so this mostly
# records why and leaves the rest to --net; on a re-run after a bind it is
# the whole thing
net_up
