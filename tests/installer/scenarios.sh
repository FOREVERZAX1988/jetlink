#!/usr/bin/env bash
# The installer, end to end, inside a throwaway Ubuntu container: real files
# written, real units and scripts installed, and every system command the
# installer calls (apt, systemd, the server, nvpmodel, ...) replaced by
# fake.sh. run.sh starts the container; this runs in it, as root, with the
# source tree at /src and the Docker releases' installers under /releases.
#
# Each scenario is a computer (a JetPack 7.2 Jetson, a JetPack 6 one, a PC,
# an install one of the Docker releases left) plus the answers typed at the
# questions, and checks what the installer left behind, what it ran, and what
# it told the user.
set -uo pipefail

SRC=/src
FAKE_BIN=/tmp/fakebin
export FAKE_BIN FAKE_LOG=/tmp/fake.log FAKE_STATE=/tmp/fake-state
export JETLINK_TEST_DT_MODEL=/tmp/dt-model JETLINK_TEST_MEM_SLEEP=/tmp/mem-sleep
export JETLINK_TEST_PROC_VERSION=/tmp/proc-version JETLINK_TEST_SYSTEMD_RUN=/tmp
# the lock a native server makes to be held awake; none until a scenario says
export JETLINK_TEST_AWAKE_LOCK=/tmp/jetlink-awake.lock
LOCK=$JETLINK_TEST_AWAKE_LOCK
# the same questions on every machine: plenty of disk, and no swap yet
export JETLINK_TEST_FREE_GB=100 JETLINK_TEST_SWAPS=/tmp/swaps
# nothing waited on here is real, so there is nothing to wait for
export JETLINK_TEST_POLL_S=0
printf 'Filename\tType\tSize\tUsed\tPriority\n/dev/zram0 partition 1000000 0 5\n' >/tmp/swaps
PATH="$FAKE_BIN:$PATH"
OUT=/tmp/out.txt
UNITS=/etc/systemd/system
FAILED=0 PASSED=0 SCENARIO=''

ok() { PASSED=$((PASSED + 1)); }
fail() { FAILED=$((FAILED + 1)); printf '    FAIL [%s] %s\n' "$SCENARIO" "$*"; }
check() {  # check "failure message" command...: pass when the command succeeds
  local msg=$1
  shift
  if "$@"; then ok; else fail "$msg"; fi
}
refute() {  # refute "failure message" command...: pass when the command fails
  local msg=$1
  shift
  if "$@"; then fail "$msg"; else ok; fi
}
expect_out() { check "output lacks: $1" grep -qF -- "$1" "$OUT"; }
expect_no_out() { refute "output has: $1" grep -qF -- "$1" "$OUT"; }
expect_ran() { check "never ran: $1" grep -qF -- "$1" "$FAKE_LOG"; }
expect_not_ran() { refute "ran: $1" grep -qF -- "$1" "$FAKE_LOG"; }
expect_file() { check "missing file: $1" test -e "$1"; }
expect_no_file() { refute "file should be gone: $1" test -e "$1"; }
expect_in() { check "$1 lacks: $2" grep -qF -- "$2" "$1"; }
expect_not_in() { refute "$1 has: $2" grep -qF -- "$2" "$1"; }
expect_rc() { check "exit $RC, wanted $1" test "$RC" = "$1"; }
expect_link() { check "$1 points at $(readlink -f "$1" 2>/dev/null), wanted $2" test "$(readlink -f "$1" 2>/dev/null)" = "$2"; }
first_line() { grep -nF -- "$1" "$FAKE_LOG" | head -n 1 | cut -d: -f1; }
expect_before() {  # expect_before A B: the first run of A came before the first run of B
  local a b
  a="$(first_line "$1")" b="$(first_line "$2")"
  check "never ran: $1" test -n "$a"
  check "never ran: $2" test -n "$b"
  if [ -n "$a" ] && [ -n "$b" ]; then check "$1 ran after $2" test "$a" -lt "$b"; fi
}
apt_install() { printf 'apt-get -o DPkg::Lock::Timeout=900 -y install --no-install-recommends %s' "$1"; }

reset_box() {
  rm -rf /etc/jetlink /usr/local/lib/jetlink /usr/local/bin/jetlink /opt/jetlink /var/lib/jetlink /mnt/data \
    "$UNITS"/jetlink-* /etc/udev/rules.d/99-jetlink-usb-wakeup.rules \
    /etc/systemd/journald.conf.d/60-jetlink.conf "$FAKE_STATE" "$FAKE_LOG" "$FAKE_BIN" \
    /etc/nv_tegra_release /etc/nvpmodel.conf /tmp/dt-model /tmp/mem-sleep /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    "$LOCK"
  cp /tmp/fstab.orig /etc/fstab
  mkdir -p "$FAKE_STATE"
  echo "Linux version 6.8.0-fake (gcc) #1 SMP" >/tmp/proc-version
  mkdir -p "$FAKE_BIN" "$UNITS"
  local c
  for c in uname apt-get apt-cache dpkg dpkg-query ldconfig df systemctl journalctl nvpmodel ubuntu-drivers \
      udevadm fallocate mkswap swapon swapoff jetson_clocks curl gpg; do
    ln -sf "$SRC/tests/installer/fake.sh" "$FAKE_BIN/$c"
  done
  unset FAKE_ARCH FAKE_SMI FAKE_PUBLISHED FAKE_PM_REBOOT FAKE_SERVER_BROKEN FAKE_GPU_BROKEN \
    FAKE_TRT10 FAKE_NO_CURL FAKE_ROOT_FREE_GB FAKE_IMAGE_GB FAKE_DOWNLOAD_FAILS FAKE_BAD_SUM FAKE_NO_PLUGIN \
    FAKE_DOCKER_STUCK FAKE_TRT11_BUILDS
  export JETLINK_REPO_URL=file:///tmp/repo FAKE_LATEST=v0.10.0 JETLINK_TEST_SYSTEMD_RUN=/tmp
}

jetson() {  # jetson L4T_RELEASE REVISION
  printf '# R%s (release), REVISION: %s, GCID: 1, BOARD: generic, EABI: aarch64, DATE: now\n' "$1" "$2" \
    >/etc/nv_tegra_release
  printf 'NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super\0' >/tmp/dt-model
  echo 's2idle [deep]' >/tmp/mem-sleep
  cat >/etc/nvpmodel.conf <<'EOF'
< POWER_MODEL ID=0 NAME=15W >
< POWER_MODEL ID=1 NAME=25W >
< POWER_MODEL ID=2 NAME=MAXN_SUPER >
< POWER_MODEL ID=3 NAME=7W >
EOF
  # no TensorRT on the host: the Docker era had it only in the image
  export FAKE_ARCH=aarch64
  [ "$1" = 36 ] && export FAKE_TRT10=10.3.0.30-1+cuda12.5
  return 0
}

pc() {  # pc DRIVER
  export FAKE_ARCH=x86_64 FAKE_SMI="NVIDIA GeForce RTX 4070 Laptop GPU, $1, 8.9"
  ln -sf "$SRC/tests/installer/fake.sh" "$FAKE_BIN/nvidia-smi"
}

wsl() { echo "Linux version 6.6.87.2-microsoft-standard-WSL2 (root@fake) #1 SMP" >/tmp/proc-version; }

with_docker() { ln -sf "$SRC/tests/installer/fake.sh" "$FAKE_BIN/docker"; }

with_trt() {  # with_trt VERSION MAJOR: TensorRT already on the host
  echo "$1" >"$FAKE_STATE/pkg-libnvinfer$2"
  echo "$1" >"$FAKE_STATE/pkg-libnvonnxparsers$2"
}

answers() { printf '%b' "$1" >/tmp/answers; }

# run_installer curl|checkout "answers" [installer args...]: as curl | bash
# runs it, the script on stdin, or from the checkout; "" means no terminal
run_installer() {
  local how=$1 input=''
  if [ -n "$2" ]; then answers "$2"; input=/tmp/answers; fi
  shift 2
  if [ "$how" = curl ]; then
    JETLINK_INPUT=$input bash -s -- "$@" <"$SRC/install.sh" >"$OUT" 2>&1
  else
    JETLINK_INPUT=$input bash "$SRC/install.sh" "$@" </dev/null >"$OUT" 2>&1
  fi
  RC=$?
}

old_install() {  # old_install TAG [args...]: that Docker release's curl | bash, with --yes
  local tag=$1 rc
  shift
  FAKE_LATEST=$tag FAKE_PUBLISHED=1 bash <"/releases/$tag/install.sh" -s -- --yes "$@" >/tmp/old.txt 2>&1
  rc=$?
  check "the $tag install failed" test "$rc" = 0
  check "$tag left no Docker server" grep -q '^JETLINK_IMAGE=' /etc/jetlink/server.env
  [ "$rc" = 0 ] || sed 's/^/    | /' /tmp/old.txt
  : >"$FAKE_LOG"
}

cli() {  # cli ARGS...: the jetlink command, as installed
  jetlink "$@" >"$OUT" 2>&1
  RC=$?
}

# scenario "name": the one before shows its output if it failed, and this one
# starts with an empty command log
FAILED_BEFORE=0
scenario() {
  show_on_failure
  : >"$FAKE_LOG"
  FAILED_BEFORE=$FAILED SCENARIO="$1"
  printf '  %s\n' "$1"
}

show_on_failure() {
  [ -n "$SCENARIO" ] || return 0
  if [ "$FAILED" -gt "$FAILED_BEFORE" ] || [ -n "${SHOW_OUTPUT:-}" ]; then
    echo "    --- installer output ---"
    sed 's/^/    | /' "$OUT"
    echo "    --- commands run ---"
    sed 's/^/    | /' "$FAKE_LOG" 2>/dev/null | head -80
  fi
}

make_release() {  # make_release TAG VERSION [ASSET]: a tarball per arch with its .sha256, as CI publishes
  local tag=$1 ver=$2 asset=${3:-$2} arch d name
  mkdir -p "/tmp/releases/$tag"
  for arch in aarch64 x86_64; do
    d="$(mktemp -d)"
    mkdir -p "$d/bin" "$d/share/jetlink/systemd" "$d/share/jetlink/udev"
    ln -s "$SRC/tests/installer/fake.sh" "$d/bin/jetlink-server"
    echo "$ver" >"$d/VERSION"
    cp "$SRC/LICENSE" "$d/"
    cp "$SRC/scripts/jetlink-server.service" "$d/share/jetlink/systemd/"
    cp "$SRC"/scripts/*.rules "$d/share/jetlink/udev/"
    name="jetlink-server-$asset-linux-$arch.tar.gz"
    tar -czf "/tmp/releases/$tag/$name" -C "$d" .
    (cd "/tmp/releases/$tag" && sha256sum "$name" >"$name.sha256")
    rm -rf "$d"
  done
}

cp /etc/fstab /tmp/fstab.orig 2>/dev/null || : >/tmp/fstab.orig
# what `curl | bash` clones: the tree under test, committed
rm -rf /tmp/repo /tmp/releases /tmp/dev
git init -q -b main /tmp/repo
# -R, not -a: the bind-mounted tree belongs to the CI runner's user, and a repo
# owned by someone else is "dubious ownership" to the root git below
cp -R "$SRC/." /tmp/repo/
git -C /tmp/repo add -A
git -C /tmp/repo -c user.name=test -c user.email=test@example.invalid commit -qm "tree under test"
# releases, as git ls-remote sees them: v0.10.0 is the highest by number, not
# v0.9.0, and v0.11.0rc1 is a prerelease
for t in v0.9.0 v0.10.0 v0.11.0rc1; do git -C /tmp/repo tag "$t"; done
# the Docker releases, each its own commit, tagged as on GitHub
for dir in /releases/v*; do
  t="$(basename "$dir")"
  rm -rf "/tmp/old-$t" /tmp/old-index
  cp -R "$dir" "/tmp/old-$t"
  (cd "/tmp/old-$t" && GIT_DIR=/tmp/repo/.git GIT_WORK_TREE="/tmp/old-$t" GIT_INDEX_FILE=/tmp/old-index git add -A)
  tree="$(GIT_DIR=/tmp/repo/.git GIT_INDEX_FILE=/tmp/old-index git write-tree)"
  commit="$(git -C /tmp/repo -c user.name=test -c user.email=test@example.invalid commit-tree "$tree" -m "$t")"
  git -C /tmp/repo tag "$t" "$commit"
done
rm -f /tmp/old-index
# the server tarballs the fake GitHub hands out: two releases, and main's
# build on the edge prerelease
make_release v0.9.0 0.9.0
make_release v0.10.0 0.10.0
make_release edge 0.11.0-dev.1 edge
# a build of the kind the hardware bench installs by hand: a dev version, its
# files inside one folder, and a unit of its own
dev=/tmp/dev/jetlink-server-0.12.0-dev
mkdir -p "$dev/bin" "$dev/share/jetlink/systemd" "$dev/share/jetlink/udev"
ln -s "$SRC/tests/installer/fake.sh" "$dev/bin/jetlink-server"
echo 0.12.0-dev >"$dev/VERSION"
{ cat "$SRC/scripts/jetlink-server.service"; echo "# the 0.12.0-dev build's unit"; } >"$dev/share/jetlink/systemd/jetlink-server.service"
cp "$SRC"/scripts/*.rules "$dev/share/jetlink/udev/"
tar -czf /tmp/dev/jetlink-server-0.12.0-dev-linux-aarch64.tar.gz -C /tmp/dev jetlink-server-0.12.0-dev
# shellcheck disable=SC1091
. /etc/os-release
echo "installer scenarios on $PRETTY_NAME"
# NVIDIA's package source for this Ubuntu, as the installer picks it
DIST="ubuntu${VERSION_ID//./}"
# the TensorRT build a PC gets, which has to be the one build-linux.sh
# compiles the x86_64 server against
PC_TRT="$(sed -n 's/^PC_TRT=//p' "$SRC/install.sh")"
PC_TRT_INSTALL="$(apt_install "libnvinfer11=$PC_TRT libnvonnxparsers11=$PC_TRT libnvinfer-plugin11=$PC_TRT")"

scenario "JetPack 7.2 Jetson, always-on power, fresh install"
reset_box; jetson 39 2.1
# questions: power (1 = always on), let the comma shut it down, the status
# page's port (Enter), go ahead
run_installer curl '1\ny\n\ny\n'
expect_rc 0
expect_out "Orin Nano"
expect_out "JetPack 7 (Jetson Linux 39.2.1)"
expect_out "How is the Jetson powered in the car?"
expect_out "Which port should the status page use?"
expect_out "Install NVIDIA TensorRT from JetPack's package source"
expect_no_out "fastest power mode ("
expect_out "Jetlink is installed and running"
expect_out "Status page:"
expect_ran "$(apt_install "libnvinfer10 libnvonnxparsers10 libnvinfer-plugin10")"
# the plugins came with it
expect_not_ran "libnvinfer-plugin10="
expect_ran "apt-get -o DPkg::Lock::Timeout=900 -y clean"
expect_out "TensorRT 10.16.2.10"
refute "Docker or its toolkit was touched" grep -qE '^(docker|nvidia-ctk) |install .*(docker|nvidia-container)' "$FAKE_LOG"
expect_ran "https://github.com/zoompilot/jetlink/releases/download/v0.10.0/jetlink-server-0.10.0-linux-aarch64.tar.gz.sha256"
expect_ran "https://github.com/zoompilot/jetlink/releases/download/v0.10.0/jetlink-server-0.10.0-linux-aarch64.tar.gz"
expect_link /opt/jetlink/current /opt/jetlink/0.10.0
expect_before "jetlink-server backends --backend trt" "systemctl restart jetlink-server"
expect_out "The server can use the GPU: TensorRT 10.16.2.10 on Orin-sm87"
expect_ran "nvpmodel -m 2"
expect_in /etc/jetlink/server.env "JETLINK_CACHE_DIR=/mnt/data/jetlink"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_in /etc/jetlink/server.env "JETLINK_STATUS_PORT=5600"
expect_in /etc/jetlink/server.env "JETLINK_JETSON=1"
expect_in /etc/jetlink/server.env "JETLINK_FLAVOR=linux-aarch64"
expect_in /etc/jetlink/server.env "JETLINK_SERVER_VERSION=0.10.0"
expect_in /etc/jetlink/server.env "JETLINK_POWEROFF=--poweroff"
expect_not_in /etc/jetlink/server.env "JETLINK_IMAGE"
expect_in /etc/jetlink/install.conf "JETLINK_POWER=always"
expect_in /etc/jetlink/install.conf "JETLINK_POWEROFF_WITH_COMMA=1"
expect_in /etc/jetlink/install.conf "JETLINK_SOURCE=git"
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.10.0"
check "the unit is not the server's own" cmp -s "$UNITS/jetlink-server.service" /opt/jetlink/0.10.0/share/jetlink/systemd/jetlink-server.service
expect_in "$UNITS/jetlink-server.service.d/10-cache.conf" "RequiresMountsFor=/mnt/data/jetlink"
expect_in "$UNITS/jetlink-server.service.d/20-jetson-clocks.conf" "ExecStartPre=-/usr/bin/jetson_clocks"
expect_file /usr/local/bin/jetlink
expect_file /etc/udev/rules.d/99-jetlink-usb-wakeup.rules
expect_no_file /usr/local/lib/jetlink
expect_no_file "$UNITS/jetlink-poweroff.path"
expect_file /etc/systemd/journald.conf.d/60-jetlink.conf
expect_in /etc/fstab "/mnt/data/jetlink-swapfile none swap sw 0 0"
expect_file "$FAKE_STATE/masked-systemd-networkd-wait-online.service"
expect_ran "systemctl enable jetlink-server"
# the helper
jetlink status >/tmp/status.txt 2>&1
expect_in /tmp/status.txt "Jetlink is running"
expect_in /tmp/status.txt "server         0.10.0 (TensorRT 10.16.2.10)"
expect_in /tmp/status.txt "comma          not connected"
expect_in /tmp/status.txt "status page    http://"
expect_in /tmp/status.txt ".local:5600"
expect_in /tmp/status.txt "always on: sleeps when the car is off; the comma can shut it down"
jetlink models list >/dev/null 2>&1
expect_ran "jetlink-server models list --cache /mnt/data/jetlink"
jetlink models --help >/dev/null 2>&1
expect_ran "jetlink-server models --help"
# the installed unit's command line, with server.env's settings
jetlink run --log-level debug >/dev/null 2>&1
expect_ran "jetlink-server --usb --backend trt --cache /mnt/data/jetlink --sleep-after 120 --status-port 5600 --poweroff --log-level debug"
systemctl start jetlink-server

scenario "update keeps the answers and asks nothing"
# a native server that sleeps, with the awake lock it makes at start
: >"$LOCK" && chmod 644 "$LOCK"
cli update
expect_rc 0
expect_out "Getting the newest Jetlink (v0.10.0)"
expect_no_out "A few questions"
expect_no_out "Go ahead?"
expect_out "Jetlink is installed and running"
# it serves through the downloads, held awake so it cannot suspend the Jetson
# under them, and stops only for the switch
expect_out "Holding this computer awake for the update"
expect_ran "jetlink-server-0.10.0-linux-aarch64.tar.gz [held awake]"
expect_ran "apt-cache policy libnvinfer10 [held awake]"
expect_before "releases/download/v0.10.0" "systemctl stop jetlink-server"
expect_before "jetlink-server backends" "systemctl stop jetlink-server"
expect_out "Stopping the running Jetlink server for the update"
expect_ran "jetlink-server started: native, sleep 120"
expect_file /etc/jetlink/server.env.prev
expect_no_out "The previous Jetlink server is running again."
expect_in /etc/jetlink/install.conf "JETLINK_POWER=always"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_not_ran "nvpmodel -m"
# TensorRT is there; JetPack 7.2 only looks for a newer one
expect_not_ran "$(apt_install "libnvinfer10")"
expect_ran "apt-cache policy libnvinfer10"
expect_link /opt/jetlink/current /opt/jetlink/0.10.0
expect_no_file /opt/jetlink/previous

scenario "jetlink caffeinate holds the server awake: a command, -t, until stopped"
lock_free() { flock --exclusive --nonblock "$LOCK" true; }
# a command: held while it runs, and its exit status comes back
jetlink caffeinate sh -c "flock -xn $LOCK true && echo free || echo held; exit 7" >"$OUT" 2>&1; RC=$?
expect_rc 7
expect_out "held"
check "still held after the command" lock_free
jetlink caffeinate jetlink status >/tmp/status.txt 2>&1
expect_in /tmp/status.txt "held awake by jetlink caffeinate"
jetlink status >/tmp/status.txt 2>&1
expect_not_in /tmp/status.txt "held awake"
# -t SECONDS
jetlink caffeinate -t 2 >"$OUT" 2>&1 &
pid=$!
sleep 1
refute "not held during -t" lock_free
wait "$pid"; RC=$?
expect_rc 0
expect_out "Holding the Jetson awake for 2 s"
check "still held after -t" lock_free
# no arguments: until it is stopped
jetlink caffeinate >"$OUT" 2>&1 &
pid=$!
sleep 1
refute "not held" lock_free
kill -TERM "$pid"; wait "$pid"; RC=$?
expect_rc 0
expect_out "Holding the Jetson awake; Ctrl-C to let it sleep."
check "still held after it stopped" lock_free
# a server without the lock (from before it, or in Docker): nothing to hold
rm -f "$LOCK"
cli caffeinate
expect_rc 1
expect_out "The server is not running natively; nothing to hold."

scenario "a second run offers to keep the settings"
run_installer curl 'y\n'
expect_rc 0
expect_out "Jetlink is already installed. Keep your current settings and update it?"
expect_no_out "How is the Jetson powered in the car?"

scenario "a failed update puts the previous server back"
echo '# the previous settings' >>/etc/jetlink/server.env
echo "# 0.10.0's own unit" >>/opt/jetlink/0.10.0/share/jetlink/systemd/jetlink-server.service
# the new server never gets as far as waiting for the comma
export FAKE_SERVER_BROKEN=1
run_installer curl '' --update --ref v0.9.0
expect_rc 1
# systemd's third restart of it, seen as it is logged
expect_in /var/log/jetlink-install.log "the server keeps restarting"
expect_out "The previous Jetlink server is running again."
expect_in /etc/jetlink/server.env "# the previous settings"
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
expect_link /opt/jetlink/current /opt/jetlink/0.10.0
expect_in "$UNITS/jetlink-server.service" "# 0.10.0's own unit"
check "the previous server was not started again" test "$(grep -c "systemctl restart jetlink-server" "$FAKE_LOG")" -ge 2
refute "the unit was left stopped" test -f "$FAKE_STATE/stopped-jetlink-server"
unset FAKE_SERVER_BROKEN

scenario "a fresh install has no server to stop"
reset_box; jetson 39 2.1
run_installer curl '' --yes
expect_rc 0
expect_not_ran "systemctl stop jetlink-server"
expect_no_file /etc/jetlink/server.env.prev
# --yes takes the recommended wiring: always on, and the comma may shut it down
expect_in /etc/jetlink/install.conf "JETLINK_POWER=always"
expect_in /etc/jetlink/install.conf "JETLINK_POWEROFF_WITH_COMMA=1"
expect_in /etc/jetlink/server.env "JETLINK_STATUS_PORT=5600"

scenario "a server that cannot use the GPU fails with advice, before anything changes"
reset_box; jetson 39 2.1
FAKE_GPU_BROKEN=1 run_installer curl '' --yes
expect_rc 1
expect_out "TensorRT cannot run on this computer."
expect_out "The Jetlink server cannot use the GPU: no CUDA driver: libcuda.so.1: cannot open shared object file"
expect_no_out "ort: not usable"
expect_no_file "$UNITS/jetlink-server.service"
expect_no_file /opt/jetlink/current

scenario "curl | bash installs the newest release, as GitHub's API names it"
reset_box; jetson 39 2.1
FAKE_LATEST=v0.9.0 run_installer curl '' --yes
expect_rc 0
# the API's answer, not the highest tag
expect_out "Getting Jetlink (v0.9.0)"
expect_ran "releases/download/v0.9.0/jetlink-server-0.9.0-linux-aarch64.tar.gz"
expect_file /opt/jetlink/src/.git
expect_in /etc/jetlink/install.conf "JETLINK_SOURCE=git"
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.9.0"
expect_in /etc/jetlink/server.env "JETLINK_SERVER_VERSION=0.9.0"
jetlink status >/tmp/status.txt 2>&1
expect_in /tmp/status.txt "v0.9.0 (follows releases)"

scenario "jetlink update moves to the next release and keeps the one before"
cli update
expect_rc 0
expect_out "Getting the newest Jetlink (v0.10.0)"
expect_ran "releases/download/v0.10.0/jetlink-server-0.10.0-linux-aarch64.tar.gz"
expect_link /opt/jetlink/current /opt/jetlink/0.10.0
expect_link /opt/jetlink/previous /opt/jetlink/0.9.0
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.10.0"

scenario "an install that saved main before 0.5.0 follows releases; no API, so the tags"
# what the installer wrote before 0.5.0: its default, main, and no version
sed -i -e 's/^JETLINK_REF=.*/JETLINK_REF=main/' -e '/^JETLINK_VERSION=/d' /etc/jetlink/install.conf
unset FAKE_LATEST
run_installer curl '' --update
expect_rc 0
expect_out "Jetlink now follows releases; for development builds, use --ref main."
expect_ran "releases/latest"
expect_ran "releases/download/v0.10.0/jetlink-server-0.10.0-linux-aarch64.tar.gz"
expect_not_ran "0.11.0rc1"
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.10.0"

scenario "--ref main pins development builds from the edge prerelease, and an update keeps them"
export FAKE_LATEST=v0.10.0
run_installer curl '' --update --ref main
expect_rc 0
expect_ran "releases/download/edge/jetlink-server-edge-linux-aarch64.tar.gz"
expect_in /etc/jetlink/install.conf "JETLINK_REF=main"
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=main"
expect_link /opt/jetlink/current /opt/jetlink/0.11.0-dev.1
expect_link /opt/jetlink/previous /opt/jetlink/0.10.0
# older than the one before: gone
expect_no_file /opt/jetlink/0.9.0
: >"$FAKE_LOG"
run_installer curl '' --update
expect_rc 0
expect_no_out "now follows releases"
expect_ran "releases/download/edge/jetlink-server-edge-linux-aarch64.tar.gz"
expect_in /etc/jetlink/install.conf "JETLINK_REF=main"

scenario "--ref vX.Y.Z pins a release; --ref latest follows them again"
run_installer curl '' --update --ref v0.9.0
expect_rc 0
# a server that sleeps and has no awake lock (from before it) stops first
expect_before "systemctl stop jetlink-server" "releases/download/v0.9.0"
expect_not_ran "[held awake]"
expect_ran "releases/download/v0.9.0/jetlink-server-0.9.0-linux-aarch64.tar.gz"
expect_in /etc/jetlink/install.conf "JETLINK_REF=v0.9.0"
: >"$FAKE_LOG"
run_installer curl '' --update
expect_rc 0
expect_ran "releases/download/v0.9.0/"
: >"$FAKE_LOG"
run_installer curl '' --update --ref latest
expect_rc 0
expect_ran "releases/download/v0.10.0/"
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
expect_link /opt/jetlink/previous /opt/jetlink/0.9.0

scenario "no answer from GitHub: an update stays put, a first install stops"
unset FAKE_LATEST
export JETLINK_REPO_URL=file:///nonexistent
run_installer curl '' --update
expect_rc 0
expect_out "Could not look up the newest release; staying on v0.10.0."
expect_ran "releases/download/v0.10.0/"
reset_box; jetson 39 2.1
unset FAKE_LATEST
export JETLINK_REPO_URL=file:///nonexistent
run_installer curl '' --yes
expect_rc 1
expect_out "Could not find the newest Jetlink release."
expect_no_file /etc/jetlink
expect_not_ran "apt-get"

scenario "a download cut off part way is tried again"
reset_box; jetson 39 2.1
FAKE_DOWNLOAD_FAILS=2 run_installer curl '' --yes
expect_rc 0
expect_out "Downloading the Jetlink server (v0.10.0)"
expect_in /var/log/jetlink-install.log "the download was interrupted; trying again"
expect_in /etc/jetlink/server.env "JETLINK_SERVER_VERSION=0.10.0"

scenario "a damaged download is refused"
reset_box; jetson 39 2.1
FAKE_BAD_SUM=1 run_installer curl '' --yes
expect_rc 1
expect_out "its checksum does not match"
expect_no_file /opt/jetlink/current
expect_no_file "$UNITS/jetlink-server.service"

scenario "a branch with no ready-made server says how to build one"
reset_box; jetson 39 2.1
run_installer curl '' --yes --ref my-branch
expect_rc 1
expect_out "There is no ready-made Jetlink server for my-branch."
expect_out "scripts/build-linux.sh linux-aarch64"
expect_no_file /etc/jetlink
expect_not_ran "apt-get"

scenario "JetPack 7.2 follows the newest TensorRT, and refuses one too old"
reset_box; jetson 39 2.1
with_trt 10.16.2.10-1+cuda13.2 10
FAKE_TRT10=10.16.3.1-1+cuda13.2 run_installer curl '' --yes
expect_rc 0
expect_out "Updating TensorRT to 10.16.3.1"
expect_ran "apt-get -o DPkg::Lock::Timeout=900 -y install --only-upgrade --no-install-recommends libnvinfer10 libnvonnxparsers10 libnvinfer-plugin10"
expect_not_ran "$(apt_install "libnvinfer10")"
expect_out "TensorRT 10.16.3.1"
# the TensorRT it had came without plugins: they come now, the same build
expect_ran "$(apt_install "libnvinfer-plugin10=10.16.3.1-1+cuda13.2")"
check "wanted the package list fetched once" test "$(grep -c "apt-get .* update" "$FAKE_LOG")" = 1
reset_box; jetson 39 2.1
with_trt 10.16.1.1-1+cuda13.2 10
FAKE_TRT10=10.16.1.1-1+cuda13.2 run_installer curl '' --yes
expect_rc 1
expect_out "Jetlink needs 10.16.2.10 or newer"
expect_no_file "$UNITS/jetlink-server.service"

scenario "JetPack 6.2 Jetson, switched power"
reset_box; jetson 36 4.3
# questions: power (2 = switched), the status page's port (Enter), go ahead (Enter)
run_installer curl '2\n\n\n'
expect_rc 0
expect_out "JetPack 6 (Jetson Linux 36.4.3)"
expect_ran "$(apt_install "libnvinfer10 libnvonnxparsers10")"
expect_out "TensorRT 10.3.0.30"
# JetPack 6 stays on its TensorRT 10.3
expect_not_ran "apt-cache policy"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=0"
expect_in /etc/jetlink/server.env "JETLINK_FLAVOR=linux-aarch64"
expect_ran "nvpmodel -m 2"
expect_in /etc/fstab "/mnt/data/jetlink-swapfile none swap sw 0 0"
expect_no_file /etc/udev/rules.d/99-jetlink-usb-wakeup.rules
expect_file "$UNITS/jetlink-server.service.d/20-jetson-clocks.conf"
expect_in /etc/jetlink/server.env "JETLINK_POWEROFF=''"

scenario "jetlink setup changes the answers: power, and the status page off"
# questions: power (1 = always on), the comma may shut it down, port 0, go ahead
answers '1\ny\n0\ny\n'
JETLINK_INPUT=/tmp/answers jetlink setup >"$OUT" 2>&1; RC=$?
expect_rc 0
expect_out "How is the Jetson powered in the car?"
expect_no_out "Show a read-only status page"
expect_no_out "Status page:"
expect_in /etc/jetlink/install.conf "JETLINK_POWER=always"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_in /etc/jetlink/server.env "JETLINK_STATUS_PORT=0"
expect_in /etc/jetlink/server.env "JETLINK_POWEROFF=--poweroff"
expect_file /etc/udev/rules.d/99-jetlink-usb-wakeup.rules
jetlink status >/tmp/status.txt 2>&1
expect_not_in /tmp/status.txt "status page"
# the file an earlier installer wrote for "no" goes; one made by hand stays
echo "Written by the Jetlink installer: the comma may not power this computer off." >/mnt/data/jetlink/poweroff-dry-run
run_installer curl '' --update
expect_rc 0
expect_no_file /mnt/data/jetlink/poweroff-dry-run
touch /mnt/data/jetlink/poweroff-dry-run
run_installer curl '' --update
expect_rc 0
expect_file /mnt/data/jetlink/poweroff-dry-run

scenario "a power mode that needs a restart says so"
reset_box; jetson 39 2.1
FAKE_PM_REBOOT=1 run_installer curl '' --yes
expect_rc 0
expect_out "Restart this computer once"

scenario "PC with a driver too old for CUDA 13"
reset_box; pc 575.64.03
run_installer curl 'y\n'
expect_rc 0
expect_out "has NVIDIA driver 575.64.03 and needs 580 or newer"
expect_ran "ubuntu-drivers install nvidia:580-open"
expect_out "Restart the computer, then run the installer again"
expect_no_file /etc/jetlink/server.env

scenario "PC ready to go"
reset_box; pc 580.95.05
export FAKE_NO_CURL=1
# questions: start at boot, the status page's port (Enter), go ahead
run_installer curl 'y\n\ny\n'
expect_rc 0
expect_out "NVIDIA driver 580.95.05"
expect_no_out "How is the Jetson powered"
expect_out "Install NVIDIA TensorRT ${PC_TRT%%-*} from NVIDIA's package source"
check "install.sh's PC_TRT $PC_TRT is not the build build-linux.sh compiles against" \
  grep -qF "libnvinfer-headers-dev_${PC_TRT}_amd64.deb" "$SRC/scripts/build-linux.sh"
check "never installed libcurl4" grep -qE '^apt-get .* install --no-install-recommends .*libcurl4' "$FAKE_LOG"
expect_ran "https://developer.download.nvidia.com/compute/cuda/repos/$DIST/x86_64/cuda-keyring_1.1-1_all.deb"
expect_ran "dpkg -i"
# the CUDA 13 build, by its exact version, and never the meta packages
expect_ran "$PC_TRT_INSTALL"
refute "installed a TensorRT meta package" grep -qE 'install .*(tensorrt|cuda12\.9)' "$FAKE_LOG"
# for libcurl4, then again for NVIDIA's new package source
check "wanted the package list fetched twice" test "$(grep -c "apt-get .* update" "$FAKE_LOG")" = 2
expect_ran "releases/download/v0.10.0/jetlink-server-0.10.0-linux-x86_64.tar.gz"
expect_in /etc/jetlink/server.env "JETLINK_JETSON=0"
expect_in /etc/jetlink/server.env "JETLINK_CACHE_DIR=/var/lib/jetlink"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=0"
expect_in /etc/jetlink/server.env "JETLINK_FLAVOR=linux-x86_64"
expect_no_file /etc/systemd/journald.conf.d/60-jetlink.conf
expect_no_file "$UNITS/jetlink-server.service.d/20-jetson-clocks.conf"
expect_in /etc/jetlink/server.env "JETLINK_POWEROFF=''"
expect_out "Keep this computer plugged in and awake"
# a server that never sleeps serves until the new one is downloaded and checked
: >"$FAKE_LOG"
cli update
expect_rc 0
expect_before "releases/download/v0.10.0" "systemctl stop jetlink-server"
expect_before "jetlink-server backends" "systemctl stop jetlink-server"

scenario "uninstall removes it, keeps the models, and says how to remove TensorRT"
mkdir -p /var/lib/jetlink/models && echo x >/var/lib/jetlink/models/m.onnx
# questions: remove?, delete the models?
run_installer checkout 'y\nn\n' --uninstall
expect_rc 0
expect_out "Jetlink is removed."
expect_out "TensorRT stays installed; to remove it: sudo apt remove libnvinfer11 libnvonnxparsers11 libnvinfer-plugin11"
expect_no_file /etc/jetlink
expect_no_file /opt/jetlink
expect_no_file /usr/local/bin/jetlink
expect_no_file "$UNITS/jetlink-server.service"
expect_file /var/lib/jetlink/models/m.onnx
refute "removed a package" grep -qE '^apt-get .* (remove|purge)' "$FAKE_LOG"

scenario "Windows (WSL) is allowed, and marked untested"
reset_box; pc 580.95.05; wsl
run_installer curl '' --yes
expect_rc 0
expect_out "Windows (WSL) support is untested."
expect_out "usbipd"
expect_ran "$PC_TRT_INSTALL"
# a Linux driver or CUDA package inside WSL breaks the Windows driver's
refute "installed CUDA or a driver in WSL" grep -qE 'install .*(cuda |cuda-drivers|cuda-toolkit|nvidia-driver)' "$FAKE_LOG"
expect_in /etc/jetlink/install.conf "WSL"
reset_box; pc 575.64.03; wsl
run_installer curl '' --yes
expect_rc 1
expect_out "Update the NVIDIA driver in Windows"
expect_not_ran "ubuntu-drivers"
reset_box; pc 580.95.05; wsl
JETLINK_TEST_SYSTEMD_RUN=/nonexistent run_installer curl '' --yes
expect_rc 1
expect_out "systemd=true"

scenario "a PC whose package source lacks the server's TensorRT build stops"
reset_box; pc 580.95.05
FAKE_TRT11_BUILDS="11.3.0.99-1+cuda12.9 11.2.1.2-1+cuda13.3" run_installer curl '' --yes
expect_rc 1
expect_out "NVIDIA's package source has no TensorRT $PC_TRT."
expect_not_ran "libnvinfer11="
expect_no_file "$UNITS/jetlink-server.service"

scenario "dry run changes nothing"
reset_box; jetson 39 2.1
run_installer curl '' --yes --dry-run
expect_rc 0
expect_out "Dry run: stopping here. Nothing was changed."
expect_no_file /etc/jetlink
expect_no_file /opt/jetlink
expect_not_ran "apt-get"

scenario "JetPack 5 is refused with a way forward"
reset_box; jetson 35 6.0
run_installer checkout '' --yes
expect_rc 1
expect_out "Flash JetPack 7.2.1"

scenario "a Jetson that cannot deep-sleep defaults to switched power"
reset_box; jetson 39 2.1
echo 's2idle' >/tmp/mem-sleep
run_installer curl '' --yes
expect_rc 0
expect_in /etc/jetlink/install.conf "JETLINK_POWER=switched"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=0"

scenario "no swap on a disk too small for it, said plainly"
reset_box; jetson 39 2.1
JETLINK_TEST_FREE_GB=20 run_installer curl '' --yes
expect_rc 0
expect_out "Not enough disk space for 8 GB of swap"
refute "no swap should be added" grep -q swapfile /etc/fstab
expect_ran "nvpmodel -m 2"

scenario "no terminal and no --yes"
reset_box; jetson 39 2.1
run_installer checkout ''
expect_rc 1
expect_out "There is no terminal to ask questions in."

scenario "--binary installs a server built elsewhere"
reset_box; jetson 39 2.1
# from a checkout the installer has nothing to download
run_installer checkout '' --yes
expect_rc 1
expect_out "From a checkout, the installer installs a server you built"
run_installer checkout '' --yes --binary /tmp/releases/v0.10.0/jetlink-server-0.10.0-linux-x86_64.tar.gz
expect_rc 1
expect_out "is for another kind of computer; this one needs a linux-aarch64 build"
run_installer checkout '' --yes --binary /tmp/releases/v0.10.0/jetlink-server-0.10.0-linux-aarch64.tar.gz
expect_rc 0
expect_out "Install the Jetlink server from /tmp/releases/v0.10.0/jetlink-server-0.10.0-linux-aarch64.tar.gz"
expect_not_ran "releases/download"
expect_link /opt/jetlink/current /opt/jetlink/0.10.0
expect_in /etc/jetlink/install.conf "JETLINK_SOURCE=local"
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=local"
# run from the checkout again, it keeps the server it has
: >"$FAKE_LOG"
run_installer checkout '' --update
expect_rc 0
expect_out "Keep the Jetlink server that is installed"
expect_link /opt/jetlink/current /opt/jetlink/0.10.0

scenario "--binary over a release install leaves its source where it is"
reset_box; jetson 39 2.1
run_installer curl '' --yes
head_before="$(git -C /opt/jetlink/src rev-parse HEAD)"
: >"$FAKE_LOG"
bash /opt/jetlink/src/install.sh --update --binary /tmp/dev/jetlink-server-0.12.0-dev-linux-aarch64.tar.gz >"$OUT" 2>&1; RC=$?
expect_rc 0
check "the source moved" test "$(git -C /opt/jetlink/src rev-parse HEAD)" = "$head_before"
expect_not_ran "releases/latest"
expect_not_ran "releases/download"
expect_link /opt/jetlink/current /opt/jetlink/0.12.0-dev
expect_link /opt/jetlink/previous /opt/jetlink/0.10.0
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.10.0"
# the unit that came with the binary, not the source's
expect_in "$UNITS/jetlink-server.service" "# the 0.12.0-dev build's unit"
# a release's build says which release it is
run_installer curl '' --update --binary /tmp/releases/v0.9.0/jetlink-server-0.9.0-linux-aarch64.tar.gz
expect_rc 0
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.9.0"
expect_not_in "$UNITS/jetlink-server.service" "0.12.0-dev"

scenario "TensorRT already here without its plugins: they come, or it does without"
reset_box; jetson 36 4.3
with_trt 10.3.0.30-1+cuda12.5 10
run_installer curl '' --yes
expect_rc 0
expect_not_ran "$(apt_install "libnvinfer10")"
expect_out "Installing TensorRT's plugins"
expect_ran "$(apt_install "libnvinfer-plugin10=10.3.0.30-1+cuda12.5")"
# a package source without them: said, and not a failure
reset_box; jetson 36 4.3
with_trt 10.3.0.30-1+cuda12.5 10
FAKE_NO_PLUGIN=1 run_installer curl '' --yes
expect_rc 0
expect_out "Could not install TensorRT's plugins; Jetlink's models do not need them."
expect_out "Jetlink is installed and running"
# with them already there, nothing to do
reset_box; jetson 36 4.3
with_trt 10.3.0.30-1+cuda12.5 10
echo 10.3.0.30-1+cuda12.5 >"$FAKE_STATE/pkg-libnvinfer-plugin10"
run_installer curl '' --yes
expect_rc 0
expect_not_ran "install --no-install-recommends libnvinfer"

# From the Docker releases: each installed by its own installer, then moved by
# its own `jetlink update`, which runs this tree's installer.

scenario "a v0.4.3 JetPack 7.2 install moves out of Docker on jetlink update"
reset_box; jetson 39 2.1; with_docker
old_install v0.4.3 --ref v0.4.3
# what 0.4.x saved: main, its default, and no version
sed -i -e 's/^JETLINK_REF=.*/JETLINK_REF=main/' -e '/^JETLINK_VERSION=/d' /etc/jetlink/install.conf
# drop-ins of the user's: one runs docker, one does not
printf '[Service]\nExecStartPre=-/usr/bin/docker pull ghcr.io/zoompilot/jetlink:edge-cuda\n' >"$UNITS/jetlink-server.service.d/50-pull.conf"
printf '[Service]\nNice=-5\n' >"$UNITS/jetlink-server.service.d/60-nice.conf"
mkdir -p /mnt/data/jetlink/engines && echo plan >/mnt/data/jetlink/engines/abc.plan
echo '{"sha256": "abc"}' >/mnt/data/jetlink/last-loaded.json
cli update
expect_rc 0
expect_out "Jetlink now follows releases"
expect_out "Move Jetlink out of Docker"
expect_out "Jetlink is installed and running"
expect_out "It no longer runs in Docker"
# the Docker server serves until the native one is downloaded, has its
# TensorRT and has passed the GPU check
expect_before "$(apt_install "libnvinfer10 libnvonnxparsers10")" "systemctl stop jetlink-server"
expect_before "releases/download/v0.10.0/jetlink-server-0.10.0-linux-aarch64.tar.gz" "systemctl stop jetlink-server"
expect_before "jetlink-server backends" "systemctl stop jetlink-server"
expect_ran "docker rm -f jetlink"
# meanwhile it serves without sleeping, so it cannot suspend the Jetson under
# apt; the native server starts with the user's sleep
expect_out "Keeping this computer awake for the update"
expect_before "jetlink-server started: docker, sleep 0" "releases/download/v0.10.0"
expect_before "jetlink-server started: docker, sleep 0" "$(apt_install "libnvinfer10")"
expect_ran "jetlink-server started: native, sleep 120"
check "server.env.prev lost the sleep" grep -q '^JETLINK_SLEEP_AFTER=120' /etc/jetlink/server.env.prev
# its images go once the native server is up, and Docker stays
expect_before "jetlink-server started: native" "docker rmi"
expect_ran "docker rmi ghcr.io/zoompilot/jetlink:0.4.3-cuda"
refute "removed a package" grep -qE '^apt-get .* (remove|purge)' "$FAKE_LOG"
expect_in /etc/jetlink/server.env "JETLINK_CACHE_DIR=/mnt/data/jetlink"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_in /etc/jetlink/server.env "JETLINK_STATUS_PORT=5600"
expect_not_in /etc/jetlink/server.env "JETLINK_IMAGE"
expect_not_in /etc/jetlink/server.env "JETLINK_GPU_ARGS"
expect_in /etc/jetlink/server.env.prev "JETLINK_IMAGE="
# the answers, all kept
expect_in /etc/jetlink/install.conf "JETLINK_POWER=always"
expect_in /etc/jetlink/install.conf "JETLINK_POWEROFF_WITH_COMMA=1"
expect_in /etc/jetlink/server.env "JETLINK_POWEROFF=--poweroff"
expect_in /etc/jetlink/install.conf "JETLINK_AUTOSTART=1"
expect_in /etc/jetlink/install.conf "JETLINK_SWAP_FILE=/mnt/data/jetlink-swapfile"
expect_in /etc/jetlink/install.conf "JETLINK_MASKED_UNITS=systemd-networkd-wait-online.service"
expect_in /etc/jetlink/install.conf "JETLINK_JOURNALD_CAPPED=1"
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.10.0"
# the Jetson as it was set up
expect_not_ran "nvpmodel -m"
expect_not_ran "fallocate"
check "the swap file is in fstab once" test "$(grep -c jetlink-swapfile /etc/fstab)" = 1
expect_file /etc/udev/rules.d/99-jetlink-usb-wakeup.rules
expect_file /mnt/data/jetlink/engines/abc.plan
expect_file /mnt/data/jetlink/last-loaded.json
expect_ran "systemctl enable jetlink-server"
# the Docker setup, saved; the native one in its place
expect_in /etc/jetlink/docker-era/systemd/jetlink-server.service "run-server"
expect_file /etc/jetlink/docker-era/systemd/jetlink-server.service.d/50-pull.conf
expect_file /etc/jetlink/docker-era/lib/run-server
expect_in /etc/jetlink/docker-era/enabled "jetlink-poweroff.path"
expect_not_in /etc/jetlink/docker-era/enabled "jetlink-poweroff.service"
expect_out "Your drop-in 50-pull.conf runs Docker, so it is set aside"
expect_no_file "$UNITS/jetlink-server.service.d/50-pull.conf"
expect_file "$UNITS/jetlink-server.service.d/60-nice.conf"
expect_in "$UNITS/jetlink-server.service.d/10-cache.conf" "RequiresMountsFor=/mnt/data/jetlink"
expect_file "$UNITS/jetlink-server.service.d/20-jetson-clocks.conf"
expect_in "$UNITS/jetlink-server.service" "/opt/jetlink/current/bin/jetlink-server"
expect_no_file /usr/local/lib/jetlink
expect_no_file "$UNITS/jetlink-poweroff.path"
expect_no_file "$UNITS/jetlink-poweroff.service"
expect_ran "systemctl disable --now jetlink-poweroff.path"
expect_in /usr/local/bin/jetlink "SERVER=/opt/jetlink/current/bin/jetlink-server"
jetlink status >/tmp/status.txt 2>&1
expect_in /tmp/status.txt "server         0.10.0"

scenario "a v0.5.0 JetPack 6 install moves out of Docker"
reset_box; jetson 36 4.3; with_docker
old_install v0.5.0
cli update
expect_rc 0
expect_out "Jetlink is installed and running"
expect_ran "$(apt_install "libnvinfer10 libnvonnxparsers10 libnvinfer-plugin10")"
expect_out "TensorRT 10.3.0.30"
expect_not_ran "apt-cache policy"
expect_ran "docker rmi ghcr.io/zoompilot/jetlink:0.5.0-jetpack6"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_in /etc/jetlink/server.env "JETLINK_FLAVOR=linux-aarch64"
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.10.0"
expect_no_file /usr/local/lib/jetlink

scenario "a v0.5.0 JetPack 7.2 install short of room on / deletes its images first"
reset_box; jetson 39 2.1; with_docker
old_install v0.5.0
FAKE_ROOT_FREE_GB=2 FAKE_IMAGE_GB=8 cli update
expect_rc 0
expect_out "deleting Jetlink's Docker images first"
expect_before "systemctl stop jetlink-server" "docker rmi"
expect_before "docker rmi" "$(apt_install "libnvinfer10 libnvonnxparsers10")"
check "an image is left" test ! -s "$FAKE_STATE/images"
expect_out "Jetlink is installed and running"
# and without room even then: nothing moved, and the way back said
reset_box; jetson 39 2.1; with_docker
old_install v0.5.0
FAKE_ROOT_FREE_GB=0 FAKE_IMAGE_GB=1 cli update
expect_rc 1
expect_out "Not enough free space on / for TensorRT"
expect_out "jetlink update --ref v0.6.0"
expect_not_ran "$(apt_install "libnvinfer10")"
expect_in "$UNITS/jetlink-server.service" "run-server"
expect_in /etc/jetlink/server.env "JETLINK_IMAGE="
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"

scenario "a v0.6.0 PC install moves out of Docker"
reset_box; pc 580.95.05; with_docker
old_install v0.6.0
cli update
expect_rc 0
expect_out "Jetlink is installed and running"
expect_ran "$PC_TRT_INSTALL"
expect_ran "releases/download/v0.10.0/jetlink-server-0.10.0-linux-x86_64.tar.gz"
expect_ran "docker rmi ghcr.io/zoompilot/jetlink:0.6.0-cuda"
refute "removed the toolkit" grep -qE '^apt-get .* (remove|purge)' "$FAKE_LOG"
expect_in /etc/jetlink/server.env "JETLINK_JETSON=0"
expect_in /etc/jetlink/server.env "JETLINK_CACHE_DIR=/var/lib/jetlink"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=0"
expect_in /etc/jetlink/server.env "JETLINK_FLAVOR=linux-x86_64"
expect_in /etc/jetlink/install.conf "JETLINK_AUTOSTART=1"
expect_no_file "$UNITS/jetlink-server.service.d/20-jetson-clocks.conf"
expect_in /etc/jetlink/server.env "JETLINK_POWEROFF=''"

scenario "a failed move puts the Docker server back, and the next update finishes it"
reset_box; jetson 39 2.1; with_docker
old_install v0.6.0
# the status page's own unit, from before it moved into the server
printf '[Service]\nExecStart=/usr/bin/python3 /usr/local/lib/jetlink/web/jetlink_web.py\n[Install]\nWantedBy=multi-user.target\n' \
  >"$UNITS/jetlink-web.service"
# a failure while it serves without sleeping puts the sleep back
FAKE_BAD_SUM=1 cli update
expect_rc 1
expect_ran "jetlink-server started: docker, sleep 0"
check "not started again with its sleep" test "$(grep 'jetlink-server started' "$FAKE_LOG" | tail -n 1)" = "jetlink-server started: docker, sleep 120"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_out "The previous Jetlink server is running again."
: >"$FAKE_LOG"
export FAKE_SERVER_BROKEN=1
cli update
expect_rc 1
expect_ran "jetlink-server started: docker, sleep 0"
check "not started again with its sleep" test "$(grep 'jetlink-server started' "$FAKE_LOG" | tail -n 1)" = "jetlink-server started: docker, sleep 120"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_out "The previous Jetlink server is running again."
expect_in "$UNITS/jetlink-server.service" "run-server"
expect_no_file "$UNITS/jetlink-server.service.d/20-jetson-clocks.conf"
expect_in /etc/jetlink/server.env "JETLINK_IMAGE="
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.6.0"
expect_file /usr/local/lib/jetlink/run-server
expect_in /usr/local/bin/jetlink "docker run"
expect_file "$UNITS/jetlink-poweroff.path"
expect_file "$UNITS/jetlink-web.service"
expect_ran "systemctl enable --now jetlink-poweroff.path"
expect_ran "systemctl enable --now jetlink-web.service"
expect_not_ran "docker rmi"
check "the Docker server was not started again" test "$(grep -c "systemctl restart jetlink-server" "$FAKE_LOG")" -ge 2
refute "the unit was left stopped" test -f "$FAKE_STATE/stopped-jetlink-server"
unset FAKE_SERVER_BROKEN
: >"$FAKE_LOG"
cli update
expect_rc 0
expect_out "Jetlink is installed and running"
expect_not_in /etc/jetlink/server.env "JETLINK_IMAGE"
expect_no_file "$UNITS/jetlink-web.service"
expect_ran "systemctl disable --now jetlink-web.service"
expect_ran "docker rmi ghcr.io/zoompilot/jetlink:0.6.0-cuda"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"

scenario "a Docker server that will not stop keeps serving, and no native one starts beside it"
reset_box; jetson 39 2.1; with_docker
old_install v0.6.0
FAKE_DOCKER_STUCK=1 cli update
expect_rc 1
expect_out "The Docker server did not stop."
expect_before "docker rm -f jetlink" "docker ps -q --filter name=^/?jetlink$"
expect_not_ran "jetlink-server started: native"
expect_in "$UNITS/jetlink-server.service" "run-server"
expect_in /etc/jetlink/server.env "JETLINK_IMAGE="
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_out "The previous Jetlink server is running again."
expect_not_ran "docker rmi"

scenario "going back to v0.6.0 runs its own installer, and the curl line comes forward"
FAKE_PUBLISHED=1 cli update --ref v0.6.0
expect_rc 0
expect_out "Go back to v0.6.0, which runs Jetlink in Docker"
expect_out "v0.6.0 runs Jetlink in Docker; its own installer takes over from here."
expect_ran "docker pull ghcr.io/zoompilot/jetlink:0.6.0-cuda"
expect_in "$UNITS/jetlink-server.service" "run-server"
expect_in /etc/jetlink/server.env "JETLINK_IMAGE_REF=ghcr.io/zoompilot/jetlink:0.6.0-cuda"
# the answers the native install kept reach it
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_in /etc/jetlink/install.conf "JETLINK_REF=v0.6.0"
expect_in /etc/jetlink/install.conf "JETLINK_POWER=always"
expect_in /etc/jetlink/install.conf "JETLINK_POWEROFF_WITH_COMMA=1"
expect_file "$UNITS/jetlink-poweroff.path"
expect_file /opt/jetlink/0.10.0/bin/jetlink-server
: >"$FAKE_LOG"
run_installer curl '' --update --ref latest
expect_rc 0
expect_out "Move Jetlink out of Docker"
expect_not_in /etc/jetlink/server.env "JETLINK_IMAGE"
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
expect_in "$UNITS/jetlink-server.service" "/opt/jetlink/current/bin/jetlink-server"

scenario "uninstall after a move removes the Docker leftovers too"
echo "jetlink:local-cuda" >>"$FAKE_STATE/images"
mkdir -p /mnt/data/jetlink/engines && echo plan >/mnt/data/jetlink/engines/abc.plan
# questions: remove?, delete the old images?, delete the models?
run_installer checkout 'y\ny\nn\n' --uninstall
expect_rc 0
expect_out "Jetlink is removed."
expect_ran "docker rmi jetlink:local-cuda"
expect_no_file /etc/jetlink
expect_no_file /opt/jetlink
expect_out "sudo apt remove libnvinfer10 libnvonnxparsers10 libnvinfer-plugin10"
expect_file /mnt/data/jetlink/engines/abc.plan
show_on_failure

echo
if [ "$FAILED" -eq 0 ]; then
  echo "installer scenarios: $PASSED checks passed"
else
  echo "installer scenarios: $FAILED failed, $PASSED passed"
  exit 1
fi
