#!/usr/bin/env bash
# Jetlink installer, for an NVIDIA Jetson or a Linux PC with an NVIDIA GPU.
#
#   curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/main/install.sh | bash
#
# It checks the computer, asks a few questions, installs NVIDIA's TensorRT if
# it is missing, gets the Jetlink server, and starts it at boot. Running it
# again is safe: it offers to keep your answers and brings everything up to
# date. An install that ran the server in Docker moves to the native server
# and keeps its answers, models and engines. Afterwards the `jetlink` command
# manages the install.
#
# Options, for support and scripts; the questions cover everything else:
#   --yes            take the recommended answer to every question
#   --update         keep the saved answers and update, asking nothing
#   --reconfigure    ask the questions again
#   --ref REF        a tag or branch (default: latest, the newest release)
#   --binary FILE    install this server tarball instead of downloading one
#   --dry-run        check and ask, then show the plan without changing anything
#   --uninstall      remove Jetlink
#
# Everything runs from main(), called on the last line, so a download cut off
# part way through runs nothing, and nothing reading stdin can eat the script.
set -Eeuo pipefail

REPO_URL="${JETLINK_REPO_URL:-https://github.com/zoompilot/jetlink.git}"
RAW_URL=https://raw.githubusercontent.com/zoompilot/jetlink
API_URL=https://api.github.com/repos/zoompilot/jetlink
RELEASES_URL=https://github.com/zoompilot/jetlink/releases/download
ETC_DIR=/etc/jetlink
CONF="$ETC_DIR/install.conf"
ENV_FILE="$ETC_DIR/server.env"
# what the Docker era installed, kept for a failed move and for going back by hand
DOCKER_ERA_DIR="$ETC_DIR/docker-era"
BIN=/usr/local/bin/jetlink
# the Docker era's launcher and helpers; nothing is installed here any more
LIB_DIR=/usr/local/lib/jetlink
# a directory per server version, `current` and `previous` links to two of
# them, and `src`, the source the jetlink command updates from
SRC_ROOT=/opt/jetlink
UNIT_DIR=/etc/systemd/system
UNIT=jetlink-server
# in a server's directory: the unit and udev rules built with it
SHARE=share/jetlink
# what the Docker-era installers made (the status page's own unit was never
# released, but a bench has it): a move from Docker saves these, and a failed
# one puts them back
DOCKER_ERA_UNITS="$UNIT.service $UNIT.service.d jetlink-poweroff.path jetlink-poweroff.service jetlink-web.service jetlink-web.service.d"
CLOCKS_DROPIN="$UNIT_DIR/$UNIT.service.d/20-jetson-clocks.conf"
WAKE_RULE=/etc/udev/rules.d/99-jetlink-usb-wakeup.rules
JOURNALD_DROPIN=/etc/systemd/journald.conf.d/60-jetlink.conf
# the last release that ran in Docker: `jetlink update --ref` it to go back
DOCKER_LAST=v0.6.0
# CUDA 13 needs driver 580; TensorRT needs a Turing (7.5) or newer GPU
MIN_DRIVER=580
MIN_CC=75
MIN_DISK_GB=15
SWAP_GB=8
# JetPack 7.2 runs the newest TensorRT the Jetson repository has, and nothing
# older than this.
JP7_MIN_TRT=10.16.2.10
# PCs get no TensorRT package: NVIDIA's x86_64 runtime wheel of the build the
# server is compiled against (what pip's tensorrt-cu13==11.3.0.99 fetches from
# NVIDIA's index) is unpacked into a directory of its own, which every server
# version shares and the unit names with --tensorrt-libs
PC_TRT=11.3.0.99
PC_TRT_WHEEL=https://pypi.nvidia.com/tensorrt-cu13-libs/tensorrt_cu13_libs-11.3.0.99-py3-none-manylinux_2_28_x86_64.whl
PC_TRT_SHA256="${JETLINK_TEST_TRT_SHA256:-cef957e0b48a525f2bcc8849742de06883cedcc6b96185b9cf0b17c9c75a858a}"
TRT_ROOT="$SRC_ROOT/tensorrt"
PC_TRT_DIR="$TRT_ROOT/$PC_TRT"
# free space that installing TensorRT takes, the download and the files
# together: NVIDIA's package indexes give libnvinfer, its ONNX parser and its
# plugins as 2.33 GB + 3.05 GB on JetPack 7.2 and 0.11 + 0.40 on 6.2; a PC's
# wheel is 3.81 GB, and the libraries unpacked from it 2.69 GB
TRT_GB_JP7=6
TRT_GB_JP6=1
TRT_GB_PC=7
# units that hold up boot waiting for a network the car does not have
WAIT_ONLINE_UNITS="systemd-networkd-wait-online.service NetworkManager-wait-online.service"
# where detection looks; the installer's tests point these at fakes
DT_MODEL="${JETLINK_TEST_DT_MODEL:-/proc/device-tree/model}"
MEM_SLEEP="${JETLINK_TEST_MEM_SLEEP:-/sys/power/mem_sleep}"
SWAPS="${JETLINK_TEST_SWAPS:-/proc/swaps}"
PROC_VERSION="${JETLINK_TEST_PROC_VERSION:-/proc/version}"
SYSTEMD_RUN="${JETLINK_TEST_SYSTEMD_RUN:-/run/systemd/system}"
AWAKE_LOCK="${JETLINK_TEST_AWAKE_LOCK:-/run/jetlink-awake.lock}"
# seconds between looks at something the installer waits on
POLL_S="${JETLINK_TEST_POLL_S:-5}"
# A new server has 3 minutes to say it serves, and then has to stay up. One
# that loads the model it ran last is watched until that is done: a TensorRT
# that changed makes it rebuild the plan, which takes minutes, so the 20
# minute wait is only how long the installer watches, and the server serves
# all the while.
READY_S=180
SETTLE_S="${JETLINK_TEST_SETTLE_S:-10}"
PRELOAD_S="${JETLINK_TEST_PRELOAD_S:-1200}"

OPT_YES=0 OPT_UPDATE=0 OPT_RECONFIGURE=0 OPT_DRY_RUN=0 OPT_UNINSTALL=0
OPT_REF="" OPT_BINARY=""
# as given, for the installer of an older release to take over with
ARGS=()

# ---------------------------------------------------------------------------
# Output

B='' D='' G='' Y='' R='' N=''
setup_colors() {
  if [ -t 1 ] && [ "${TERM:-dumb}" != dumb ]; then
    B=$'\e[1m' D=$'\e[2m' G=$'\e[32m' Y=$'\e[33m' R=$'\e[31m' N=$'\e[0m'
  fi
}
say() { printf '%s\n' "$*"; }
good() { printf '  %s✓%s %s\n' "$G" "$N" "$*"; }
note() { printf '  %s!%s %s\n' "$Y" "$N" "$*"; }
bad() { printf '  %s✗%s %s\n' "$R" "$N" "$*"; }
heading() { printf '\n%s%s%s\n' "$B" "$*" "$N"; }
die() {
  local first="$1"
  shift
  printf '\n%s%s%s\n' "$R$B" "$first" "$N"
  local line
  for line in "$@"; do printf '  %s\n' "$line"; done
  printf '\n'
  # a second Ctrl-C must not cut the way back short
  trap '' INT TERM HUP
  restore_previous_server
  save_log
  exit 1
}

LOG="${TMPDIR:-/tmp}/jetlink-install.$$.log"
save_log() {
  [ -s "$LOG" ] || return 0
  if [ "$OPT_DRY_RUN" != 1 ] && as_root cp "$LOG" /var/log/jetlink-install.log 2>/dev/null; then
    printf '  %sThe full log is in /var/log/jetlink-install.log%s\n\n' "$D" "$N"
  else
    printf '  %sThe full log is in %s%s\n\n' "$D" "$LOG" "$N"
  fi
}

on_error() {
  local rc=$? line=$1
  trap - ERR
  printf '\n%sSomething went wrong (line %s, exit %s).%s\n' "$R$B" "$line" "$rc" "$N"
  if [ -s "$LOG" ]; then
    printf '  Last lines of the log:\n'
    tail -n 15 "$LOG" | sed 's/^/    /'
  fi
  printf '\n  Running the installer again is safe. If it keeps failing, open an issue at\n'
  printf '  https://github.com/zoompilot/jetlink/issues with the log attached.\n\n'
  trap '' INT TERM HUP
  restore_previous_server
  save_log
  exit "$rc"
}

# Ctrl-C, a closed terminal or ssh session, or a kill: the way back a failure
# takes, then out with the signal's status. A closed terminal takes the output
# with it, so from then on everything goes to the log.
on_signal() {
  local sig=$1 rc=$2
  trap '' INT TERM HUP
  trap - ERR
  set +e
  if [ "$sig" = HUP ]; then exec >>"$LOG" 2>&1; else exec 1>&4 2>&5; fi
  printf '\n%sStopped by %s.%s\n' "$R$B" "SIG$sig" "$N"
  restore_previous_server
  save_log
  exit "$rc"
}

# `run_step LABEL cmd...`: run a command with its output in the log, and a
# spinner with the elapsed time so a 20 minute download still looks alive.
# Returns the command's status; `step` stops the installer on a failure.
run_step() {
  local label="$1"
  shift
  if [ "$OPT_DRY_RUN" = 1 ]; then
    printf '  %s·%s %s\n' "$D" "$N" "$label"
    return 0
  fi
  printf '\n==> %s\n' "$label" >>"$LOG"
  local rc=0 t0=$SECONDS
  if [ -t 1 ]; then
    # a failure is the step's to report: the ERR trap, inherited, would put
    # the old server back from in here, and then the step's die again
    ( trap - ERR; "$@" ) </dev/null >>"$LOG" 2>&1 &
    local pid=$! i=0 frames=$'|/-\\'
    while kill -0 "$pid" 2>/dev/null; do
      printf '\r  %s%s%s %s %s%s%s ' "$D" "${frames:i++%4:1}" "$N" "$label" "$D" "$(elapsed $((SECONDS - t0)))" "$N"
      sleep 0.25
    done
    wait "$pid" || rc=$?
    printf '\r\e[K'
  else
    "$@" </dev/null >>"$LOG" 2>&1 || rc=$?
  fi
  if [ "$rc" -eq 0 ]; then
    if [ $((SECONDS - t0)) -ge 2 ]; then
      good "$label ${D}$(elapsed $((SECONDS - t0)))${N}"
    else
      good "$label"
    fi
    return 0
  fi
  bad "$label"
  return "$rc"
}

step() {
  run_step "$@" && return 0
  printf '\n  Last lines of the log:\n'
  tail -n 20 "$LOG" | sed 's/^/    /'
  die "That step failed." "Running the installer again is safe." \
    "If it keeps failing, open an issue at https://github.com/zoompilot/jetlink/issues"
}

elapsed() {
  local s=$1
  if [ "$s" -lt 60 ]; then printf '%ss' "$s"; else printf '%dm%02ds' $((s / 60)) $((s % 60)); fi
}

# a >= b, for version numbers
version_ge() {
  [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n 1)" = "$1" ]
}

# ---------------------------------------------------------------------------
# Questions. Answers are read from the terminal, not stdin, which is the
# script itself under curl | bash.

INTERACTIVE=0
open_input() {
  if [ -n "${JETLINK_INPUT:-}" ]; then  # the installer's tests
    exec 3<"$JETLINK_INPUT"
    INTERACTIVE=1
  elif [ "$OPT_YES" = 1 ] || [ "$OPT_UPDATE" = 1 ]; then
    INTERACTIVE=0
  elif [ -r /dev/tty ] && (exec 3</dev/tty) 2>/dev/null; then
    exec 3</dev/tty
    INTERACTIVE=1
  else
    die "There is no terminal to ask questions in." \
      "Run this in a terminal, or add --yes to take the recommended answers:" \
      "  curl -fsSL $RAW_URL/main/install.sh | bash -s -- --yes"
  fi
}

# The helpers below set the caller's variable by name, so their own locals are
# all __-prefixed: a local with the caller's name would take the answer.
read_answer() {
  local __ra_reply=''
  IFS= read -r __ra_reply <&3 || true
  printf -v "$1" '%s' "$__ra_reply"
}

# ask_intro "question" ["explanation"...]
ask_intro() {
  printf '\n  %s%s%s\n' "$B" "$1" "$N"
  shift
  local __in_line
  for __in_line in "$@"; do printf '  %s%s%s\n' "$D" "$__in_line" "$N"; done
}

# ask_number VAR default MIN MAX: until a number from MIN to MAX, or Enter
ask_number() {
  local __nb_a
  while true; do
    printf '  Type a number and press Enter [%s] ' "$2"
    read_answer __nb_a
    [ -z "$__nb_a" ] && __nb_a=$2
    if [[ "$__nb_a" =~ ^[0-9]{1,5}$ ]] && [ "$((10#$__nb_a))" -ge "$3" ] && [ "$((10#$__nb_a))" -le "$4" ]; then break; fi
    printf '  Please type a number from %d to %d.\n' "$3" "$4"
  done
  printf -v "$1" '%s' "$((10#$__nb_a))"
}

# ask_yn VAR default(y|n) "question" ["explanation"...]
ask_yn() {
  local __yn_var=$1 __yn_def=$2
  shift 2
  if [ "$INTERACTIVE" != 1 ]; then
    printf -v "$__yn_var" '%s' "$__yn_def"
    return
  fi
  ask_intro "$@"
  local __yn_hint="Y/n"
  [ "$__yn_def" = n ] && __yn_hint="y/N"
  local __yn_a
  while true; do
    printf '  [%s] ' "$__yn_hint"
    read_answer __yn_a
    __yn_a="$(printf '%s' "$__yn_a" | tr '[:upper:]' '[:lower:]')"
    case "$__yn_a" in
      '') __yn_a=$__yn_def; break ;;
      y|yes) __yn_a=y; break ;;
      n|no) __yn_a=n; break ;;
      *) printf '  Please type y or n.\n' ;;
    esac
  done
  printf -v "$__yn_var" '%s' "$__yn_a"
}

# ask_choice VAR default_number "question" "option 1" "option 2" ...
ask_choice() {
  local __ch_var=$1 __ch_def=$2
  shift 2
  if [ "$INTERACTIVE" != 1 ]; then
    printf -v "$__ch_var" '%s' "$__ch_def"
    return
  fi
  ask_intro "$1"
  shift
  local __ch_i=1 __ch_opt
  for __ch_opt in "$@"; do
    printf '    %s%d)%s %s\n' "$B" "$__ch_i" "$N" "$__ch_opt"
    __ch_i=$((__ch_i + 1))
  done
  ask_number "$__ch_var" "$__ch_def" 1 $#
}

# ask_port VAR default "question" ["explanation"...]: 0 to 65535
ask_port() {
  local __pt_var=$1 __pt_def=$2
  shift 2
  if [ "$INTERACTIVE" != 1 ]; then
    printf -v "$__pt_var" '%s' "$__pt_def"
    return
  fi
  ask_intro "$@"
  ask_number "$__pt_var" "$__pt_def" 0 65535
}

# ---------------------------------------------------------------------------
# Root. The script runs as the user so curl | bash works without sudo, and
# asks sudo for each change.

SUDO=''
as_root() {
  if [ -z "$SUDO" ]; then "$@"; else sudo -n -- "$@"; fi
}

get_root() {
  if [ "$(id -u)" -eq 0 ]; then
    SUDO=''
    return
  fi
  command -v sudo >/dev/null 2>&1 || die "Jetlink needs administrator rights to install, and sudo is missing." \
    "Run the installer as root instead."
  SUDO=sudo
  if ! sudo -n true 2>/dev/null; then
    say ""
    say "  Jetlink needs administrator rights to install. Enter your password if asked."
    # the password prompt comes from the terminal, not the piped script
    # shellcheck disable=SC2024
    sudo -v </dev/tty || die "Could not get administrator rights."
  fi
  # a download can outlast sudo's 15 minutes
  ( while kill -0 $$ 2>/dev/null; do sudo -n true 2>/dev/null; sleep 50; done ) &
}

# root_write PATH MODE: stdin into a root-owned file
root_write() {
  local path=$1 mode=${2:-644} tmp
  tmp="$(mktemp)"
  cat >"$tmp"
  as_root install -D -m "$mode" "$tmp" "$path"
  rm -f "$tmp"
}

apt_get() {
  # a fresh Jetson runs unattended-upgrades for a while after its first boot
  as_root env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 -y "$@"
}

# the package list, once a run; a new package source clears APT_UPDATED
APT_UPDATED=0
apt_update() {
  [ "$APT_UPDATED" = 1 ] && return 0
  step "Getting the package list" apt_get update
  APT_UPDATED=1
}

# the loader's cache holds this library
has_lib() {
  local libs
  libs="$({ ldconfig -p || /sbin/ldconfig -p; } 2>/dev/null || true)"
  [[ $libs == *"$1 "* ]]
}

pkg_version() {
  dpkg-query -W -f '${Version}' "$1" 2>/dev/null || true
}

# on the filesystem that holds $1, or will once it is made
free_gb() {
  local where=$1
  while [ ! -d "$where" ]; do where="$(dirname "$where")"; done
  df -Pk "$where" 2>/dev/null | awk 'NR == 2 {printf "%d", $4 / 1048576}'
}

# ---------------------------------------------------------------------------
# What this computer is

ARCH='' OS_ID='' OS_NAME=''
JETSON=0 L4T='' L4T_MAJOR=0 L4T_MINOR=0 JETPACK='' JP_MAJOR=0 MODEL='' WSL=0
GPU_NAME='' DRIVER='' DRIVER_MAJOR=0 GPU_CC=0 GPU_PRESENT=0
# FLAVOR names the server build: linux-aarch64 (Jetson) or linux-x86_64 (PC)
FLAVOR='' PLATFORM_NAME=''
TRT_GB=0 TRT_PRESENT=0 TRT_VERSION=''
# the flags that point the server at a PC's TensorRT; none on a Jetson
TRT_ARGS=()
DISK_GB=0
DEEP_SLEEP=0
PM_BEST_ID='' PM_BEST_NAME='' PM_CURRENT=''

detect() {
  [ "$(uname -s)" = Linux ] || die "This installer is for Linux: a Jetson, or a PC running Linux." \
    "On a Mac, use the Jetlink app from https://github.com/zoompilot/jetlink/releases"
  if grep -qi microsoft "$PROC_VERSION" 2>/dev/null; then WSL=1; fi
  ARCH="$(uname -m)"
  if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    OS_ID="${ID:-}" OS_NAME="${PRETTY_NAME:-Linux}"
  fi
  command -v apt-get >/dev/null 2>&1 || die "This installer needs Ubuntu or Debian (apt)." \
    "See https://github.com/zoompilot/jetlink/blob/main/docs/platforms.md for other systems."

  if [ -f /etc/nv_tegra_release ] || grep -qa tegra /proc/device-tree/compatible 2>/dev/null; then
    JETSON=1
    detect_jetson
    if has_lib libnvinfer.so.10 && has_lib libnvonnxparser.so.10; then
      TRT_PRESENT=1 TRT_VERSION="$(pkg_version libnvinfer10)"
    fi
  else
    detect_pc
    if [ -f "$PC_TRT_DIR/libnvinfer.so.11" ] && [ -f "$PC_TRT_DIR/libnvonnxparser.so.11" ]; then
      TRT_PRESENT=1 TRT_VERSION=$PC_TRT
    fi
  fi
}

detect_jetson() {
  local line rev
  line="$(head -n 1 /etc/nv_tegra_release 2>/dev/null || true)"
  if [[ "$line" =~ R([0-9]+)\ \(release\),\ REVISION:\ ([0-9.]+) ]]; then
    L4T_MAJOR="${BASH_REMATCH[1]}"
    rev="${BASH_REMATCH[2]}"
  else
    # r39 and later may drop the file; the core package has the version
    rev="$(pkg_version nvidia-l4t-core)"
    L4T_MAJOR="${rev%%.*}"
    rev="${rev#*.}"
    rev="${rev%%-*}"
  fi
  L4T_MINOR="${rev%%.*}"
  [[ "$L4T_MAJOR" =~ ^[0-9]+$ ]] || L4T_MAJOR=0
  [[ "$L4T_MINOR" =~ ^[0-9]+$ ]] || L4T_MINOR=0
  L4T="$L4T_MAJOR.$rev"
  MODEL="$(tr -d '\0' <"$DT_MODEL" 2>/dev/null || echo "NVIDIA Jetson")"
  MODEL="${MODEL/ Engineering Reference Developer Kit/}"

  case "$L4T_MAJOR" in
    39)
      [ "$L4T_MINOR" -ge 2 ] || die "This Jetson runs Jetson Linux $L4T; Jetlink needs JetPack 7.2 (Jetson Linux 39.2) or newer." \
        "Flash JetPack 7.2.1: https://developer.nvidia.com/embedded/jetpack"
      JP_MAJOR=7 JETPACK="JetPack 7" TRT_GB=$TRT_GB_JP7 ;;
    36)
      [ "$L4T_MINOR" -ge 4 ] || die "This Jetson runs JetPack 6 with Jetson Linux $L4T, which is too old." \
        "Update to JetPack 7.2.1 (recommended) or 6.2: https://developer.nvidia.com/embedded/jetpack"
      JP_MAJOR=6 JETPACK="JetPack 6" TRT_GB=$TRT_GB_JP6 ;;
    38)
      die "JetPack 7.0 and 7.1 are not supported. Update to JetPack 7.2.1:" \
        "https://developer.nvidia.com/embedded/jetpack" ;;
    *)
      die "This Jetson's software (Jetson Linux $L4T) is not supported." \
        "Flash JetPack 7.2.1 (recommended) or 6.2: https://developer.nvidia.com/embedded/jetpack" ;;
  esac
  [ "$ARCH" = aarch64 ] || die "Unexpected: a Jetson that is not aarch64 ($ARCH)."
  # TensorRT 10 from JetPack's own package source, on both JetPacks
  FLAVOR=linux-aarch64
  PLATFORM_NAME="$MODEL, $JETPACK (Jetson Linux $L4T)"

  if grep -qw deep "$MEM_SLEEP" 2>/dev/null; then DEEP_SLEEP=1; fi
  detect_power_modes
}

detect_power_modes() {
  command -v nvpmodel >/dev/null 2>&1 || return 0
  local conf=/etc/nvpmodel.conf best_rank=-1 id name rank watts
  [ -r "$conf" ] || return 0
  while read -r id name; do
    case "$name" in
      MAXN_SUPER) rank=1000 ;;
      MAXN) rank=900 ;;
      *W*)
        watts="${name%%W*}"
        watts="${watts//[!0-9]/}"
        rank="${watts:-0}" ;;
      *) rank=0 ;;
    esac
    if [ "$rank" -gt "$best_rank" ]; then
      best_rank=$rank PM_BEST_ID=$id PM_BEST_NAME=$name
    fi
  done < <(sed -n 's/.*POWER_MODEL ID=\([0-9]*\) NAME=\([^ >]*\).*/\1 \2/p' "$conf")
  PM_CURRENT="$(power_mode_now)"
  return 0
}

power_mode_now() {
  nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode: *//p' | head -n 1
}

detect_pc() {
  [ "$ARCH" = x86_64 ] || die "On an Arm computer, Jetlink supports NVIDIA Jetson only." \
    "This one is $ARCH and does not look like a Jetson."
  FLAVOR=linux-x86_64 TRT_GB=$TRT_GB_PC TRT_ARGS=(--tensorrt-libs "$PC_TRT_DIR")
  local q smi=nvidia-smi
  # WSL keeps the Windows driver's tools here, not always on the PATH
  if ! command -v nvidia-smi >/dev/null 2>&1 && [ -x /usr/lib/wsl/lib/nvidia-smi ]; then smi=/usr/lib/wsl/lib/nvidia-smi; fi
  if command -v "$smi" >/dev/null 2>&1 \
      && q="$("$smi" --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null | head -n 1)" \
      && [ -n "$q" ]; then
    GPU_PRESENT=1
    GPU_NAME="$(printf '%s' "$q" | cut -d, -f1 | sed 's/^ *//; s/ *$//')"
    DRIVER="$(printf '%s' "$q" | cut -d, -f2 | tr -d ' ')"
    DRIVER_MAJOR="${DRIVER%%.*}"
    GPU_CC="$(printf '%s' "$q" | cut -d, -f3 | tr -d ' .')"
  else
    local dev
    for dev in /sys/bus/pci/devices/*; do
      if [ "$(cat "$dev/vendor" 2>/dev/null)" = 0x10de ] && grep -q '^0x03' "$dev/class" 2>/dev/null; then
        GPU_PRESENT=1
      fi
    done
    GPU_NAME="NVIDIA GPU"
  fi
  [ "$GPU_PRESENT" = 1 ] || die "No NVIDIA GPU found." \
    "Jetlink needs an NVIDIA GPU (GeForce RTX 20 series or newer) or a Jetson."
  [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] || DRIVER_MAJOR=0
  [[ "$GPU_CC" =~ ^[0-9]+$ ]] || GPU_CC=0
  if [ "$GPU_CC" -gt 0 ] && [ "$GPU_CC" -lt "$MIN_CC" ]; then
    die "The $GPU_NAME is too old for TensorRT." "Jetlink needs a GeForce RTX 20 series (Turing) or newer GPU."
  fi
  PLATFORM_NAME="$GPU_NAME, $OS_NAME"
  [ "$WSL" = 1 ] && PLATFORM_NAME="$PLATFORM_NAME (WSL)"
  return 0
}

# ---------------------------------------------------------------------------
# Answers, saved in install.conf (the questions) and server.env (what the
# server runs with), so an update asks nothing

POWER='' SLEEP_AFTER=0 POWEROFF_WITH_COMMA=0 ADD_SWAP=0 AUTOSTART=1 STATUS_PORT=5600
CACHE_DIR='' REF='' SOURCE='' SOURCE_DIR='' COMMIT=''
# REF is what the install follows: latest (the newest release), a tag or a
# branch. RESOLVED is the tag or branch that gave, or `local` for a checkout,
# saved as JETLINK_VERSION (not VERSION, which /etc/os-release sets).
RESOLVED=''
SWAP_FILE='' MASKED_UNITS='' JOURNALD_CAPPED=0 NEED_REBOOT=0
HAD_INSTALL=0
# 1 when the install to update runs the server in Docker (0.6.0 and older)
DOCKER_ERA=0

load_previous() {
  # a Docker-era unit, even from before the installer wrote install.conf
  if grep -qs docker "$UNIT_DIR/$UNIT.service" || grep -qs '^JETLINK_IMAGE=' "$ENV_FILE"; then
    DOCKER_ERA=1
  fi
  [ -r "$CONF" ] || return 0
  HAD_INSTALL=1
  # only from the files: the jetlink command runs this with them exported
  local JETLINK_REF='' JETLINK_VERSION='' JETLINK_STATUS_PORT='' JETLINK_SLEEP_AFTER_HELD=''
  # shellcheck disable=SC1090
  . "$CONF"
  # what the server runs with, the sleep delay and the cache among it
  # shellcheck disable=SC1090
  [ -r "$ENV_FILE" ] && . "$ENV_FILE"
  POWER="${JETLINK_POWER:-}"
  # a run stopped while hold_sleep held the server awake left the answer here
  SLEEP_AFTER="${JETLINK_SLEEP_AFTER_HELD:-${JETLINK_SLEEP_AFTER:-0}}"
  POWEROFF_WITH_COMMA="${JETLINK_POWEROFF_WITH_COMMA:-0}"
  AUTOSTART="${JETLINK_AUTOSTART:-1}"
  CACHE_DIR="${JETLINK_CACHE_DIR:-}"
  STATUS_PORT="${JETLINK_STATUS_PORT:-$STATUS_PORT}"
  SWAP_FILE="${JETLINK_SWAP_FILE:-}"
  MASKED_UNITS="${JETLINK_MASKED_UNITS:-}"
  JOURNALD_CAPPED="${JETLINK_JOURNALD_CAPPED:-0}"
  REF="${JETLINK_REF:-}"
  RESOLVED="${JETLINK_VERSION:-}"
  return 0
}

# --ref, else the saved choice, else latest. Installers before 0.5.0 saved
# main, their default, without asking, and no JETLINK_VERSION: such an
# install follows releases now.
choose_ref() {
  # a checkout installs what is in it, so a ref there would do nothing
  if [ "$SOURCE" = local ] && [ -n "$OPT_REF" ]; then
    die "--ref does nothing from a checkout, which installs what is in it." \
      "For $OPT_REF, run the installer of that ref instead:" \
      "  curl -fsSL $RAW_URL/$OPT_REF/install.sh | bash -s -- --update --ref $OPT_REF"
  fi
  if [ -n "$OPT_REF" ]; then
    REF="$OPT_REF"
  elif [ "$REF" = main ] && [ -z "$RESOLVED" ]; then
    REF=latest RESOLVED=main
    if [ "$SOURCE" != local ]; then
      note "Jetlink now follows releases; for development builds, use --ref main."
    fi
  fi
  REF="${REF:-latest}"
}

# Without a terminal every question takes its default, which is the
# recommended answer, or the saved one on a reinstall.
ask_questions() {
  if [ "$INTERACTIVE" = 1 ]; then
    heading "A few questions"
    say "  Press Enter to take the recommended answer."
  fi

  if [ "$JETSON" = 1 ]; then
    # always on is the recommended wiring, for a Jetson that can deep-sleep
    local prev="$POWER" def=1 choice always
    if [ "$prev" = switched ] || { [ -z "$prev" ] && [ "$DEEP_SLEEP" = 0 ]; }; then def=2; fi
    always="Always on ${D}(recommended)${N}: sleeps when the car is off to save battery, wakes when you start the car"
    [ "$DEEP_SLEEP" = 1 ] || always="Always on: stays awake when the car is off ${D}(this Jetson cannot sleep)${N}"
    ask_choice choice "$def" "How is the Jetson powered in the car?" \
      "$always" \
      "Switched: turns on and off with the car"
    if [ "$choice" = 1 ]; then
      set_always_on
      local off offdef=y
      [ "$prev" = always ] && [ "$POWEROFF_WITH_COMMA" = 0 ] && offdef=n
      ask_yn off "$offdef" "Allow the comma to shut down the Jetson to protect the car battery?" \
        "The comma does this when it shuts itself down for low battery. The Jetson then" \
        "stays off until its power is reconnected."
      if [ "$off" = y ]; then POWEROFF_WITH_COMMA=1; else POWEROFF_WITH_COMMA=0; fi
    else
      POWER=switched SLEEP_AFTER=0 POWEROFF_WITH_COMMA=0
    fi

    AUTOSTART=1
  else
    local auto autodef=y
    [ "$AUTOSTART" = 0 ] && autodef=n
    ask_yn auto "$autodef" "Start Jetlink automatically when this computer starts?" \
      "If you say no, start it yourself with: jetlink start"
    if [ "$auto" = y ]; then AUTOSTART=1; else AUTOSTART=0; fi
  fi

  ask_port STATUS_PORT "$STATUS_PORT" "Which port should the status page use?" \
    "A read-only page of what Jetlink is doing, for a phone on the same network" \
    "(the comma's hotspot in the car). 0 turns it off."
}

set_always_on() {
  POWER=always SLEEP_AFTER=0
  if [ "$DEEP_SLEEP" = 1 ]; then
    SLEEP_AFTER=120
  else
    note "This Jetson's software cannot deep-sleep, so it stays awake when the car is off (about 7 W)."
  fi
}

# Not questions: the large models need both, so every Jetson install gets
# them, updates included. A move from Docker keeps the Jetson as it was set up.
jetson_musts() {
  [ "$JETSON" = 1 ] || return 0
  if [ -n "$PM_BEST_NAME" ]; then
    if [ "$PM_BEST_NAME" != MAXN_SUPER ] && [[ "$MODEL" == *"Orin Nano"* ]]; then
      note "This JetPack install does not offer the Orin Nano's Super modes; JetPack 7.2.1's"
      note "installer sets them up. Jetlink still works, a little slower, in $PM_BEST_NAME."
    fi
  fi
  ADD_SWAP=0
  if [ -n "$SWAP_FILE" ]; then
    ADD_SWAP=1
  elif [ "$DOCKER_ERA" = 0 ] && swap_short; then
    if [ "$DISK_GB" -ge $((MIN_DISK_GB + SWAP_GB + 5)) ]; then
      ADD_SWAP=1
    else
      note "Not enough disk space for ${SWAP_GB} GB of swap, so the 1.7 GB models may fail to prepare."
    fi
  fi
  return 0
}

# less than the swap the 1.7 GB models need; JetPack's zram is compressed
# memory, not swap, and does not count
swap_short() {
  local kb
  kb=$(awk 'NR > 1 && $1 !~ /zram/ {s += $3} END {printf "%d", s}' "$SWAPS" 2>/dev/null || echo 0)
  [ "${kb:-0}" -lt $(((SWAP_GB - 1) * 1048576)) ]
}

# ---------------------------------------------------------------------------
# The plan

show_found() {
  heading "This computer"
  if [ "$JETSON" = 1 ]; then
    good "$MODEL"
    good "$JETPACK (Jetson Linux $L4T)"
  else
    good "$GPU_NAME"
    if [ "$DRIVER_MAJOR" -ge "$MIN_DRIVER" ]; then
      good "NVIDIA driver $DRIVER"
    elif [ -n "$DRIVER" ]; then
      bad "NVIDIA driver $DRIVER ${D}(needs $MIN_DRIVER or newer)${N}"
    else
      bad "No NVIDIA driver installed"
    fi
    good "$OS_NAME"
    [ "$WSL" = 1 ] && note "Windows (WSL) support is untested."
  fi
  if [ "$TRT_PRESENT" = 1 ]; then good "TensorRT ${TRT_VERSION%%-*}"; fi
  if [ "$DISK_GB" -ge "$MIN_DISK_GB" ]; then
    good "$DISK_GB GB of free disk space"
  else
    bad "$DISK_GB GB of free disk space ${D}(needs $MIN_DISK_GB GB)${N}"
  fi
  return 0
}

preflight() {
  if [ "$HAD_INSTALL" = 0 ] && [ "$DOCKER_ERA" = 0 ] && [ "$DISK_GB" -lt "$MIN_DISK_GB" ]; then
    die "Not enough free disk space: $DISK_GB GB, and Jetlink needs $MIN_DISK_GB GB." \
      "Free some space (or use a bigger drive) and run the installer again."
  fi
  if [ "$WSL" = 1 ] && [ ! -d "$SYSTEMD_RUN" ]; then
    die "Jetlink runs as a systemd service, and this WSL runs without systemd." \
      "Add these two lines to /etc/wsl.conf, run 'wsl --shutdown' in Windows, and try again:" \
      "  [boot]" "  systemd=true"
  fi
  if [ "$OPT_DRY_RUN" != 1 ] && [ -z "$OPT_BINARY" ] \
      && ! curl -fsS --max-time 15 -o /dev/null https://github.com 2>/dev/null; then
    die "No internet connection." "The installer downloads TensorRT and the Jetlink server; connect and try again."
  fi
  if [ "$JETSON" = 0 ] && [ "$DRIVER_MAJOR" -lt "$MIN_DRIVER" ]; then
    offer_driver
  fi
  return 0
}

offer_driver() {
  local what="needs an NVIDIA driver"
  [ -n "$DRIVER" ] && what="has NVIDIA driver $DRIVER and needs"
  if [ "$WSL" = 1 ]; then
    # WSL uses the Windows driver; one installed inside would break it
    die "This computer $what $MIN_DRIVER or newer." \
      "Update the NVIDIA driver in Windows, then run the installer again."
  fi
  if [ "$OS_ID" != ubuntu ]; then
    die "This computer $what $MIN_DRIVER or newer." \
      "Install it from your distribution or https://www.nvidia.com/drivers, restart," \
      "and run the installer again."
  fi
  local go
  ask_yn go y "This computer $what $MIN_DRIVER or newer. Install it now?" \
    "A restart is needed afterwards. With Secure Boot on, you will be asked to" \
    "choose a password now and confirm it on a blue screen when the computer restarts."
  [ "$go" = y ] || die "Jetlink cannot run without NVIDIA driver $MIN_DRIVER or newer." \
    "Install it, restart, and run the installer again."
  [ "$OPT_DRY_RUN" = 1 ] && { step "Install NVIDIA driver $MIN_DRIVER" true; return 0; }
  get_root
  step "Getting the driver list" apt_get update
  step "Installing Ubuntu's driver tool" apt_get install ubuntu-drivers-common
  # open kernel modules: what NVIDIA recommends for Turing and newer, and all
  # CUDA 13 supports
  # from the terminal when there is one: with Secure Boot on, the driver asks
  # for the password it will want confirmed at the next boot
  local input=/dev/null
  if [ -r /dev/tty ] && (exec 3</dev/tty) 2>/dev/null; then input=/dev/tty; fi
  if ! as_root ubuntu-drivers install "nvidia:${MIN_DRIVER}-open" <"$input" >>"$LOG" 2>&1; then
    die "Installing the NVIDIA driver failed." \
      "Install driver $MIN_DRIVER or newer yourself (Software & Updates > Additional Drivers)," \
      "restart, and run the installer again."
  fi
  heading "The NVIDIA driver is installed."
  say "  Restart the computer, then run the installer again:"
  if [ "$REF" = latest ]; then
    say "    curl -fsSL $RAW_URL/main/install.sh | bash"
  else
    say "    curl -fsSL $RAW_URL/$REF/install.sh | bash -s -- --ref $REF"
  fi
  say ""
  save_log
  exit 0
}

show_plan() {
  heading "Here is the plan"
  if [ "$GOING_BACK" = 1 ]; then
    say "  • Go back to $RESOLVED, which runs Jetlink in Docker: its own installer takes over"
    return 0
  fi
  if [ "$DOCKER_ERA" = 1 ]; then
    say "  • Move Jetlink out of Docker ${D}(your settings, models and engines stay)${N}"
  fi
  if [ "$TRT_PRESENT" = 0 ]; then
    if [ "$JP_MAJOR" = 7 ]; then
      say "  • Install NVIDIA TensorRT from JetPack's package source ${D}(about 2.3 GB)${N}"
    elif [ "$JETSON" = 1 ]; then
      say "  • Install NVIDIA TensorRT from JetPack's package source"
    else
      say "  • Download NVIDIA TensorRT $PC_TRT into $PC_TRT_DIR ${D}(3.8 GB, 2.7 GB once unpacked)${N}"
    fi
  fi
  if [ -n "$OPT_BINARY" ]; then
    say "  • Install the Jetlink server from $OPT_BINARY"
  elif [ "$REUSE_SERVER" = 1 ]; then
    say "  • Keep the Jetlink server that is installed"
  else
    say "  • Download the Jetlink server $RESOLVED"
  fi
  if [ "$AUTOSTART" = 1 ]; then
    say "  • Start Jetlink every time this computer starts"
  else
    say "  • Start Jetlink now ${D}(not at every boot)${N}"
  fi
  if [ "$JETSON" = 1 ]; then
    if [ "$SLEEP_AFTER" != 0 ]; then
      say "  • Sleep when the car is off to save battery, and wake when you start the car"
    fi
    if [ "$POWEROFF_WITH_COMMA" = 1 ]; then
      say "  • Let the comma shut down the Jetson to protect the car battery"
    fi
    if [ "$DOCKER_ERA" = 0 ] && [ -n "$PM_BEST_ID" ] && [ "$PM_CURRENT" != "$PM_BEST_NAME" ]; then
      say "  • Switch to the fastest power mode, $PM_BEST_NAME, which the large models need ${D}(may need a restart)${N}"
    fi
    if [ "$ADD_SWAP" = 1 ] && [ -z "$SWAP_FILE" ]; then
      say "  • Add ${SWAP_GB} GB of swap, which the largest models need while they are prepared"
    fi
    say "  • Start up without waiting for a network, and keep the system log small"
  fi
  if [ "$STATUS_PORT" != 0 ]; then
    say "  • Show a read-only status page on port $STATUS_PORT"
  fi
  if [ "$DOCKER_ERA" = 1 ]; then
    say "  • Delete Jetlink's Docker images once the new server runs ${D}(Docker itself stays)${N}"
  fi
  say "  ${D}Models and prepared engines go in $CACHE_DIR${N}"
}

# ---------------------------------------------------------------------------
# Doing it

detect_source() {
  local here=''
  if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  fi
  if [ -n "$here" ] && [ "$here" != "$SRC_ROOT/src" ] \
      && [ -f "$here/scripts/jetlink" ] && [ -f "$here/scripts/jetlink-server.service" ]; then
    # run from a checkout: install exactly what is in it
    SOURCE=local SOURCE_DIR="$here"
  else
    SOURCE=git SOURCE_DIR="$SRC_ROOT/src"
  fi
}

# --binary: install a build made elsewhere, with the scripts from the source
# that is already here (a checkout, or the clone an earlier install made)
# rather than moving that source to a release.
KEEP_SOURCE=0
check_binary() {
  [ -n "$OPT_BINARY" ] || return 0
  [ -f "$OPT_BINARY" ] || die "No such file: $OPT_BINARY"
  OPT_BINARY="$(cd "$(dirname "$OPT_BINARY")" && pwd)/$(basename "$OPT_BINARY")"
  case "$(basename "$OPT_BINARY")" in
    *-linux-aarch64.tar.gz|*-linux-x86_64.tar.gz)
      [[ "$(basename "$OPT_BINARY")" == *"-$FLAVOR.tar.gz" ]] \
        || die "$(basename "$OPT_BINARY") is for another kind of computer; this one needs a $FLAVOR build." ;;
  esac
  if [ "$SOURCE" = local ] || [ -f "$SOURCE_DIR/install.sh" ]; then KEEP_SOURCE=1; fi
  # the jetlink command comes from that source too, and a release's from
  # before the native server drives the Docker one
  if [ "$KEEP_SOURCE" = 1 ] && [ -f "$SOURCE_DIR/scripts/jetlink-run-server" ]; then
    die "$SOURCE_DIR holds Jetlink from before the native server, and --binary installs its scripts." \
      "Move it to the tree the server was built from first, for example:" \
      "  sudo git -C $SOURCE_DIR fetch --depth 1 origin <branch> && sudo git -C $SOURCE_DIR reset --hard FETCH_HEAD" \
      "  sudo $SOURCE_DIR/install.sh --update --binary $OPT_BINARY"
  fi
  return 0
}

# The tag or branch to check out. latest is looked up on every run, so an
# update moves to the newest release; with no answer an update stays on the
# one it has, and a first install stops rather than guess.
resolve_ref() {
  local had="$RESOLVED"
  if [ "$SOURCE" = local ]; then
    RESOLVED=local
  elif [ "$KEEP_SOURCE" = 1 ]; then
    RESOLVED="${had:-local}"
  elif [ "$REF" != latest ]; then
    RESOLVED="$REF"
  else
    RESOLVED="$(latest_release)"
    if [ -n "$RESOLVED" ]; then
      printf '\n==> the newest release is %s\n' "$RESOLVED" >>"$LOG"
    elif [ -n "$had" ] && [ "$had" != local ]; then
      RESOLVED="$had"
      note "Could not look up the newest release; staying on $had."
    else
      die "Could not find the newest Jetlink release." \
        "GitHub did not answer. Check the connection and run the installer again," \
        "or name a release: --ref v0.5.0"
    fi
  fi
  stay_native "$had"
}

# An update or a setup does not take a native install back to Docker by
# itself, while the release it follows (the newest, until one runs natively)
# is one that ran in Docker: only --ref goes there. The server here stays, and
# so does its source.
stay_native() {
  local had=$1
  if [ -n "$OPT_REF" ] || [ "$SOURCE" = local ] || [ "$KEEP_SOURCE" = 1 ] || [ "$DOCKER_ERA" = 1 ] \
      || [ ! -L "$SRC_ROOT/current" ] || ! docker_release "$RESOLVED"; then
    return 0
  fi
  note "$RESOLVED runs Jetlink in Docker, so the server installed here stays. To go back to it:"
  note "  curl -fsSL $RAW_URL/$RESOLVED/install.sh | bash -s -- --update --ref $RESOLVED"
  RESOLVED="$had" KEEP_SOURCE=1 REUSE_SERVER=1
}

# a release from before the native server
docker_release() {
  [[ $1 =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] && version_ge "${DOCKER_LAST#v}" "${1#v}"
}

# The newest release's tag, or nothing: GitHub's latest release, never a draft
# or a prerelease, else the highest vX.Y.Z tag (the API allows 60 requests an
# hour from one address). Nothing in here fails, since the ERR trap would fire
# inside the command substitution that calls it.
latest_release() {
  local json refs re='"tag_name"[[:space:]]*:[[:space:]]*"(v[0-9]+\.[0-9]+\.[0-9]+)"'
  json="$(curl -fsSL --max-time 20 "$API_URL/releases/latest" 2>>"$LOG" || true)"
  if [[ $json =~ $re ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  refs="$(git ls-remote --tags --refs "$REPO_URL" 'v*' 2>>"$LOG" || true)"
  printf '%s\n' "$refs" \
    | sed -n 's#.*refs/tags/\(v[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)$#\1#p' \
    | sort -t. -k1.2,1n -k2,2n -k3,3n | tail -n 1
}

# Where the server for RESOLVED is: a release's own assets, or for main the
# edge prerelease, which CI refreshes on every push under fixed names.
# GOING_BACK: a release from before the native server, which only its own
# installer can put back (the switch itself is decided on its source).
ASSET_URL='' GOING_BACK=0 REUSE_SERVER=0
choose_server() {
  if [ -n "$OPT_BINARY" ] || [ "$REUSE_SERVER" = 1 ]; then return 0; fi
  if [ "$SOURCE" = local ]; then
    if [ "$DOCKER_ERA" = 0 ] && [ -L "$SRC_ROOT/current" ]; then
      REUSE_SERVER=1
      return 0
    fi
    die "From a checkout, the installer installs a server you built:" \
      "  scripts/build-linux.sh $FLAVOR" \
      "  ./install.sh --binary dist/jetlink-server-<version>-$FLAVOR.tar.gz"
  fi
  case "$RESOLVED" in
    main) ASSET_URL="$RELEASES_URL/edge/jetlink-server-edge-$FLAVOR.tar.gz" ;;
    v[0-9]*)
      ASSET_URL="$RELEASES_URL/$RESOLVED/jetlink-server-${RESOLVED#v}-$FLAVOR.tar.gz"
      if docker_release "$RESOLVED"; then GOING_BACK=1; fi ;;
    *) no_server ;;
  esac
}

no_server() {
  die "There is no ready-made Jetlink server for $RESOLVED." \
    "Build one with scripts/build-linux.sh $FLAVOR and install it with --binary."
}

prepare_source() {
  if [ "$SOURCE" = local ] || [ "$KEEP_SOURCE" = 1 ]; then
    COMMIT="$(git -C "$SOURCE_DIR" rev-parse --short HEAD 2>/dev/null || echo local)"
    return 0
  fi
  if [ -d "$SOURCE_DIR/.git" ]; then
    step "Getting Jetlink ($RESOLVED)" as_root sh -c "git -C '$SOURCE_DIR' fetch --depth 1 origin '$RESOLVED' && git -C '$SOURCE_DIR' reset --hard FETCH_HEAD"
  else
    step "Getting Jetlink ($RESOLVED)" as_root sh -c "rm -rf '$SOURCE_DIR' && mkdir -p '$SRC_ROOT' && git clone --depth 1 --branch '$RESOLVED' '$REPO_URL' '$SOURCE_DIR'"
  fi
  COMMIT="$(as_root git -C "$SOURCE_DIR" rev-parse --short HEAD)"
  hand_over
}

# A release that ran the server in Docker (the launcher is the sign) is
# installed by its own installer: `jetlink update --ref v0.6.0` goes back to
# it. That one reads the same answers and puts its Docker server over this
# one; the native files stay, unused, in /opt/jetlink.
hand_over() {
  [ -f "$SOURCE_DIR/scripts/jetlink-run-server" ] || return 0
  [ -z "$OPT_BINARY" ] || die "$RESOLVED runs Jetlink in Docker, so --binary does not apply to it."
  note "$RESOLVED runs Jetlink in Docker; its own installer takes over from here."
  note "To come back later: curl -fsSL $RAW_URL/main/install.sh | bash -s -- --update --ref latest"
  save_log >/dev/null
  exec bash "$SOURCE_DIR/install.sh" "${ARGS[@]}"
}

install_base_packages() {
  local missing=() p wanted=(curl git ca-certificates libcurl4)
  # a PC unpacks TensorRT's wheel
  [ "$JETSON" = 1 ] || wanted+=(unzip)
  for p in "${wanted[@]}"; do
    case "$p" in
      ca-certificates) [ -d /etc/ssl/certs ] || missing+=("$p") ;;
      # the server's one library beyond the C and C++ runtimes
      libcurl4) has_lib libcurl.so.4 || missing+=("$p") ;;
      *) command -v "$p" >/dev/null 2>&1 || missing+=("$p") ;;
    esac
  done
  [ ${#missing[@]} -eq 0 ] && return 0
  apt_update
  step "Installing ${missing[*]}" apt_get install --no-install-recommends "${missing[@]}"
}

# The running server keeps serving while the slow parts download and install,
# and stops only for the switch, so a failure before then leaves it as it was
# (hold_sleep has the one exception). Its settings are kept as server.env.prev
# and install.conf.prev: a failed update puts them back and starts it again
# (restore_previous_server), and after a good one they are a way back by hand.
SERVER_STOPPED=0 CHANGED=0 IMAGES_REMOVED=0 SLEEP_HELD=0
ENV_PREV="$ETC_DIR/server.env.prev"
CONF_PREV="$ETC_DIR/install.conf.prev"

backup_install() {
  local held
  held="$(sed -n 's/^JETLINK_SLEEP_AFTER_HELD=//p' "$ENV_FILE" 2>/dev/null | tail -n 1 || true)"
  if [ -n "$held" ]; then
    # a run stopped while it held the server awake: its sleep is the setting
    { grep -Ev '^JETLINK_SLEEP_AFTER(_HELD)?=' "$ENV_FILE"; echo "JETLINK_SLEEP_AFTER=$held"; } | root_write "$ENV_PREV"
  elif [ -f "$ENV_FILE" ]; then
    as_root cp -p "$ENV_FILE" "$ENV_PREV"
  fi
  if [ -f "$CONF" ]; then as_root cp -p "$CONF" "$CONF_PREV"; fi
  [ "$DOCKER_ERA" = 1 ] || return 0
  # everything the move replaces, and which of the units were enabled
  as_root rm -rf "$DOCKER_ERA_DIR"
  as_root install -d -m 755 "$DOCKER_ERA_DIR/systemd"
  local f u enabled=''
  for u in $DOCKER_ERA_UNITS; do
    f="$UNIT_DIR/$u"
    [ -e "$f" ] || continue
    as_root cp -a "$f" "$DOCKER_ERA_DIR/systemd/"
    case "$u" in
      *.service|*.path)
        if [ "$(as_root systemctl is-enabled "$u" 2>/dev/null || true)" = enabled ]; then
          enabled="$enabled$u"$'\n'
        fi ;;
    esac
  done
  printf '%s' "$enabled" | root_write "$DOCKER_ERA_DIR/enabled"
  if [ -d "$LIB_DIR" ]; then as_root cp -a "$LIB_DIR" "$DOCKER_ERA_DIR/lib"; fi
  for f in "$BIN" "$CONF"; do
    if [ -f "$f" ]; then as_root cp -p "$f" "$DOCKER_ERA_DIR/"; fi
  done
  # the settings as they were, not as a held sleep left them
  if [ -f "$ENV_FILE" ]; then as_root cp -p "$ENV_PREV" "$DOCKER_ERA_DIR/$(basename "$ENV_FILE")"; fi
  good "The Docker setup is saved in $DOCKER_ERA_DIR"
}

# A server that sleeps would suspend the computer under a long download the
# moment the comma lets go. A native one does not while its awake lock is held
# (jetlink caffeinate), so this run holds it until it exits; one from before
# the lock stops now instead of at the switch. The Docker era's does not know
# the lock: it restarts without sleeping, and a failure or an interruption puts
# its server.env back. The sleep it had stays in server.env beside the 0, for
# the next run, should this one be killed before it can put it back.
hold_sleep() {
  local after
  [ -f "$UNIT_DIR/$UNIT.service" ] && [ -f "$ENV_FILE" ] || return 0
  # from the copy backup_install made, which a held sleep does not change
  after="$(sed -n 's/^JETLINK_SLEEP_AFTER=//p' "$ENV_PREV" 2>/dev/null | tail -n 1 || true)"
  [ "${after%.*}" -gt 0 ] 2>/dev/null || return 0
  as_root systemctl is-active --quiet "$UNIT" || return 0
  if [ "$DOCKER_ERA" = 0 ]; then
    if [ -r "$AWAKE_LOCK" ] && exec 8<"$AWAKE_LOCK" && flock --shared --wait 10 8; then
      good "Holding this computer awake for the update"
    else
      stop_running_server
    fi
    return 0
  fi
  SLEEP_HELD=1
  { grep -v '^JETLINK_SLEEP_AFTER=' "$ENV_PREV"; echo 'JETLINK_SLEEP_AFTER=0'; echo "JETLINK_SLEEP_AFTER_HELD=$after"; } \
    | root_write "$ENV_FILE"
  step "Keeping this computer awake for the update" as_root systemctl restart "$UNIT"
}

stop_running_server() {
  [ "$SERVER_STOPPED" = 0 ] || return 0
  [ -f "$UNIT_DIR/$UNIT.service" ] || return 0
  if as_root systemctl is-active --quiet "$UNIT"; then
    step "Stopping the running Jetlink server for the update" as_root systemctl stop "$UNIT"
    SERVER_STOPPED=1
  fi
  # the unit removes its container on the way down; one a crash left holds the
  # name, and the comma's USB interface, which the native server cannot share
  if [ "$DOCKER_ERA" = 1 ] && command -v docker >/dev/null 2>&1; then
    as_root docker rm -f jetlink >>"$LOG" 2>&1 || true
    if as_root docker ps -q --filter 'name=^/?jetlink$' 2>>"$LOG" | grep -q .; then
      die "The Docker server did not stop." "Stop it (sudo docker rm -f jetlink) and run the installer again."
    fi
  fi
  return 0
}

# After a failed update: the previous files and settings, and the previous
# server running if it was.
restore_previous_server() {
  if [ -n "$TRT_STAGE" ]; then as_root rm -rf "$TRT_STAGE" >>"$LOG" 2>&1 || true; fi
  [ "$CHANGED" = 1 ] || [ "$SERVER_STOPPED" = 1 ] || [ "$SLEEP_HELD" = 1 ] || return 0
  local changed=$CHANGED was_running=$SERVER_STOPPED held=$SLEEP_HELD
  CHANGED=0 SERVER_STOPPED=0 SLEEP_HELD=0
  if [ "$changed" = 1 ] || [ "$held" = 1 ]; then
    if [ -f "$ENV_PREV" ]; then as_root cp -p "$ENV_PREV" "$ENV_FILE" >>"$LOG" 2>&1 || true; fi
  fi
  if [ "$changed" = 1 ]; then
    if [ -f "$CONF_PREV" ]; then as_root cp -p "$CONF_PREV" "$CONF" >>"$LOG" 2>&1 || true; fi
    if [ "$DOCKER_ERA" = 1 ]; then
      restore_docker_era
    elif [ -n "$OLD_CURRENT" ]; then
      { point_current "$OLD_CURRENT" && install_unit "$OLD_CURRENT"; } >>"$LOG" 2>&1 || true
    fi
  fi
  as_root systemctl daemon-reload >>"$LOG" 2>&1 || true
  if [ "$IMAGES_REMOVED" = 1 ]; then
    as_root systemctl stop "$UNIT" >>"$LOG" 2>&1 || true
    note "The Docker server's images were deleted to make room, so it cannot start again."
    note "Run the installer again, or go back to Docker with: jetlink update --ref $DOCKER_LAST"
  elif [ "$was_running" = 1 ] || [ "$held" = 1 ]; then
    if as_root systemctl restart "$UNIT" >>"$LOG" 2>&1; then
      note "The previous Jetlink server is running again."
    else
      note "The previous Jetlink server did not start again; see: journalctl -u $UNIT"
    fi
  elif [ "$changed" = 1 ]; then
    as_root systemctl stop "$UNIT" >>"$LOG" 2>&1 || true
  fi
}

restore_docker_era() {
  [ -d "$DOCKER_ERA_DIR/systemd" ] || return 0
  {
    as_root rm -rf "$UNIT_DIR/$UNIT.service" "$UNIT_DIR/$UNIT.service.d"
    as_root cp -a "$DOCKER_ERA_DIR/systemd/." "$UNIT_DIR/"
    if [ -d "$DOCKER_ERA_DIR/lib" ]; then
      as_root rm -rf "$LIB_DIR"
      as_root cp -a "$DOCKER_ERA_DIR/lib" "$LIB_DIR"
    fi
    if [ -f "$DOCKER_ERA_DIR/jetlink" ]; then as_root cp -p "$DOCKER_ERA_DIR/jetlink" "$BIN"; fi
    as_root systemctl daemon-reload
    local u
    while read -r u; do
      case "$u" in
        '') ;;
        "$UNIT.service") as_root systemctl enable "$u" ;;
        *) as_root systemctl enable --now "$u" ;;
      esac
    done <"$DOCKER_ERA_DIR/enabled"
  } >>"$LOG" 2>&1 || true
}

# A jetlink-* unit the installer did not make, set up by hand on a bench,
# stays as it is, and is not saved or started again with the Docker era's.
# One that runs may hold the comma's USB interface the server needs.
others_units() {
  local f u
  for f in "$UNIT_DIR"/jetlink-*.service; do
    [ -e "$f" ] || continue
    u="$(basename "$f")"
    case " $DOCKER_ERA_UNITS " in *" $u "*) continue ;; esac
    if as_root systemctl is-active --quiet "$u" || [ "$(as_root systemctl is-enabled "$u" 2>/dev/null || true)" = enabled ]; then
      note "$u is not the installer's, and stays as it is. If it serves the comma too, stop it:"
      note "  sudo systemctl disable --now $u"
    fi
  done
  return 0
}

# ---------------------------------------------------------------------------
# The runtime: TensorRT on the host, where the Docker era had it in the image

# TensorRT's library, ONNX parser and plugins for major $1, as apt names them
trt_packages() {
  printf '%s ' "libnvinfer$1" "libnvonnxparsers$1" "libnvinfer-plugin$1"
}

# which of those, for each major given, apt has installed
apt_trt_packages() {
  local m p
  for m in "$@"; do
    for p in $(trt_packages "$m"); do
      if [ -n "$(pkg_version "$p")" ]; then printf ' %s' "$p"; fi
    done
  done
}

ensure_runtime() {
  if [ "$JETSON" = 0 ]; then
    pc_trt
    good "TensorRT $PC_TRT"
    return 0
  fi
  jetson_trt
  trt_plugins
  # what apt downloaded is as big again as what it installed
  if [ "$APT_UPDATED" = 1 ]; then apt_get clean >>"$LOG" 2>&1 || true; fi
  if ! { has_lib libnvinfer.so.10 && has_lib libnvonnxparser.so.10; }; then
    die "TensorRT 10 is not where the server can load it." \
      "Check that libnvinfer.so.10 appears in: ldconfig -p"
  fi
  TRT_VERSION="$(pkg_version libnvinfer10)"
  printf '\n==> TensorRT %s\n' "${TRT_VERSION:-from outside the package manager}" >>"$LOG"
  if [ "$JP_MAJOR" = 7 ] && [ -n "$TRT_VERSION" ] && ! version_ge "${TRT_VERSION%%-*}" "$JP7_MIN_TRT"; then
    die "This Jetson has TensorRT ${TRT_VERSION%%-*}, and Jetlink needs $JP7_MIN_TRT or newer." \
      "Update JetPack (sudo apt update && sudo apt upgrade) and run the installer again."
  fi
  if [ -n "$TRT_VERSION" ]; then good "TensorRT ${TRT_VERSION%%-*}"; else good "TensorRT 10"; fi
}

jetson_trt() {
  if [ "$TRT_PRESENT" = 1 ]; then
    if [ "$JP_MAJOR" = 7 ]; then newest_jetson_trt; fi
    return 0
  fi
  make_room_for_trt
  apt_update
  # shellcheck disable=SC2046
  step "Installing TensorRT" apt_get install --no-install-recommends $(trt_packages 10)
}

# JetPack 7.2 follows the newest TensorRT 10 in NVIDIA's Jetson repository; a
# plan built by the one before fails to load and is built again, once.
newest_jetson_trt() {
  local have want
  apt_update
  have="$(pkg_version libnvinfer10)"
  want="$(apt-cache policy libnvinfer10 2>/dev/null | sed -n 's/^ *Candidate: *//p' | head -n 1 || true)"
  if [ -z "$have" ] || [ -z "$want" ] || [ "$want" = '(none)' ] || version_ge "$have" "$want"; then
    return 0
  fi
  if [ "$(free_gb /)" -lt "$TRT_GB" ]; then
    note "Not enough room on / to update TensorRT; staying on ${have%%-*}."
    return 0
  fi
  # shellcheck disable=SC2046
  if ! run_step "Updating TensorRT to ${want%%-*}" apt_get install --only-upgrade --no-install-recommends $(trt_packages 10); then
    note "Could not update TensorRT; staying on ${have%%-*}."
  fi
}

# A PC's TensorRT: the wheel's libraries (libnvinfer, its ONNX parser and
# plugins, and the Linux builder resources, not the ones for building Windows
# plans) into $PC_TRT_DIR, which later versions reuse. It downloads beside
# that directory, whose filesystem the space check was for, and appears whole
# or not at all. A TensorRT an earlier installer put in with apt stays, unused.
TRT_STAGE=''
pc_trt() {
  local old wheel
  old="$(apt_trt_packages 11)"
  [ -z "$old" ] || note "TensorRT from apt is no longer used here; to remove it: sudo apt remove$old"
  [ "$TRT_PRESENT" = 1 ] && return 0
  make_room_for_trt
  # owned by the user, who downloads into it; a failure removes it
  TRT_STAGE="$TRT_ROOT/.new" wheel="$TRT_ROOT/.new/${PC_TRT_WHEEL##*/}"
  as_root rm -rf "$TRT_STAGE"
  as_root install -d -m 755 -o "$(id -u)" -g "$(id -g)" "$TRT_STAGE"
  step "Downloading TensorRT $PC_TRT (3.8 GB)" download "$PC_TRT_WHEEL" "$wheel"
  printf '%s  %s\n' "$PC_TRT_SHA256" "$wheel" | sha256sum -c --status \
    || die "The TensorRT download is damaged: its checksum does not match." "Run the installer again."
  step "Unpacking TensorRT" as_root unzip -q -j -o "$wheel" 'tensorrt_libs/lib*.so*' '*/LICENSE.txt' \
    -x '*_win_*' -d "$TRT_STAGE/lib"
  as_root rm -rf "$PC_TRT_DIR"
  as_root mv -T "$TRT_STAGE/lib" "$PC_TRT_DIR"
  as_root rm -rf "$TRT_STAGE"
  TRT_STAGE=''
}

# TensorRT's plugin library, where TensorRT came without it: the servers in
# Docker had it, so a plan with a plugin layer loads here too. The same build
# as libnvinfer, which it depends on exactly. Jetlink's own models use no
# plugin, so doing without is not an error.
trt_plugins() {
  local have
  has_lib libnvinfer_plugin.so.10 && return 0
  have="$(pkg_version libnvinfer10)"
  # a TensorRT from outside the package manager brings its own, or none
  [ -n "$have" ] || return 0
  apt_update
  if ! run_step "Installing TensorRT's plugins" \
      apt_get install --no-install-recommends "libnvinfer-plugin10=$have"; then
    note "Could not install TensorRT's plugins; Jetlink's models do not need them."
  fi
}

# TensorRT goes on / (a PC's under /opt/jetlink). When the Docker era's images
# are what fills it they go first, which stops the old server now rather than
# at the switch; going back to it downloads them again. Images Docker keeps on
# another disk would free nothing there, so they stay.
make_room_for_trt() {
  local where=/ free docker_root
  [ "$JETSON" = 1 ] || where=$TRT_ROOT
  free="$(free_gb "$where")"
  [ "$free" -ge "$TRT_GB" ] && return 0
  if [ "$DOCKER_ERA" = 1 ] && [ -n "$(docker_images)" ]; then
    docker_root="$(as_root docker info -f '{{.DockerRootDir}}' 2>>"$LOG" || true)"
    if [ -n "$docker_root" ] && [ "$(mount_of "$docker_root")" = "$(mount_of "$where")" ]; then
      note "$free GB free for TensorRT, which needs $TRT_GB GB: deleting Jetlink's Docker images first."
      stop_running_server
      remove_docker_images
      free="$(free_gb "$where")"
      [ "$free" -ge "$TRT_GB" ] && return 0
    else
      note "Jetlink's Docker images are not on the disk TensorRT goes on (${docker_root:-Docker did not say where}), so they stay."
    fi
  fi
  die "Not enough free space for TensorRT: $free GB on $where, and it needs $TRT_GB GB." \
    "Free some space and run the installer again."
}

# the mount point of the filesystem that holds $1, or will once it is made
mount_of() {
  local where=$1
  while [ ! -d "$where" ]; do where="$(dirname "$where")"; done
  { df -P "$where" 2>/dev/null || true; } | awk 'NR == 2 {print $NF}'
}

# every image the Docker-era installers pulled or built
docker_images() {
  command -v docker >/dev/null 2>&1 || return 0
  as_root docker image ls --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
    | grep -E '^(jetlink|ghcr\.io/zoompilot/jetlink):' | sort -u || true
}

remove_docker_images() {
  local images
  images="$(docker_images)"
  [ -n "$images" ] || return 0
  # shellcheck disable=SC2086
  as_root docker rmi $images >>"$LOG" 2>&1 || true
  IMAGES_REMOVED=1
  good "Jetlink's Docker images deleted"
}

# ---------------------------------------------------------------------------
# The server: /opt/jetlink/<version>/, and `current` pointing at it

NEW_DIR='' SERVER_VERSION='' OLD_CURRENT=''

get_server() {
  if [ "$REUSE_SERVER" = 1 ]; then
    NEW_DIR="$(readlink -f "$SRC_ROOT/current")"
    SERVER_VERSION="$(tr -d '[:space:]' <"$NEW_DIR/VERSION" 2>/dev/null || basename "$NEW_DIR")"
    return 0
  fi
  local tmp name rc=0
  tmp="$(mktemp -d)"
  if [ -n "$OPT_BINARY" ]; then
    name="$(basename "$OPT_BINARY")"
    cp "$OPT_BINARY" "$tmp/$name"
    if [ -f "$OPT_BINARY.sha256" ]; then cp "$OPT_BINARY.sha256" "$tmp/$name.sha256"; fi
  else
    name="${ASSET_URL##*/}"
    printf '\n==> %s\n' "$ASSET_URL.sha256" >>"$LOG"
    download "$ASSET_URL.sha256" "$tmp/$name.sha256" >>"$LOG" 2>&1 || rc=$?
    if [ "$rc" = 22 ]; then
      no_server
    elif [ "$rc" != 0 ]; then
      die "Could not download the Jetlink server." "Check the connection and run the installer again."
    fi
    step "Downloading the Jetlink server ($RESOLVED)" download "$ASSET_URL" "$tmp/$name"
  fi
  # the .sha256 names the tarball beside it, as build-linux.sh and CI write it
  if [ -f "$tmp/$name.sha256" ] && ! (cd "$tmp" && sha256sum -c --status "$name.sha256"); then
    die "The Jetlink server download is damaged: its checksum does not match." "Run the installer again."
  fi
  unpack_server "$tmp/$name"
  rm -rf "$tmp"
  # a build of a release names it, so an update later knows where it stands
  if [ -n "$OPT_BINARY" ] && [ "$SOURCE" != local ] && [[ $SERVER_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    RESOLVED="v$SERVER_VERSION"
  fi
  good "Jetlink server $SERVER_VERSION"
}

# A dropped connection is tried again; a file that is not there (curl's 22,
# an HTTP error) is not.
download() {
  local url=$1 out=$2 attempt rc=0
  for attempt in 1 2 3; do
    rc=0
    # a minute without progress abandons the attempt
    curl -fL --connect-timeout 20 --speed-limit 1024 --speed-time 60 -o "$out" "$url" || rc=$?
    case "$rc" in
      0) return 0 ;;
      22) return 22 ;;
    esac
    if [ "$attempt" != 3 ]; then
      echo "the download was interrupted; trying again"
      sleep "$POLL_S"
    fi
  done
  return "$rc"
}

# Into a directory named for the version inside. The files sit at the top of
# the tarball or in its one folder. A version already here is replaced whole;
# a server running from it keeps the files it has open.
unpack_server() {
  local tarball=$1 stage="$SRC_ROOT/.new-$$" top ver
  as_root rm -rf "$stage"
  as_root mkdir -p "$stage"
  # owned by root, not by whoever built it: root runs it
  as_root tar -xzf "$tarball" --no-same-owner -C "$stage" >>"$LOG" 2>&1 \
    || die "$(basename "$tarball") could not be unpacked."
  top="$stage"
  if [ ! -e "$stage/bin/jetlink-server" ]; then
    top="$(find "$stage" -mindepth 3 -maxdepth 3 -path '*/bin/jetlink-server' | head -n 1 || true)"
    top="${top%/bin/jetlink-server}"
  fi
  if [ -z "$top" ] || [ ! -x "$top/bin/jetlink-server" ] || [ ! -f "$top/$SHARE/systemd/$UNIT.service" ]; then
    as_root rm -rf "$stage"
    die "$(basename "$tarball") has no bin/jetlink-server and $SHARE/systemd/$UNIT.service in it."
  fi
  # a tree from before the native server ships a unit that runs Docker, which
  # systemd would start and fail with every 2 seconds
  if ! grep -q "^ExecStart=$SRC_ROOT/current/bin/jetlink-server" "$top/$SHARE/systemd/$UNIT.service"; then
    as_root rm -rf "$stage"
    die "The unit in $(basename "$tarball") does not start $SRC_ROOT/current/bin/jetlink-server." \
      "It was built from an older tree. Build the tarball from the current one:" \
      "  scripts/build-linux.sh $FLAVOR"
  fi
  ver="$(tr -d '[:space:]' <"$top/VERSION" 2>/dev/null || true)"
  if [ -z "$ver" ]; then
    ver="$(basename "$tarball" | sed -n 's/^jetlink-server-\(.*\)-linux-[a-z0-9_]*\.tar\.gz$/\1/p')"
  fi
  if ! [[ $ver =~ ^[0-9A-Za-z][0-9A-Za-z._+-]*$ ]] || [ "$ver" = current ] || [ "$ver" = previous ] || [ "$ver" = src ] || [ "$ver" = tensorrt ]; then
    as_root rm -rf "$stage"
    die "Cannot tell which version $(basename "$tarball") is."
  fi
  SERVER_VERSION="$ver" NEW_DIR="$SRC_ROOT/$ver"
  as_root rm -rf "$NEW_DIR"
  as_root mv "$top" "$NEW_DIR"
  as_root rm -rf "$stage"
}

# The installer's GPU check is the server's own: TensorRT has to load and see
# a GPU. It runs before the old server stops, so a server that cannot work
# never replaces one that does.
check_gpu() {
  local out rc=0 trt
  printf '\n==> %s/bin/jetlink-server backends --backend trt %s\n' "$NEW_DIR" "${TRT_ARGS[*]}" >>"$LOG"
  out="$(as_root "$NEW_DIR/bin/jetlink-server" backends --backend trt "${TRT_ARGS[@]}" 2>&1)" || rc=$?
  printf '%s\n' "$out" >>"$LOG"
  # "trt: usable: TensorRT 10.16.2.10 on Orin-sm87", or "trt: not usable: why"
  trt="$(printf '%s\n' "$out" | sed -n 's/^trt: //p' | head -n 1)"
  if [ "$rc" = 0 ]; then
    good "The server can use the GPU: ${trt#usable: }"
    return 0
  fi
  if [ -n "$trt" ]; then
    bad "The Jetlink server cannot use the GPU: ${trt#not usable: }"
  else
    bad "The Jetlink server did not start."
    printf '%s\n' "$out" | tail -n 5 | sed 's/^/    /'
  fi
  local hint="Restart the computer and run the installer again: a new driver needs a restart."
  [ "$JETSON" = 1 ] && hint="Check that JetPack installed completely (sudo apt install nvidia-jetpack), then run the installer again."
  die "TensorRT cannot run on this computer." "$hint"
}

# An atomic switch: a new link renamed over the old one.
point_current() {
  as_root ln -sfn "$1" "$SRC_ROOT/current.new"
  as_root mv -Tf "$SRC_ROOT/current.new" "$SRC_ROOT/current"
}

switch_server() {
  OLD_CURRENT=''
  if [ -L "$SRC_ROOT/current" ]; then OLD_CURRENT="$(readlink -f "$SRC_ROOT/current")"; fi
  CHANGED=1
  point_current "$NEW_DIR"
}

# After a good start: the one before becomes `previous`, the way back by hand,
# and any older one goes.
keep_previous() {
  if [ -n "$OLD_CURRENT" ] && [ "$OLD_CURRENT" != "$NEW_DIR" ] && [ -d "$OLD_CURRENT" ]; then
    as_root ln -sfn "$OLD_CURRENT" "$SRC_ROOT/previous"
  fi
  local prev='' d
  if [ -L "$SRC_ROOT/previous" ]; then prev="$(readlink -f "$SRC_ROOT/previous")"; fi
  for d in "$SRC_ROOT"/*; do
    if [ -L "$d" ] || [ ! -x "$d/bin/jetlink-server" ] || [ "$d" = "$NEW_DIR" ] || [ "$d" = "$prev" ]; then
      continue
    fi
    as_root rm -rf "$d"
    printf '\n==> removed the old server %s\n' "$d" >>"$LOG"
  done
  # a PC's TensorRT from before the one this server uses
  for d in "$TRT_ROOT"/*; do
    if [ -d "$d" ] && [ "$d" != "$PC_TRT_DIR" ]; then as_root rm -rf "$d"; fi
  done
}

# ---------------------------------------------------------------------------
# The rest of the computer

configure_jetson() {
  [ "$JETSON" = 1 ] || return 0
  if [ "$DOCKER_ERA" = 0 ] && [ -n "$PM_BEST_ID" ] && [ "$PM_CURRENT" != "$PM_BEST_NAME" ]; then
    set_power_mode
  fi
  if [ "$ADD_SWAP" = 1 ] && [ -z "$SWAP_FILE" ]; then
    SWAP_FILE="$(dirname "$CACHE_DIR")/jetlink-swapfile"
    step "Adding ${SWAP_GB} GB of swap" add_swap "$SWAP_FILE"
  fi
  local u masked=''
  for u in $WAIT_ONLINE_UNITS; do
    if as_root systemctl list-unit-files "$u" 2>/dev/null | grep -q "^$u"; then
      if [ "$(as_root systemctl is-enabled "$u" 2>/dev/null || true)" != masked ]; then
        as_root systemctl mask "$u" >>"$LOG" 2>&1 || true
      fi
      masked="$masked $u"
    fi
  done
  MASKED_UNITS="${masked# }"
  [ -n "$MASKED_UNITS" ] && good "Starts without waiting for a network"
  if [ "$JOURNALD_CAPPED" != 1 ]; then
    printf '# Jetlink: keep the system log from filling a small root partition\n[Journal]\nSystemMaxUse=200M\n' \
      | root_write "$JOURNALD_DROPIN"
    as_root systemctl restart systemd-journald >>"$LOG" 2>&1 || true
    JOURNALD_CAPPED=1
    good "System log limited to 200 MB"
  fi
}

set_power_mode() {
  printf '\n==> nvpmodel -m %s\n' "$PM_BEST_ID" >>"$LOG"
  # nvpmodel asks whether to reboot when the new mode needs one; say no here
  # and tell the user at the end
  printf 'no\n' | as_root nvpmodel -m "$PM_BEST_ID" >>"$LOG" 2>&1 || true
  local now
  now="$(power_mode_now)"
  if [ "$now" = "$PM_BEST_NAME" ]; then
    good "Power mode set to $PM_BEST_NAME"
    PM_CURRENT="$now"
  else
    NEED_REBOOT=1
    note "Power mode $PM_BEST_NAME takes effect after a restart."
  fi
}

add_swap() {
  local file=$1
  as_root mkdir -p "$(dirname "$file")"
  as_root fallocate -l "${SWAP_GB}G" "$file"
  as_root chmod 600 "$file"
  as_root mkswap "$file"
  as_root swapon "$file"
  grep -q "^$file " /etc/fstab || printf '%s none swap sw 0 0\n' "$file" | as_root tee -a /etc/fstab >/dev/null
}

install_files() {
  as_root install -d -m 755 "$ETC_DIR"
  as_root mkdir -p "$CACHE_DIR"
  as_root install -D -m 755 "$SOURCE_DIR/scripts/jetlink" "$BIN"
  install_unit "$NEW_DIR"
  printf '# Jetlink: the cache has to be mounted before the server starts\n[Unit]\nRequiresMountsFor=%s\n' "$CACHE_DIR" \
    | root_write "$UNIT_DIR/$UNIT.service.d/10-cache.conf"
  if [ "$JETSON" = 1 ]; then
    # jetson_clocks pins the clocks and turns DVFS off, so the GPU sits at the
    # power mode's ceiling instead of ramping between frames. A reboot undoes
    # it, so it runs before every start, and after nvpmodel, whose mode sets
    # the ceiling and would undo it too.
    printf '# Jetlink: the GPU at full clock while the server runs\n[Unit]\nAfter=nvpmodel.service\n[Service]\nExecStartPre=-/usr/bin/jetson_clocks\n' \
      | root_write "$CLOCKS_DROPIN"
  else
    as_root rm -f "$CLOCKS_DROPIN"
  fi
  if [ "$DOCKER_ERA" = 1 ]; then set_aside_docker_dropins; fi

  # the hubs are armed for remote wakeup at boot by the rule, and again by
  # the server before every suspend
  if [ "$SLEEP_AFTER" != 0 ]; then
    as_root install -D -m 644 "$NEW_DIR/$SHARE/udev/99-jetlink-usb-wakeup.rules" "$WAKE_RULE"
    as_root udevadm control --reload-rules >>"$LOG" 2>&1 || true
    as_root udevadm trigger --subsystem-match=usb --action=add >>"$LOG" 2>&1 || true
  else
    as_root rm -f "$WAKE_RULE"
  fi

  remove_docker_era_files
  # the "no" to the poweroff question before the server took a flag for it
  if grep -qs "Written by the Jetlink installer" "$CACHE_DIR/poweroff-dry-run"; then
    as_root rm -f "$CACHE_DIR/poweroff-dry-run"
  fi
  write_env
  write_conf
}

# the unit of the server in directory $1, so the two always match
install_unit() {
  as_root install -D -m 644 "$1/$SHARE/systemd/$UNIT.service" "$UNIT_DIR/$UNIT.service"
}

# A drop-in of the user's for the Docker-era unit that runs docker would stop
# the native server starting. The backup has it.
set_aside_docker_dropins() {
  local f
  for f in "$UNIT_DIR/$UNIT.service.d"/*.conf; do
    [ -f "$f" ] || continue
    case "$f" in */10-cache.conf|"$CLOCKS_DROPIN") continue ;; esac
    grep -qi docker "$f" || continue
    as_root rm -f "$f"
    note "Your drop-in $(basename "$f") runs Docker, so it is set aside in $DOCKER_ERA_DIR/systemd/$UNIT.service.d"
  done
  return 0
}

# What the server does itself now: the launcher, the hub wakeup script, the
# poweroff flag's units, and the status page's own process (never released,
# now in the server).
remove_docker_era_files() {
  local u
  for u in jetlink-poweroff.path jetlink-web.service; do
    if [ -e "$UNIT_DIR/$u" ]; then as_root systemctl disable --now "$u" >>"$LOG" 2>&1 || true; fi
  done
  as_root rm -rf "$UNIT_DIR/jetlink-poweroff.path" "$UNIT_DIR/jetlink-poweroff.service" \
    "$UNIT_DIR/jetlink-web.service" "$UNIT_DIR/jetlink-web.service.d" "$LIB_DIR"
}

write_env() {
  # the unit passes the server --poweroff only on a Jetson whose user said yes
  local poweroff=''
  if [ "$JETSON" = 1 ] && [ "$POWEROFF_WITH_COMMA" = 1 ]; then poweroff=--poweroff; fi
  {
    echo "# Written by the Jetlink installer; run it again (jetlink setup) to change these."
    printf 'JETLINK_CACHE_DIR=%q\n' "$CACHE_DIR"
    printf 'JETLINK_SLEEP_AFTER=%q\n' "$SLEEP_AFTER"
    printf 'JETLINK_STATUS_PORT=%q\n' "$STATUS_PORT"
    # flags the unit passes as they are, or nothing when empty
    printf 'JETLINK_POWEROFF="%s"\n' "$poweroff"
    printf 'JETLINK_TENSORRT="%s"\n' "${TRT_ARGS[*]}"
    printf 'JETLINK_JETSON=%q\n' "$JETSON"
    printf 'JETLINK_FLAVOR=%q\n' "$FLAVOR"
    printf 'JETLINK_SERVER_VERSION=%q\n' "$SERVER_VERSION"
  } | root_write "$ENV_FILE"
}

write_conf() {
  {
    echo "# The answers the Jetlink installer was given; it reads them back on an update."
    printf 'JETLINK_REF=%q\n' "$REF"
    printf 'JETLINK_VERSION=%q\n' "$RESOLVED"
    printf 'JETLINK_SOURCE=%q\n' "$SOURCE"
    printf 'JETLINK_SOURCE_DIR=%q\n' "$SOURCE_DIR"
    printf 'JETLINK_COMMIT=%q\n' "$COMMIT"
    printf 'JETLINK_PLATFORM_NAME=%q\n' "$PLATFORM_NAME"
    printf 'JETLINK_POWER=%q\n' "$POWER"
    printf 'JETLINK_POWEROFF_WITH_COMMA=%q\n' "$POWEROFF_WITH_COMMA"
    printf 'JETLINK_AUTOSTART=%q\n' "$AUTOSTART"
    printf 'JETLINK_SWAP_FILE=%q\n' "$SWAP_FILE"
    printf 'JETLINK_MASKED_UNITS=%q\n' "$MASKED_UNITS"
    printf 'JETLINK_JOURNALD_CAPPED=%q\n' "$JOURNALD_CAPPED"
    printf 'JETLINK_INSTALLED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } | root_write "$CONF"
}

start_server() {
  as_root systemctl daemon-reload
  if [ "$AUTOSTART" = 1 ]; then
    as_root systemctl enable "$UNIT" >>"$LOG" 2>&1
  else
    as_root systemctl disable "$UNIT" >>"$LOG" 2>&1 || true
  fi
  local since
  since="$(date '+%Y-%m-%d %H:%M:%S')"
  as_root systemctl restart "$UNIT"
  step "Starting the Jetlink server" wait_ready "$since"
  if [[ $(server_journal "$since") == *"$PRELOAD_LINE"* ]]; then
    step "Loading the model it ran last" wait_engine "$since"
    if [ -z "$(engine_line "$since")" ]; then
      note "It is still preparing that model, and goes on in the background: jetlink logs"
    fi
  fi
  # only now, with the new server up and staying up: until then the old one,
  # its settings and its images are the way back
  SERVER_STOPPED=0 CHANGED=0 SLEEP_HELD=0
  keep_previous
  if [ "$DOCKER_ERA" = 1 ]; then remove_docker_images; fi
}

# The server says the first once it serves, whatever the comma is doing; the
# others are what builds before that line said, for a --binary of one.
READY_RE='jetlink-server is serving|waiting for a jetlink gadget|client connected|nothing on the comma is serving it yet'
# systemd's line when it starts a server that exited again
RESTART_RE='restart counter is at [0-9]'
# said before the server serves when it loads the model it ran last
PRELOAD_LINE='preloading the engine loaded last'

server_journal() {
  as_root journalctl -u "$UNIT" --since "$1" --no-pager -o cat 2>/dev/null || true
}

# Up means the server said it serves and then stayed up; a restart means it
# crashed. The journal is followed, so either shows the moment it is written.
# The follower ends at its next line or at the timeout.
wait_ready() {
  local since=$1 line
  line="$(grep -m1 -E "$READY_RE|$RESTART_RE" \
    < <(as_root timeout "$READY_S" journalctl -f -u "$UNIT" --since "$since" -o cat 2>/dev/null) || true)"
  case "$line" in
    *"restart counter"*|'')
      server_journal "$since" | tail -n 30
      if [ -n "$line" ]; then echo "the server crashed, and systemd started it again"; else echo "the server did not report ready within 3 minutes"; fi
      return 1 ;;
  esac
  echo "$line"
  # one loading a model is watched until that is done, by wait_engine
  [[ $(server_journal "$since") == *"$PRELOAD_LINE"* ]] && return 0
  stays_up "$since"
}

# The model the server ran last, loaded, or its preparation failed (the
# server serves without it then, and the comma sends it again); a restart
# means loading it crashed the server. After PRELOAD_S the server is left to it.
wait_engine() {
  local since=$1 line deadline=$((SECONDS + PRELOAD_S))
  while :; do
    line="$(engine_line "$since")"
    case "$line" in
      *"restart counter"*)
        server_journal "$since" | tail -n 30
        echo "loading the model crashed the server, and systemd started it again"
        return 1 ;;
      ?*) echo "$line"; break ;;
    esac
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "still preparing the model after $(elapsed "$PRELOAD_S"); the server serves meanwhile"
      break
    fi
    sleep "$POLL_S"
  done
  stays_up "$since"
}

engine_line() {
  server_journal "$1" | grep -m1 -E "engine ready|engine preparation failed|$RESTART_RE" || true
}

# Still running a moment later and never restarted: a server can crash just
# after it said it serves, when the comma is first seen or the model loads.
stays_up() {
  local since=$1 restarts
  sleep "$SETTLE_S"
  restarts="$(as_root systemctl show -p NRestarts --value "$UNIT" 2>/dev/null || true)"
  if as_root systemctl is-active --quiet "$UNIT" && [ "${restarts:-0}" = 0 ]; then
    return 0
  fi
  server_journal "$since" | tail -n 30
  echo "the server did not stay up: systemd started it again ${restarts:-?} times"
  return 1
}

finish() {
  save_log >/dev/null
  heading "${G}Jetlink is installed and running.${N}"
  if [ "$DOCKER_ERA" = 1 ]; then
    say "  It no longer runs in Docker. Docker stays installed for anything else that uses it,"
    say "  and the old setup is kept in $DOCKER_ERA_DIR."
  fi
  say ""
  say "  ${B}Next, on your comma:${N}"
  say "    1. Settings > Software > Target Branch: choose ${B}jetson-trt${N} (zoompilot),"
  say "       and let it update and restart."
  say "    2. Settings > Models: set ${B}Accelerator Link${N} to ${B}USB${N}."
  if [ "$JETSON" = 1 ]; then
    say "    3. Connect the comma's USB-C port to one of this Jetson's ${B}USB-A${N} ports"
  else
    say "    3. Connect the comma's USB-C port to one of this computer's ${B}USB-A${N} ports"
  fi
  say "       with a USB 3 data cable (charge-only cables do not work)."
  say "    4. Stay parked and wait for the comma's icon to turn ${G}green${N}. The first model"
  say "       takes a few minutes to prepare."
  if [ "$WSL" = 1 ]; then
    say ""
    note "In Windows, attach the comma to WSL with usbipd: https://learn.microsoft.com/windows/wsl/connect-usb"
  fi
  if [ "$JETSON" = 0 ]; then
    say ""
    note "Keep this computer plugged in and awake while driving: sleep drops the link."
  fi
  if [ "$STATUS_PORT" != 0 ]; then
    say ""
    say "  ${B}Status page:${N} http://$(hostname 2>/dev/null || uname -n).local:$STATUS_PORT"
    say "    from a phone on the same network, like the comma's hotspot. It only shows"
    say "    what the server is doing; nothing on it changes anything."
  fi
  say ""
  say "  ${B}Handy commands:${N}"
  say "    jetlink status    is it running, and is the comma connected"
  say "    jetlink logs      watch what it is doing"
  say "    jetlink update    get the newest version"
  say "    jetlink setup     change your answers"
  if [ "$NEED_REBOOT" = 1 ]; then
    say ""
    note "${B}Restart this computer once${N} to finish switching the power mode: sudo reboot"
  fi
  say ""
}

# ---------------------------------------------------------------------------
# Removing it

uninstall() {
  load_previous
  heading "Remove Jetlink"
  if [ "$HAD_INSTALL" = 0 ] && [ ! -f "$UNIT_DIR/$UNIT.service" ]; then
    say "  Jetlink is not installed here."
    exit 0
  fi
  local go
  ask_yn go n "Remove Jetlink from this computer?" \
    "TensorRT from apt stays installed, and so does Docker if you have it."
  [ "$go" = y ] || { say "  Nothing changed."; exit 0; }
  [ "$OPT_DRY_RUN" = 1 ] && { say "  (dry run: nothing changed)"; exit 0; }
  get_root
  as_root systemctl disable --now "$UNIT" >>"$LOG" 2>&1 || true
  remove_docker_era_files
  if command -v docker >/dev/null 2>&1; then as_root docker rm -f jetlink >>"$LOG" 2>&1 || true; fi
  as_root rm -rf "$UNIT_DIR/$UNIT.service" "$UNIT_DIR/$UNIT.service.d"
  as_root systemctl daemon-reload
  as_root rm -f "$BIN" "$WAKE_RULE"
  as_root udevadm control --reload-rules >>"$LOG" 2>&1 || true
  good "Server and its settings removed"
  if [ -n "$MASKED_UNITS" ]; then
    # shellcheck disable=SC2086
    as_root systemctl unmask $MASKED_UNITS >>"$LOG" 2>&1 || true
  fi
  if [ -f "$JOURNALD_DROPIN" ]; then
    as_root rm -f "$JOURNALD_DROPIN"
    as_root systemctl restart systemd-journald >>"$LOG" 2>&1 || true
  fi
  if [ -n "$SWAP_FILE" ] && [ -f "$SWAP_FILE" ]; then
    as_root swapoff "$SWAP_FILE" >>"$LOG" 2>&1 || true
    as_root sed -i "\#^$SWAP_FILE #d" /etc/fstab
    as_root rm -f "$SWAP_FILE"
    good "Swap file removed"
  fi
  if [ -n "$(docker_images)" ]; then
    local rm_images
    ask_yn rm_images y "Delete Jetlink's old Docker images to free their disk space (about 4 GB each)?"
    if [ "$rm_images" = y ]; then
      remove_docker_images
    fi
  fi
  if [ -n "$CACHE_DIR" ] && [ -d "$CACHE_DIR" ]; then
    local size rm_cache
    size="$(as_root du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)"
    ask_yn rm_cache n "Also delete the downloaded models in $CACHE_DIR ($size)?" \
      "Keep them if you might install Jetlink again: they take a while to download."
    if [ "$rm_cache" = y ]; then
      as_root rm -rf "$CACHE_DIR"
      good "Models deleted"
    fi
  fi
  as_root rm -rf "$ETC_DIR" "$SRC_ROOT"
  heading "Jetlink is removed."
  local trt
  trt="$(apt_trt_packages 10 11)"
  [ -z "$trt" ] || say "  TensorRT from apt stays installed; to remove it: sudo apt remove$trt"
  say ""
  exit 0
}

# ---------------------------------------------------------------------------

usage() {
  sed -n '2,/^set -Eeuo/p' "${BASH_SOURCE[0]}" 2>/dev/null | sed '$d; s/^# \{0,1\}//'
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --yes|-y) OPT_YES=1 ;;
      --update) OPT_UPDATE=1 ;;
      --reconfigure) OPT_RECONFIGURE=1 ;;
      --ref) OPT_REF="${2:?--ref needs a branch or tag}"; shift ;;
      --ref=*) OPT_REF="${1#*=}" ;;
      --binary) OPT_BINARY="${2:?--binary needs a server tarball}"; shift ;;
      --binary=*) OPT_BINARY="${1#*=}" ;;
      --dry-run) OPT_DRY_RUN=1 ;;
      --uninstall) OPT_UNINSTALL=1 ;;
      --build|--image|--image=*)
        die "Jetlink no longer runs in Docker, so $1 is gone." \
          "To install a server you built: --binary jetlink-server-<version>-<flavor>.tar.gz" ;;
      -h|--help) usage; exit 0 ;;
      *) die "Unknown option: $1" "Run with --help to see the options." ;;
    esac
    shift
  done
}

main() {
  # the script has been read in full by now; nothing below may read stdin
  exec </dev/null
  ARGS=("$@")
  parse_args "$@"
  setup_colors
  # the output as it is here, for on_signal: a step sends its own to the log
  exec 4>&1 5>&2
  trap 'on_error $LINENO' ERR
  trap 'on_signal INT 130' INT
  trap 'on_signal TERM 143' TERM
  trap 'on_signal HUP 129' HUP
  : >"$LOG"

  printf '\n%sJetlink installer%s\n' "$B" "$N"
  say "  Run openpilot's large driving models on this computer, for your comma."
  [ "$OPT_DRY_RUN" = 1 ] && note "Dry run: nothing will be changed."

  open_input
  if [ "$OPT_UNINSTALL" = 1 ]; then
    uninstall
  fi

  detect
  load_previous
  detect_source
  choose_ref
  if [ -z "$CACHE_DIR" ]; then
    CACHE_DIR=/var/lib/jetlink
    [ "$JETSON" = 1 ] && CACHE_DIR=/mnt/data/jetlink
  fi
  DISK_GB="${JETLINK_TEST_FREE_GB:-$(free_gb "$CACHE_DIR")}"
  check_binary
  show_found
  preflight

  if [ "$HAD_INSTALL" = 1 ] && [ "$OPT_UPDATE" = 0 ] && [ "$OPT_RECONFIGURE" = 0 ] && [ "$INTERACTIVE" = 1 ]; then
    local keep
    ask_yn keep y "Jetlink is already installed. Keep your current settings and update it?"
    [ "$keep" = y ] && OPT_UPDATE=1
  fi
  if [ "$OPT_UPDATE" = 0 ] || [ "$HAD_INSTALL" = 0 ]; then
    ask_questions   # on an update the saved answers stand
  fi
  jetson_musts

  resolve_ref
  choose_server
  show_plan

  if [ "$OPT_UPDATE" = 0 ] || [ "$HAD_INSTALL" = 0 ]; then
    local go
    ask_yn go y "Go ahead?"
    [ "$go" = y ] || { say ""; say "  Nothing changed."; say ""; exit 0; }
  fi
  if [ "$OPT_DRY_RUN" = 1 ]; then
    heading "Dry run: stopping here. Nothing was changed."
    say ""
    exit 0
  fi

  get_root
  heading "Installing"
  install_base_packages
  prepare_source
  backup_install
  others_units
  hold_sleep
  get_server
  ensure_runtime
  check_gpu
  stop_running_server
  switch_server
  configure_jetson
  install_files
  start_server
  finish
}

main "$@"
