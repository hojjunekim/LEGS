#!/usr/bin/env bash
# Fetch the large LEGS assets that are not stored in git (see scripts/assets.sha256).
#
#   scripts/download_assets.sh core             # G1, object and table meshes, flightroom splat (required)
#   scripts/download_assets.sh paper            # wood-table splat + HDR-refit meshes (paper renders)
#   scripts/download_assets.sh aug              # appearance study (blue table, apple, box)
#   scripts/download_assets.sh sources          # raw SAM 3D meshes (mesh-only baseline, object_gen/)
#   scripts/download_assets.sh all
#   scripts/download_assets.sh --list           # groups, files and sizes
#   scripts/download_assets.sh --dry-run aug    # show what would be fetched
#   scripts/download_assets.sh core --archive FILE   # install core from a downloaded archive
#
# Files land at their repo-relative paths. Files whose checksum already matches
# are skipped, so re-running is cheap; every download is verified. The core
# group is one archive on Google Drive; the other groups are files on Hugging Face.
#
# Environment:
#   LEGS_CORE_ARCHIVE_URL  URL of the core archive (default: its Google Drive file)
#   LEGS_ASSETS_REPO       Hugging Face dataset repo (default hojjunekim/LEGS-assets)
#   LEGS_ASSETS_REVISION   branch, tag or commit (default main)
#   LEGS_ASSETS_BASE_URL   full base URL, overrides the two above (e.g. a mirror)
#   HF_TOKEN               sent as a bearer token if set (private or gated repos)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/scripts/assets.sha256"
BASE_URL="${LEGS_ASSETS_BASE_URL:-https://huggingface.co/datasets/${LEGS_ASSETS_REPO:-hojjunekim/LEGS-assets}/resolve/${LEGS_ASSETS_REVISION:-main}}"
BASE_URL="${BASE_URL%/}"

# The core group is one archive; its files and their checksums are in the manifest.
CORE_ARCHIVE=LEGS-core-assets.tar.gz
CORE_ARCHIVE_BYTES=186513382
CORE_ARCHIVE_SHA256=824321f700541e532de46d187f1641d1fa0d63e7a587de997029e3786ef27d7d
CORE_DRIVE_ID=15lTnNmjF-sB8RUzDTdFPacZu8WVzstv6
CORE_VIEW_URL="https://drive.google.com/file/d/${CORE_DRIVE_ID}/view"
DRIVE_DOWNLOAD=https://drive.usercontent.google.com/download
CORE_URL="${LEGS_CORE_ARCHIVE_URL:-$DRIVE_DOWNLOAD?id=${CORE_DRIVE_ID}&export=download&confirm=t}"
CORE_TOP=LEGS-core-assets   # top directory inside the archive

usage() {
  sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

die() { echo "error: $*" >&2; exit 1; }

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

mb() {
  if [ "$1" -lt 100000 ]; then echo "$(( ($1 + 999) / 1000 )) kB"
  else local r=$(( $1 + 50000 )); echo "$(( r / 1000000 )).$(( (r % 1000000) / 100000 )) MB"; fi
}

nbytes() { wc -c < "$1" | tr -d ' '; }

[ -f "$MANIFEST" ] || die "manifest not found: $MANIFEST"

# Manifest rows: <group> <sha256> <bytes> <path>
GROUPS_=() SHAS=() SIZES=() PATHS=()
while read -r g s n p; do
  case "$g" in ''|'#'*) continue ;; esac
  case "$p" in /*|*..*) die "refusing unsafe manifest path: $p" ;; esac
  GROUPS_+=("$g") SHAS+=("$s") SIZES+=("$n") PATHS+=("$p")
done < "$MANIFEST"
KNOWN="$(printf '%s\n' "${GROUPS_[@]}" | awk '!seen[$0]++' | tr '\n' ' ')"

# intact FILE I: FILE has the size and sha256 of manifest row I.
intact() {
  [ -f "$1" ] && [ "$(nbytes "$1")" = "${SIZES[$2]}" ] && [ "$(sha256 "$1")" = "${SHAS[$2]}" ]
}

DRY=0 LIST=0 ARCHIVE='' WANT=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --list) LIST=1 ;;
    --dry-run|-n) DRY=1 ;;
    --archive) [ $# -ge 2 ] || die "--archive needs a file"; ARCHIVE=$2; shift ;;
    --archive=*) ARCHIVE=${1#--archive=} ;;
    -*) echo "unknown option: $1" >&2; usage 2 ;;
    all) WANT+=($KNOWN) ;;
    *) [[ " $KNOWN " == *" $1 "* ]] || die "unknown group '$1' (groups: ${KNOWN}all)"
       WANT+=("$1") ;;
  esac
  shift
done
if [ -n "$ARCHIVE" ]; then
  [ -f "$ARCHIVE" ] || die "no such file: $ARCHIVE"
  ARCHIVE="$(cd "$(dirname "$ARCHIVE")" && pwd)/$(basename "$ARCHIVE")"
  [[ " ${WANT[*]-} " == *" core "* ]] || WANT+=(core)
fi

if [ "$LIST" = 1 ]; then
  for g in $KNOWN; do
    tot=0 cnt=0 miss=0 first=
    for i in "${!PATHS[@]}"; do
      [ "${GROUPS_[$i]}" = "$g" ] || continue
      tot=$(( tot + SIZES[i] )) cnt=$(( cnt + 1 ))
      if [ ! -f "$ROOT/${PATHS[$i]}" ]; then
        miss=$(( miss + 1 )); [ -n "$first" ] || first=${PATHS[$i]}
      fi
    done
    if [ "$g" = core ]; then
      echo "$g: $cnt files, $(mb $tot), downloaded as one archive ($(mb $CORE_ARCHIVE_BYTES))"
      if [ "$miss" = 0 ]; then echo "  all present"; else echo "  $miss missing, e.g. $first"; fi
      continue
    fi
    echo "$g: $cnt files, $(mb $tot)"
    for i in "${!PATHS[@]}"; do
      [ "${GROUPS_[$i]}" = "$g" ] || continue
      state="missing"
      [ -f "$ROOT/${PATHS[$i]}" ] && state="present"
      printf '  %-44s %10s  %s\n' "${PATHS[$i]}" "$(mb "${SIZES[$i]}")" "$state"
    done
  done
  exit 0
fi

[ "${#WANT[@]}" -gt 0 ] || usage 2
command -v curl >/dev/null 2>&1 || die "curl is required"

AUTH=()
[ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
PROGRESS=(-sS)
[ -t 2 ] && PROGRESS=(--progress-bar)

TMP=
trap '[ -z "$TMP" ] || rm -rf -- "$TMP"' EXIT

archive_ok() {
  [ -f "$1" ] && [ "$(nbytes "$1")" = "$CORE_ARCHIVE_BYTES" ] && [ "$(sha256 "$1")" = "$CORE_ARCHIVE_SHA256" ]
}

# fetch_core OUT: download the core archive to OUT; true once it verifies.
fetch_core() {
  local q
  curl -fL --retry 3 --retry-delay 2 "${PROGRESS[@]}" -o "$1" "$CORE_URL" && archive_ok "$1" && return 0
  [ -z "${LEGS_CORE_ARCHIVE_URL:-}" ] || return 1
  # Google Drive can answer with a "can't scan this file for viruses" page whose
  # form confirms the download; resubmit the form's hidden fields.
  if [ -f "$1" ] && grep -q 'name="uuid"' "$1"; then
    q=$(grep -o '<input type="hidden" name="[^"]*" value="[^"]*"' "$1" \
          | sed 's/.*name="\([^"]*\)" value="\([^"]*\)"/\1=\2/' | paste -s -d '&' -)
    curl -fL --retry 3 --retry-delay 2 "${PROGRESS[@]}" -o "$1" "$DRIVE_DOWNLOAD?$q" && archive_ok "$1" && return 0
  fi
  # gdown (in the legs env) follows more of Drive's download flows.
  if command -v gdown >/dev/null 2>&1; then
    rm -f "$1"
    gdown --quiet "$CORE_DRIVE_ID" -O "$1" && archive_ok "$1" && return 0
  fi
  return 1
}

fetched=0 skipped=0 failed=0

# --- core: one archive, unpacked to a temp dir under data/; each file is checked
# against the manifest before it is moved into place.
if [[ " ${WANT[*]} " == *" core "* ]]; then
  ncore=0 MISSING=()
  for i in "${!PATHS[@]}"; do
    [ "${GROUPS_[$i]}" = core ] || continue
    ncore=$(( ncore + 1 ))
    intact "$ROOT/${PATHS[$i]}" "$i" || MISSING+=("$i")
  done
  src_desc=${ARCHIVE:-$CORE_URL}
  if [ "${#MISSING[@]}" = 0 ]; then
    echo "ok       core ($ncore files)"
    skipped=$(( skipped + ncore ))
  elif [ "$DRY" = 1 ]; then
    echo "fetch    core: ${#MISSING[@]} of $ncore files, from $CORE_ARCHIVE ($(mb $CORE_ARCHIVE_BYTES))  <- $src_desc"
  else
    skipped=$(( skipped + ncore - ${#MISSING[@]} ))
    mkdir -p "$ROOT/data"
    TMP=$(mktemp -d "$ROOT/data/.core-assets.XXXXXX")
    archive=$ARCHIVE ok=1
    if [ -n "$ARCHIVE" ]; then
      echo "unpack   core: ${#MISSING[@]} of $ncore files, from $ARCHIVE"
      archive_ok "$ARCHIVE" || { echo "error: $ARCHIVE is not $CORE_ARCHIVE (size or sha256 differs)" >&2; ok=0; }
    else
      echo "fetch    core: ${#MISSING[@]} of $ncore files, from $CORE_ARCHIVE ($(mb $CORE_ARCHIVE_BYTES))"
      archive="$TMP/$CORE_ARCHIVE"
      if ! fetch_core "$archive"; then
        ok=0
        if [ -n "${LEGS_CORE_ARCHIVE_URL:-}" ]; then
          echo "error: could not download $CORE_ARCHIVE from $CORE_URL (or the file differs)" >&2
        else
          echo "error: Google Drive did not hand out $CORE_ARCHIVE. It limits downloads of a" >&2
          echo "       widely shared file for up to a day. Download it in a browser from" >&2
          echo "         $CORE_VIEW_URL" >&2
          echo "       then run: scripts/download_assets.sh core --archive <downloaded file>" >&2
        fi
      fi
    fi
    if [ "$ok" = 1 ]; then
      mkdir "$TMP/x"
      if tar -xzf "$archive" -C "$TMP/x"; then
        [ -n "$ARCHIVE" ] || rm -f "$archive"
        for i in "${MISSING[@]}"; do
          rel="${PATHS[$i]}" src="$TMP/x/$CORE_TOP/${PATHS[$i]}" dst="$ROOT/${PATHS[$i]}"
          if ! intact "$src" "$i"; then
            echo "error: $rel in $CORE_ARCHIVE does not match scripts/assets.sha256" >&2
            failed=$(( failed + 1 ))
            continue
          fi
          mkdir -p "$(dirname "$dst")"
          mv -f "$src" "$dst"
          fetched=$(( fetched + 1 ))
        done
      else
        echo "error: could not unpack $archive" >&2
        ok=0
      fi
    fi
    [ "$ok" = 1 ] || failed=$(( failed + ${#MISSING[@]} ))
    rm -rf -- "$TMP"; TMP=
  fi
fi

# --- the other groups: one file each from Hugging Face
for i in "${!PATHS[@]}"; do
  [ "${GROUPS_[$i]}" != core ] || continue
  [[ " ${WANT[*]} " == *" ${GROUPS_[$i]} "* ]] || continue
  rel="${PATHS[$i]}" want="${SHAS[$i]}" dst="$ROOT/${PATHS[$i]}"
  if intact "$dst" "$i"; then
    echo "ok       $rel"
    skipped=$(( skipped + 1 ))
    continue
  fi
  if [ "$DRY" = 1 ]; then
    echo "fetch    $rel ($(mb "${SIZES[$i]}"))  <- $BASE_URL/$rel"
    continue
  fi
  echo "fetch    $rel ($(mb "${SIZES[$i]}"))"
  mkdir -p "$(dirname "$dst")"
  part="$dst.part"
  rm -f "$part"
  if ! curl -fL --retry 3 --retry-delay 2 "${PROGRESS[@]}" ${AUTH[@]+"${AUTH[@]}"} -o "$part" "$BASE_URL/$rel"; then
    rm -f "$part"
    echo "error: could not download $BASE_URL/$rel" >&2
    echo "       check the network; for a private repo set HF_TOKEN, or point" >&2
    echo "       LEGS_ASSETS_REPO / LEGS_ASSETS_BASE_URL at a mirror" >&2
    failed=$(( failed + 1 ))
    continue
  fi
  got="$(sha256 "$part")"
  if [ "$got" != "$want" ]; then
    rm -f "$part"
    echo "error: checksum mismatch for $rel (expected $want, got $got)" >&2
    failed=$(( failed + 1 ))
    continue
  fi
  mv -f "$part" "$dst"
  fetched=$(( fetched + 1 ))
done

[ "$DRY" = 1 ] && exit 0
echo "done: $fetched fetched, $skipped already present, $failed failed"
if [[ " ${WANT[*]} " != *" core "* ]]; then
  for i in "${!PATHS[@]}"; do
    if [ "${GROUPS_[$i]}" = core ] && [ ! -f "$ROOT/${PATHS[$i]}" ]; then
      echo "note: the core assets, which the simulator and renderers need, are missing:"
      echo "      scripts/download_assets.sh core"
      break
    fi
  done
fi
[ "$failed" = 0 ]
