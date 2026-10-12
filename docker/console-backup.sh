#!/usr/bin/env bash
# Tar the console-data named volume(s) to a date-stamped archive in a backup
# directory, keeping the last 14 per volume. Intended to run nightly from cron;
# see docker/README.md. Point CONSOLE_BACKUP_DIR at a location that survives
# loss of the machine's own disk.
#
# Volume names default to what compose actually creates -- the `console-data`
# service volume is pinned to `docker_console-data` in console-compose.yaml.
# Any other default would let docker silently auto-create an empty volume and
# "back up" nothing.
set -euo pipefail

DRILL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --restore-drill) DRILL=1; shift ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

# REQUIRED: the run fails loudly (non-zero) if this volume is missing -- it
# holds jobs.sqlite3 and schedules (docker/README.md "Backup"), and a
# skip-and-continue here would exit 0, i.e. a false-success backup that
# captured nothing. OPTIONAL: further volumes named (space-separated) in
# CONSOLE_OPTIONAL_VOLUME_NAMES, for example a second instance's volume that
# may not exist on every host; an optional volume's absence is recorded but
# never fails the run.
REQUIRED_VOLUME_NAMES=(
    "${CONSOLE_VOLUME_NAME:-docker_console-data}"
)
read -r -a OPTIONAL_VOLUME_NAMES <<< "${CONSOLE_OPTIONAL_VOLUME_NAMES:-}"
# Default: a "console-backup" directory under the per-user data home
# (GMS_DATA_HOME, else $XDG_DATA_HOME/gomoveshift, else ~/.local/share/gomoveshift).
DATA_HOME="${GMS_DATA_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/gomoveshift}"
BACKUP_DIR="${CONSOLE_BACKUP_DIR:-$DATA_HOME/console-backup}"
KEEP=14

# --restore-drill proves a backup is actually restorable instead of just
# written: listing (or, worse, extracting only) the first member of an archive
# proves tar can open the file, and nothing about whether the data this script
# exists to protect would come back. So the drill opens the NEWEST archive per required
# volume, actually extracts it, and integrity-checks any sqlite database it
# finds. No docker is involved -- the archive is a plain tar.gz on disk, so
# this can run from cron or by hand with nothing running.
#
# Three verdicts, deliberately distinct: a
# required volume with no archive, or an archive that extracts nothing, is
# FAILED and exits non-zero -- that backup is not there. A database that
# extracted but could not be integrity-checked (no `sqlite3` on PATH) is
# INCOMPLETE and exits 0: nothing is known to be wrong, but the word is not
# OK, because naming what was not proven is the point. Optional volumes are
# drilled when an archive exists but never fail the run when one does not --
# their absence is already a known, accepted state.
if [ "$DRILL" -eq 1 ]; then
    drill_missing=0
    drill_unproven=""
    for volume in "${REQUIRED_VOLUME_NAMES[@]}"; do
        # shellcheck disable=SC2012
        # `ls -1t | head -1` picks the newest archive; a glob with zero
        # matches under `set -e -o pipefail` would abort the loop, so this
        # keeps the `2>/dev/null` guard.
        # `|| true`: under `set -e -o pipefail`, a glob with zero matches
        # makes `ls` exit non-zero and would otherwise abort the whole
        # script -- an empty backup dir must produce a MISSING/skip message,
        # never a silent early exit.
        newest=$(ls -1t "$BACKUP_DIR/${volume}"-*.tar.gz 2>/dev/null | head -1) || true
        if [ -z "$newest" ]; then
            echo "restore-drill: MISSING ${volume} -- no archive in $BACKUP_DIR" >&2
            drill_missing=1
            continue
        fi
        out=$(mktemp -d)
        tar -xzf "$newest" -C "$out" 2>/dev/null || true
        files=0
        while IFS= read -r f; do
            [ -f "$f" ] || continue
            files=$((files + 1))
            case "$f" in
                *.sqlite3|*.sqlite|*.db)
                    if command -v sqlite3 >/dev/null 2>&1; then
                        res=$(sqlite3 "$f" 'PRAGMA integrity_check' 2>/dev/null)
                        if [ "$res" != "ok" ]; then
                            echo "restore-drill: CORRUPT ${f#"$out"/} in ${newest} (${res})" >&2
                            drill_missing=1
                        fi
                    else
                        drill_unproven="sqlite3 not installed, so no database integrity check ran"
                    fi
                    ;;
            esac
        done < <(find "$out" -type f 2>/dev/null)
        rm -rf "$out"
        if [ "$files" -eq 0 ]; then
            echo "restore-drill: MISSING ${volume} -- ${newest} extracted nothing" >&2
            drill_missing=1
        else
            echo "restore-drill: ok ${volume} -- ${files} file(s) from ${newest}"
        fi
    done
    for volume in ${OPTIONAL_VOLUME_NAMES[@]+"${OPTIONAL_VOLUME_NAMES[@]}"}; do
        # shellcheck disable=SC2012
        # `|| true`: under `set -e -o pipefail`, a glob with zero matches
        # makes `ls` exit non-zero and would otherwise abort the whole
        # script -- an empty backup dir must produce a MISSING/skip message,
        # never a silent early exit.
        newest=$(ls -1t "$BACKUP_DIR/${volume}"-*.tar.gz 2>/dev/null | head -1) || true
        if [ -z "$newest" ]; then
            echo "restore-drill: optional ${volume} has no archive, skipped"
        else
            echo "restore-drill: optional ${volume} has an archive at ${newest} (not drilled)"
        fi
    done
    # Exit code 5, distinct from 3 (required volume missing at backup time)
    # and 4 (nothing backed up): this is "a restore was proven to fail", a
    # different failure mode from either backup-time code above.
    if [ "$drill_missing" -eq 1 ]; then
        echo "restore-drill FAILED" >&2
        exit 5
    fi
    if [ -n "$drill_unproven" ]; then
        echo "restore-drill INCOMPLETE: $drill_unproven"
        exit 0
    fi
    echo "restore-drill OK: every required volume's newest archive extracted and, where a database was present, passed integrity check"
    exit 0
fi

mkdir -p "$BACKUP_DIR"
stamp="$(date +%Y-%m-%d-%H%M%S)"
required_missing=0
backed_up=0

backup_volume() {  # $1=volume $2=required(1|0)
    local volume="$1" required="$2" candidates dest
    if ! docker volume inspect "$volume" >/dev/null 2>&1; then
        candidates="$(docker volume ls -q | tr '\n' ' ')"
        if [ "$required" -eq 1 ]; then
            echo "REQUIRED volume missing: ${volume} (existing volumes: ${candidates})" >&2
            required_missing=1
        else
            echo "skipping optional ${volume}: no such docker volume (existing volumes: ${candidates})" >&2
        fi
        return
    fi

    dest="$BACKUP_DIR/${volume}-${stamp}.tar.gz"
    docker run --rm \
        -v "${volume}:/data:ro" \
        -v "${BACKUP_DIR}:/backup" \
        alpine \
        tar -czf "/backup/$(basename "$dest")" -C /data .

    echo "backed up ${volume} -> ${dest}"
    backed_up=$((backed_up + 1))

    # shellcheck disable=SC2012
    # `|| true`: under `set -e -o pipefail`, a glob with zero prior backups
    # makes `ls` exit non-zero and would otherwise abort the whole script --
    # a pruning no-op must never look like a backup failure.
    ls -1t "$BACKUP_DIR/${volume}"-*.tar.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f || true
}

for volume in "${REQUIRED_VOLUME_NAMES[@]}"; do
    backup_volume "$volume" 1
done
for volume in ${OPTIONAL_VOLUME_NAMES[@]+"${OPTIONAL_VOLUME_NAMES[@]}"}; do
    backup_volume "$volume" 0
done

# A host that holds more than one console-data volume while no optional volume
# is configured is backing up only the required one: say so, so a dropped
# volume is visible in the cron log rather than silently skipped.
if [ "${#OPTIONAL_VOLUME_NAMES[@]}" -eq 0 ]; then
    unlisted=""
    while IFS= read -r existing; do
        case "$existing" in
            *console-data*)
                [ "$existing" = "${REQUIRED_VOLUME_NAMES[0]}" ] || unlisted="${unlisted} ${existing}"
                ;;
        esac
    done < <(docker volume ls -q 2>/dev/null || true)
    if [ -n "$unlisted" ]; then
        echo "note: console-data volume(s) not backed up:${unlisted} -- list them in CONSOLE_OPTIONAL_VOLUME_NAMES" >&2
    fi
fi

# Distinct exit codes so a caller (and a test) can tell "a required volume
# was missing" apart from "the run died for some other reason" -- a crash
# elsewhere in the script must never look identical to this specific failure.
if [ "$required_missing" -eq 1 ]; then
    echo "console-backup FAILED: a required volume was missing, backup incomplete" >&2
    exit 3
fi
if [ "$backed_up" -eq 0 ]; then
    echo "console-backup FAILED: no volume was backed up" >&2
    exit 4
fi
