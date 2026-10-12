#!/usr/bin/env bash
# Restore a console-data backup made by console-backup.sh into an EXISTING
# named volume. The target volume is a required argument so that a restore
# never lands in a default volume by accident. Restoring once into a real,
# empty volume is a human step; this script cannot assert it about itself.
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <backup-tar.gz> <volume-name>" >&2
    echo "  e.g.: $0 /path/to/backups/docker_console-data-2000-01-01-020000.tar.gz docker_console-data" >&2
    exit 1
fi

BACKUP_FILE="$1"
VOLUME_NAME="$2"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "no such backup file: $BACKUP_FILE" >&2
    exit 1
fi

if ! docker volume inspect "$VOLUME_NAME" >/dev/null 2>&1; then
    candidates="$(docker volume ls -q | tr '\n' ' ')"
    echo "no such docker volume: ${VOLUME_NAME} (existing volumes: ${candidates})" >&2
    echo "refusing to silently provision it -- if this is a genuinely new deploy," >&2
    echo "provision the volume explicitly first, then re-run this script." >&2
    exit 1
fi

docker run --rm \
    -v "${VOLUME_NAME}:/data" \
    -v "$(dirname "$(readlink -f "$BACKUP_FILE")"):/backup" \
    alpine \
    tar -xzf "/backup/$(basename "$BACKUP_FILE")" -C /data

echo "restored ${BACKUP_FILE} -> volume ${VOLUME_NAME}"
