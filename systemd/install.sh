#!/usr/bin/env bash
# Install qwen38-flash as a systemd service that runs scripts/serve.sh under the
# invoking user (must be in the docker group). Idempotent; re-run after editing the
# unit template. Requires sudo for /etc/systemd/system.
#
#   systemd/install.sh            # install + enable (starts on boot)
#   sudo systemctl start qwen38-flash
#
# Starting the unit replaces any detached container of the same name started by a
# plain `scripts/serve.sh` run.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(id -un)}"
UNIT=/etc/systemd/system/qwen38-flash.service

if ! id -nG "$RUN_USER" | tr ' ' '\n' | grep -qx docker; then
  echo "!! $RUN_USER is not in the docker group; the service would fail to start"; exit 1
fi

sed -e "s|@USER@|$RUN_USER|g" -e "s|@REPO@|$REPO|g" "$REPO/systemd/qwen38-flash.service" \
  | sudo tee "$UNIT" >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable qwen38-flash

echo ">> installed $UNIT (user=$RUN_USER, repo=$REPO), enabled at boot"
echo ">> settings: $REPO/systemd/qwen38-flash.env"
echo ">> start now:  sudo systemctl start qwen38-flash   (replaces the running detached container)"
echo ">> follow:     journalctl -u qwen38-flash -f"
