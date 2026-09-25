#!/usr/bin/env bash
# Install qwen38-flash as a systemd service that runs scripts/serve.sh under the
# invoking user (must be in the docker group). Idempotent; re-run after editing the
# unit template. Requires sudo for /etc/systemd/system and /etc/polkit-1/rules.d.
#
#   systemd/install.sh            # install + enable (starts on boot)
#   systemctl start qwen38-flash  # no sudo: a polkit rule lets the user manage this unit
#
# Starting the unit replaces any detached container of the same name started by a
# plain `scripts/serve.sh` run.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(id -un)}"
UNIT=/etc/systemd/system/qwen38-flash.service
POLKIT_RULE=/etc/polkit-1/rules.d/50-qwen38-flash.rules

if ! id -nG "$RUN_USER" | tr ' ' '\n' | grep -qx docker; then
  echo "!! $RUN_USER is not in the docker group; the service would fail to start"; exit 1
fi

sed -e "s|@USER@|$RUN_USER|g" -e "s|@REPO@|$REPO|g" "$REPO/systemd/qwen38-flash.service" \
  | sudo tee "$UNIT" >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable qwen38-flash

# Let $RUN_USER start/stop/restart this one unit without sudo. Scoped to the unit and
# to these verbs only; enable/disable and every other unit still need root.
sudo install -d -m 755 /etc/polkit-1/rules.d
sudo tee "$POLKIT_RULE" >/dev/null <<EOF
// Installed by $REPO/systemd/install.sh
polkit.addRule(function(action, subject) {
  if (action.id == "org.freedesktop.systemd1.manage-units" &&
      action.lookup("unit") == "qwen38-flash.service" &&
      subject.user == "$RUN_USER") {
    var verb = action.lookup("verb");
    if (verb == "start" || verb == "stop" || verb == "restart" || verb == "try-restart") {
      return polkit.Result.YES;
    }
  }
});
EOF
sudo chmod 644 "$POLKIT_RULE"

echo ">> installed $UNIT (user=$RUN_USER, repo=$REPO), enabled at boot"
echo ">> installed $POLKIT_RULE ($RUN_USER may start/stop/restart without sudo)"
echo ">> settings: $REPO/systemd/qwen38-flash.env"
echo ">> start now:  systemctl start qwen38-flash   (replaces the running detached container)"
echo ">> follow:     journalctl -u qwen38-flash -f"
