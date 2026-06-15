#!/usr/bin/env bash
# Script to spawn multiple devices using device_control.sh
# Usage: ./spawn_devices.sh [action]
# Actions: start_background, stop, restart, status
# Default action: start_background

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_CONTROL="$SCRIPT_DIR/device_control.sh"

# Truncate the repository's config.json (empty its contents) before spawning devices
# This targets the config.json in the parent directory of this script (atlantico-raspberry/config.json)
CONFIG_FILE="$SCRIPT_DIR/../config.json"
if [[ -e "$CONFIG_FILE" ]]; then
    : > "$CONFIG_FILE"
    echo "Emptied $CONFIG_FILE"
else
    : > "$CONFIG_FILE"
    echo "Created empty $CONFIG_FILE"
fi

# Configuration: Array of device names to spawn
# Modify this array to change the number or names of devices
DEVICE_NAMES=(
#    "rasp_6"
#    "rasp_7"
    "rasp_8"
    "rasp_9"
    "rasp_10"
)

ACTION="${1:-start_background}"

# Validate action - restricting to non-blocking or status actions
if [[ ! "$ACTION" =~ ^(start_background|stop|restart|status)$ ]]; then
    echo "Error: Action '$ACTION' is not supported for bulk execution."
    echo "Supported actions: start_background, stop, restart, status"
    echo "Note: 'start' and 'fg' are not supported as they run in foreground blocking execution."
    exit 1
fi

echo "Executing '$ACTION' for ${#DEVICE_NAMES[@]} devices..."

for name in "${DEVICE_NAMES[@]}"; do
    echo "----------------------------------------"
    echo "Processing Device: $name"
    
    # Extract ID from name (e.g. rasp_1 -> 1) by taking everything after the last underscore
    dev_id="${name##*_}"
    data_dir="data_ready/$dev_id"
    
    # execute device_control.sh with DEVICE_INSTANCE env var set
    # passing --device-name and --data-dir arguments to the underlying python module
    DEVICE_INSTANCE="$name" "$DEVICE_CONTROL" "$ACTION" --device-name "$name" --data-dir "$data_dir"
    
done

echo "----------------------------------------"
echo "Done."
