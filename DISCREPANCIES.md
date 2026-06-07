# Discrepancies between ESP32 and Raspberry Pi Clients

This document tracks functional and protocol-level differences between the ESP32 (`atlantico`) and Raspberry Pi (`atlantico-raspberry`) client implementations.

## 1. Protocol Discrepancies (MQTT Commands)

| Command / Feature | ESP32 Implementation | Raspberry Pi Implementation | Impact |
| :--- | :--- | :--- | :--- |
| `federate_waiting` | Checks if idle for the specific round and triggers `RESUME` setup. | **Ignored** | Pi clients may not "auto-resume" if they miss a global round start. |
| `federate_reboot` | Restarts the ESP32 hardware. | **Ignored** | Server cannot remotely reset the Pi client. |
| `federate_stop` | Sets state to `DONE` and stops training. | **Ignored** | Pi clients may continue training until a global unsubscribe. |
| `alive` Response | Sends `round` and `newModelState` (e.g., "idle", "busy"). | Basic `alive` notification only. | Server TUI shows less detail for Pi clients. |
| `clients` targeting | Supports JSON array of client IDs for targeted commands. | Supports only single `client` string. | Batch-targeted commands from server may be ignored by Pi. |

## 2. Feature & Reporting Discrepancies

| Feature | ESP32 Implementation | Raspberry Pi Implementation |
| :--- | :--- | :--- |
| **Double Dataset** | Supports simulating two logical clients on one device via `#ifdef DOUBLE_DATASET`. | **Not Supported** (One process = One client). |
| **Memory Stats** | Reports detailed heap usage (`fixed` and `round` memory). | **Not Reported**. |
| **Timing Stats** | Reports `previousTransmit` and `previousConstruct` diagnostics. | Calculated but **not included** in JSON payload. |
| **Precision** | Explicitly reports "float" or "double" based on build flags. | Defaults to JSON-standard float serialization. |

## 3. Recently Unified Features (Parity Achieved)

- **Balanced Metrics:** Both platforms now calculate and report macro-averaged and support-weighted (balanced) accuracy, precision, recall, and F1.
- **Dataset Aliasing:** Both platforms support `database` as an alias for `dataset`/`datasetKey`.
- **ID-Matched Binaries:** Both platforms automatically search for `.bin` files matching their client ID suffix (e.g., `*01.bin`).
- **JSON Weights:** Both platforms support the `jsonWeights` flag in `federate_start` to toggle weight reporting in JSON.
