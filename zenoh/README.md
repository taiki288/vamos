# VAMOS Zenoh transport

This first stage receives JPEG images and Go2 state, accepts heartbeat/stop
messages, and publishes `vamos/state`. It does not load a VLA model, start
navigation, or publish velocity/posture commands to Go2.

The connection/key/QoS layout follows the
[NaVILA Zenoh node](https://github.com/Oya-Tomo/NaVILA/tree/feat/2-navila-zenoh-node/zenoh).
Settings use Pydantic v2, matching this VAMOS fork.

From the repository root, run the node:

```console
uv run --group zenoh python zenoh/node.py
```

In a second terminal, send a sample image, simulated Go2 state and heartbeat:

```console
uv run --group zenoh python zenoh/diagnostic_peer.py
```

The peer prints received dimensions, image/state freshness, and heartbeat status.
Ctrl+C stops each process. The examples use direct TCP on localhost port 7447;
multicast discovery is disabled. The diagnostic peer is a test input source,
not a real robot node. Do not run it on the same keys as a real input source.

| Key | Direction from VAMOS | Encoding |
| --- | --- | --- |
| `camera/front` | receive | `image/jpeg` binary JPEG |
| `unitree/go2/state` | receive | `application/json` |
| `vamos/command` | receive | `application/json`, `heartbeat` or `stop` |
| `vamos/state` | publish | `application/json` |

`start_ready` stays false. `stop` is acknowledged in transport diagnostics only;
there is no motion controller to stop at this stage. Stale Go2 data is not exposed
as a current observation. Invalid samples do not refresh valid data timestamps.
Only the latest decoded RGB image is retained; no image files are written.

The tracked `*.example.json5` files are runnable defaults. For a different setup,
copy them to the corresponding `*.json5` runtime paths and pass `--node-config`,
`--zenoh-config`, or `--peer-config` explicitly. Runtime files are ignored by Git.
For multiple computers, edit the listener/connect endpoints and keep keys aligned
with the existing camera/Go2 nodes. Windows and Linux use the same entry points.

Run unit and real localhost, separate-process transport tests:

```console
uv run --group zenoh python -m unittest discover -s zenoh/tests -v
```
