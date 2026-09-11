# Cluster view — P1 (honest fabric report)

The Cluster page used to be a diagram editor: nodes were drawn by hand, saved to
browser storage, and a green `live` chip meant a string in that storage said
`running`. It now reports one `camelid fabric serve` proxy, read live from that
proxy's own `/v1/health`.

Captured with `frontend/scripts/capture-fabric-view.mjs`, which drives the real
built component against a scripted proxy on a separate origin — the same
cross-origin read the shipped app performs. Nodes are named by hostname because
a committed evidence bundle may contain no IPv4 literal except `127.0.0.1`.

| image | what it shows |
|---|---|
| `01-nodes-desktop.png` | Three nodes, one per state. A ready node shows its model, `2 in flight · 1 waiting` and a probe time; a node that is not ready shows its reason and an explicit *unknown* for model and load; an unreachable node adds an unknown probe time. The header reads **degraded**, because one of three is ready. |
| `02-node-detail-desktop.png` | The node drawer for the not-ready node. It reports state, reason, and an unknown for everything only a ready node publishes. It offers no worker start/stop control, because the page cannot perform one. |
| `03-detail-withheld-desktop.png` | The proxy answered but is **not bound to loopback**, so it withheld node addresses and model names. No counts, no table, an explicit reason, and the readiness it did disclose. This is not an empty fabric and must never render as one. |
| `04-empty-fabric-desktop.png` | A proxy that answered and really is configured with no nodes: counts of zero, and the command that fixes it. |
| `05-no-proxy-desktop.png` | Nothing answered. An honest failure and a copyable command — never a node list from a previous read. |
| `06-nodes-mobile-390.png` | 390×844. Each row becomes a labelled card. No horizontal overflow, and no column is dropped: a hidden column is a fact the operator silently stops being told. |

## What these images are evidence of

1. *Unknown* is rendered as a distinct value — italic, muted, dotted — never as
   `0`, never as an empty cell. A node that is not ready publishes no load at
   all, so showing `0 in flight` would invent an idle machine.
2. "Withheld" and "empty" are different screens. The proxy only discloses
   `nodes`, `models` and `node_detail` when it is bound to loopback, and its own
   tests assert that distinction server-side; these show the client keeping it.
3. Every value on screen traces to a field in the answer to a request made
   moments before. Nothing is read from browser storage except the address the
   operator typed.

Verify with `sha256sum -c SHA256SUMS`.
