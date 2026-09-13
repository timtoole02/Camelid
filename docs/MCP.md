# Connected tools (MCP preview)

Camelid can connect to Model Context Protocol servers from **Connections** in the
local Web UI or Desktop app. It supports local commands over stdio and remote
Streamable HTTP endpoints. The official Rust MCP client SDK handles protocol
initialization, discovery, and transport messages.

## Use a connection

1. Open **Connections → Add server** and give the server a name.
2. For a local server, enter its executable and arguments as a JSON array. Use
   an installed executable, for example `node` with `["/path/to/server.js"]`.
   Camelid passes arguments directly; it does not interpret shell operators.
3. For a remote server, enter its HTTPS MCP endpoint. Localhost endpoints may
   use HTTP. If authentication is needed, enter the name of an environment
   variable containing the bearer token, and start Camelid with that variable.
4. Save, then choose **Connect**. Connecting a local server starts its program
   with your user account. Expand **Details → Available tools** to inspect the discovered tools.
5. Load a model whose exact compatibility row supports tools. In Chat, open
   **Tools** beside the composer model picker. Search by tool, description, or
   server name, then select individual tools or a whole server group.
6. Send a prompt. Camelid shows the server name, tool name, and exact arguments
   in the conversation before execution. Choose **Allow once**, **Deny**, or **Stop**. An allowed
   call's result goes back to the model automatically; results remain visible
   in the conversation. Denial is also sent as a tool result.

A connection is not a sandbox. Only connect programs and services you trust.
Local servers receive the normal platform path/home/temp environment variables,
plus any variable names explicitly listed in the form. Secret values are read
by the engine and never returned in the connection catalog. Do not put secrets
in URLs or arguments. OAuth login, arbitrary custom HTTP headers, legacy HTTP+SSE
endpoints, MCP sampling, elicitation, resources, and prompts are outside this
initial tools-only interface.

## Reuse a tool selection

The picker shows the selected count and groups tools by server. Use the **+**
beside **Tool set** to save your current selection with a name, then choose that
set in another conversation. A matching name replaces the existing set. The
trash button deletes the saved set without changing the current selection.
**Clear selection** removes tools from the current conversation.

Tool sets are saved on this device using the same local storage as conversations.
They contain exact tool identifiers, not connection settings or credentials.
Applying a set never connects a server. If a saved tool is unavailable, reconnect
its server or choose **Remove unavailable** before sending. Reconfigured tools
may have different identifiers and must be selected again. Selecting an entire
server group that would exceed 16 tools leaves the current selection unchanged.

Use **Manage connections** to add or remove servers, inspect their tools, or
filter by connection status. Disconnected servers can also be connected directly
from the picker. Saving a server configuration does not connect it.

## State and limits

Connection configuration is stored in `mcp-connections.json` beside Camelid's
Workspace memory database (`$XDG_DATA_HOME/camelid`, normally
`~/.local/share/camelid`, or `%LOCALAPPDATA%/camelid` on Windows). Unix writes use
owner-only permissions and an atomic replacement. Configuration contains
variable names, not their secret values. Saved connections return disconnected
after restart; opening Camelid never launches a saved server automatically.
Tool choices and call/result history are local conversation state. Connection
configuration and tool selections are not included in conversation exports.

- Up to 16 saved connections and 128 discovered tools per server.
- Up to 16 selected tools per conversation and 8 tool rounds per send.
- Up to 20 named tool sets on this device, with names up to 80 characters.
- Initialization/discovery timeout: 20 seconds; approval lifetime: 5 minutes.
- Execution timeout: 60 seconds. Results larger than 64 KB are explicitly
  truncated and marked as errors so the model can narrow its request.
- A repeated call with the same arguments stops the loop.
- Every call has a single-use approval receipt. Retrying an approval cannot
  execute the call again. Transport-level expired-session retries are disabled.
- Stop closes an executing tool's connection. This terminates local process
  groups (Windows Job Objects) but cannot undo remote side effects. A timeout
  or lost result must never be interpreted as proof that an action did not run.

The raw definition editor is under **Tools → Advanced → Manual tool definitions**.
Enabling manual definitions clears connected selections; selecting connected
tools switches manual definitions off. Manual calls are returned for inspection
and are not executed by Camelid. Tool selections remain fixed throughout a turn,
including while approval is pending. Structured output must be off for connected
tool use.

This feature is separate from read-only Workspace. It does not change Workspace's
read-only tool profile or promote any model's tool capability. Model quality
continues to depend on the exact model's existing tool evidence.

## Local API

All `/api/mcp/*` routes require a loopback-bound engine, same-origin browser
metadata, and `X-Camelid-Mcp: 1`, in addition to the server's configured API
credential. They are unavailable on the LAN chat-only surface. Vite's same-origin
API proxy supports development. Endpoint configuration cannot come from model
output: model calls only reference namespaced tools already discovered by an
explicitly connected server.

| Route | Operation |
| --- | --- |
| `GET /api/mcp/connections` | List saved configurations, connection state, and discovered tools |
| `POST /api/mcp/connections` | Save a connection without starting it |
| `DELETE /api/mcp/connections/:id` | Disconnect and remove configuration |
| `POST /api/mcp/connections/:id/connect` | Initialize the server and discover tools |
| `POST /api/mcp/connections/:id/disconnect` | Close the connection and cancel its pending work |
| `POST /api/mcp/calls` | Freeze a known tool and JSON arguments for approval |
| `POST /api/mcp/calls/:id/decision` | Accept `{ "approved": true }` once, or deny |
| `GET /api/mcp/calls/:id` | Read execution status and result |
| `DELETE /api/mcp/calls/:id` | Cancel a pending or executing call |

## Validation

```sh
cargo test --lib api::mcp::tests
cargo clippy --lib -- -D warnings
cd frontend
npm run smoke:mcp
npm run build
npm run smoke:mcp-browser
```

The backend tests run real stdio and HTTP protocol fixtures and verify approval,
replay prevention, denial, disconnect invalidation, persistence, and origin
checks. The browser test uses a deterministic chat API fixture to verify the
request/result continuation, allow/deny/stop behavior, saved selections,
connection setup, keyboard interaction, and desktop/mobile layout. These tests establish
transport and UI behavior; they do not constitute real-model tool-quality
certification.
