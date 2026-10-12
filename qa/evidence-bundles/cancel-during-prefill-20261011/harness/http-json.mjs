// POST JSON without fetch's 300 s headers timeout: a non-streaming request on a slow
// CPU lane (or queued behind abandoned work) can take longer than that.
import http from "node:http"

export function postJson(base, path, body, signal) {
  const url = new URL(path, base)
  const payload = JSON.stringify(body)
  return new Promise((resolve, reject) => {
    const req = http.request(
      { hostname: url.hostname, port: url.port, path: url.pathname, method: "POST", signal,
        headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(payload) } },
      (res) => {
        let text = ""
        res.setEncoding("utf8")
        res.on("data", (chunk) => { text += chunk })
        res.on("end", () => {
          try {
            resolve({ status: res.statusCode, body: JSON.parse(text) })
          } catch {
            resolve({ status: res.statusCode, body: null, text })
          }
        })
        res.on("error", reject)
      },
    )
    req.on("error", reject)
    req.end(payload)
  })
}
