export async function changeRequest(apiBase, path = '', { method = 'GET', body, signal } = {}) {
  const response = await fetch(String(apiBase || '').replace(/\/$/, '') + '/api/changes' + path, {
    method, signal, headers: { 'Content-Type': 'application/json', 'X-Camelid-Changes': '1' },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  })
  const result = await response.json().catch(() => null)
  if (!response.ok) throw new Error(result?.error?.message || 'Could not load the file review (' + response.status + ').')
  return result
}
