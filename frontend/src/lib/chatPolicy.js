const CODE_FIRST_SYSTEM_PROMPT = 'begin immediately with complete runnable code. No intro. Output one self-contained file unless the user asks otherwise. For Python, start exactly with ```python, include imports, and close the fence after the complete script. For Python games, prefer tkinter from the standard library over pygame, keep it compact, and include a complete runnable event loop. For HTML output ONE self-contained file. Never use external files or script src. Include inline <style> and inline <script> with working click/game logic before </body>. Start exactly with ```html then <!doctype html> and close the fence after </html>.'

export function looksLikeCodePrompt(value) {
  const text = String(value || '').toLowerCase()
  // Planning language is authoritative even when the prompt names a language
  // and begins with "write". Otherwise "Write a Python implementation plan"
  // takes the language fast path before this guard can protect it. A direct
  // request for code remains code-y when "architecture" merely names what the
  // requested implementation follows.
  const directCodeArtifact = /\b(code|source code|runnable|single file|self-contained file)\b/.test(text)
    && /\b(build|create|generate|implement|make|output|provide|write)\b/.test(text)
  const planningDeliverable = /\b(task list|task-list|checklist|implementation plan|roadmap|methodology|multi-step plan|requirements?|architecture|phases?)\b/.test(text)
  if (planningDeliverable && !directCodeArtifact) return false
  const explicitRunnableRequest = directCodeArtifact || (
    /\b(html|css|javascript|python)\b/.test(text)
      && /\b(generate|output|write)\b/.test(text)
  )
  if (explicitRunnableRequest) return true
  return /\b(code|build|create|implement|write|make)\b/.test(text)
    && /\b(html|html5|css|javascript|js|python|py|pygame|game|pacman|pacmac|tetris|app|component|page|website)\b/.test(text)
}

export function codePolicyForMessages(messages) {
  const lastUser = [...messages].reverse().find(message => message.role === 'user')
  const content = Array.isArray(lastUser?.content) ? lastUser.content.filter(part => part.type === 'text').map(part => part.text).join(' ') : lastUser?.content
  return looksLikeCodePrompt(content) ? CODE_FIRST_SYSTEM_PROMPT : ''
}
