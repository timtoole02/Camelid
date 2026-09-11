/* Clipboard write, extracted from markdown.jsx so components that markdown.jsx
   itself renders can use it without forming an import cycle.

   Returns whether the text actually reached the clipboard: callers show a
   "Copied" confirmation, and confirming a copy that never happened is worse
   than showing nothing. Access can be denied even in a secure context. */
export const copyText = async (text) => {
  try {
    if (!navigator.clipboard?.writeText) return false
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    // Clipboard access can be denied even in a secure context; rendering still works.
    return false
  }
}

export default copyText
