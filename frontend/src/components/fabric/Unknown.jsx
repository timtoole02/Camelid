/* One way to render "we were not told", so an unknown can never be mistaken for
   a zero, a no, or an empty list anywhere in the Cluster view. */
export function Unknown({ why = null, children = 'unknown' }) {
  return (
    <span className="fabric-unknown" title={why || undefined} data-unknown="true">
      {children}
    </span>
  )
}

export default Unknown
