import { IconCollection } from '../ui/icons'
import '../../styles/knowledge.css'

const docsLabel = count => `${count.toLocaleString()} ${count === 1 ? 'doc' : 'docs'}`

/* Checkboxes for collections. Checked ids that no longer exist stay listed as
   unavailable, so they can be unchecked instead of lingering unseen. */
export function CollectionChecklist({ collections, checkedIds, onToggle, disabled = false, excludeIds = [] }) {
  const known = new Set(collections.map(collection => collection.id))
  const missing = checkedIds.filter(id => !known.has(id) && !excludeIds.includes(id))
  return <>
    {collections.filter(collection => !excludeIds.includes(collection.id)).map(collection => <label className="context-check" key={collection.id}>
      <input type="checkbox" checked={checkedIds.includes(collection.id)} disabled={disabled} onChange={event => onToggle(collection.id, event.target.checked)} />
      <IconCollection size={14} /><span>{collection.name}</span><small className="context-muted">{docsLabel(collection.doc_ids.length)}</small>
    </label>)}
    {missing.map(id => <label className="context-check" key={id}>
      <input type="checkbox" checked disabled={disabled} onChange={event => onToggle(id, event.target.checked)} aria-label="Collection unavailable" />
      <IconCollection size={14} /><span>Collection unavailable</span><small className="context-muted">deleted</small>
    </label>)}
  </>
}
