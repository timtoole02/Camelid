import { MOSAIC_STEPS } from '../hooks/useRetroTransition'

/** Sample one screen pixel per tile, then expand it into a solid square.
 * The filter uses the live view, including text and editor content; no captures
 * or copies of chat data are created. Keep defs mounted only when opted in.
 */
export function RetroTransitionFilters({ prefix }) {
  return (
    <svg className="retro-transition-filters" width="0" height="0" aria-hidden="true" focusable="false">
      <defs>
        <filter id={`${prefix}-pixel`} x="0" y="0" width="100%" height="100%" colorInterpolationFilters="sRGB">
          <feFlood floodColor="white" />
          <feComposite id={`${prefix}-sample`} in="SourceGraphic" operator="in" x="0" y="0" width="1" height="1" />
          <feTile x="0" y="0" width="100%" height="100%" />
        </filter>
        {MOSAIC_STEPS.map((size) => (
          <filter key={size} id={`${prefix}-${size}`} x="0" y="0" width="100%" height="100%" colorInterpolationFilters="sRGB">
            <feFlood x={size / 2} y={size / 2} width="1" height="1" />
            <feComposite operator="in" x="0" y="0" width={size} height={size} />
            <feTile result="grid" />
            <feComposite in="SourceGraphic" in2="grid" operator="in" />
            <feMorphology operator="dilate" radius={size / 2} />
          </filter>
        ))}
      </defs>
    </svg>
  )
}
