/* Button — variants: primary | tonal | ghost | danger | outline; sizes: sm | md | lg */
export function Button({
  variant = 'tonal',
  size = 'md',
  icon = null,
  iconRight = null,
  block = false,
  loading = false,
  loadingVariant = 'spinner',
  loadingLabel = '',
  disabled = false,
  className = '',
  children,
  type = 'button',
  'aria-label': ariaLabel,
  ...rest
}) {
  const isBarLoading = loading && loadingVariant === 'bar'
  const isSpinnerLoading = loading && loadingVariant !== 'bar'

  const classes = [
    'cx-btn',
    `cx-btn--${variant}`,
    `cx-btn--${size}`,
    block ? 'cx-btn--block' : '',
    isSpinnerLoading ? 'is-loading' : '',
    isBarLoading ? 'is-loading-bar' : '',
    !children && !isBarLoading ? 'cx-btn--icon-only' : '',
    className,
  ].filter(Boolean).join(' ')

  const resolvedLabel = loadingLabel || (typeof children === 'string' ? children : 'Loading…')

  // Children of a button are presentational in ARIA, so nothing rendered inside
  // the bar reaches assistive tech — the loading state has to live on the
  // button's own accessible name. Disabling also blurs the button, so views that
  // need this announced should additionally render an sr-only live region (see
  // DownloadedModelsView).
  const label = isBarLoading ? resolvedLabel : ariaLabel

  return (
    <button
      type={type}
      className={classes}
      aria-busy={loading || undefined}
      aria-label={label}
      disabled={disabled || loading}
      {...rest}
    >
      {isSpinnerLoading && <span className="cx-btn__spinner" aria-hidden="true" />}
      {isBarLoading && (
        <span className="cx-btn__loadbar" aria-hidden="true">
          <span className="cx-btn__loadbar-track">
            <span className="cx-btn__loadbar-fill" />
            <span className="cx-btn__loadbar-shimmer" />
            <span className="cx-btn__loadbar-line" />
          </span>
          <span className="cx-btn__loadbar-content">
            <span className="cx-btn__loadbar-dot" />
            <span className="cx-btn__loadbar-label">{resolvedLabel}</span>
          </span>
        </span>
      )}
      {!loading && icon && <span className="cx-btn__icon">{icon}</span>}
      {!isBarLoading && children && <span className="cx-btn__label">{children}</span>}
      {!loading && iconRight && <span className="cx-btn__icon">{iconRight}</span>}
    </button>
  )
}

export default Button
