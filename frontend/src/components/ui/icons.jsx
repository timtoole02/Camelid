/* Inline SVG icon set (Material-Symbols-inspired), currentColor-driven.
   Crisp at any size, no webfont dependency. */

function Svg({ size = 20, children, viewBox = '0 0 24 24', strokeIcon = false, ...rest }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox={viewBox}
      fill={strokeIcon ? 'none' : 'currentColor'}
      stroke={strokeIcon ? 'currentColor' : 'none'}
      strokeWidth={strokeIcon ? 1.8 : undefined}
      strokeLinecap={strokeIcon ? 'round' : undefined}
      strokeLinejoin={strokeIcon ? 'round' : undefined}
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  )
}

export const IconMenu = (p) => <Svg {...p}><path d="M3 6h18v2H3zM3 11h18v2H3zM3 16h18v2H3z" /></Svg>
export const IconSidebar = (p) => (
  <Svg {...p} strokeIcon>
    <rect x="3" y="4" width="18" height="16" rx="2.5" />
    <path d="M9 4v16" />
  </Svg>
)
export const IconNewChat = (p) => (
  <Svg {...p}><path d="M5 4h7v2H5v13h13v-7h2v7c0 1.1-.9 2-2 2H5a2 2 0 0 1-2-2V6c0-1.1.9-2 2-2zm14.06-.94 1.88 1.88a1 1 0 0 1 0 1.41l-7.3 7.3-2.83.59.59-2.83 7.3-7.3a1 1 0 0 1 1.41 0z" /></Svg>
)
export const IconChat = (p) => <Svg {...p}><path d="M4 4h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H8l-4 4V6a2 2 0 0 1 2-2zm3 6v2h2v-2H7zm4 0v2h2v-2h-2zm4 0v2h2v-2h-2z" /></Svg>
export const IconHistory = (p) => <Svg {...p}><path d="M13 3a9 9 0 1 0 8.94 10H19.9A7 7 0 1 1 13 5c1.93 0 3.68.78 4.95 2.05L15 10h7V3l-2.6 2.6A8.97 8.97 0 0 0 13 3zm-1 5v5l4.25 2.52.77-1.28-3.52-2.09V8H12z" /></Svg>
export const IconModels = (p) => <Svg {...p}><path d="M12 2 2 7l10 5 10-5-10-5zm0 7.2L5.3 7 12 4.3 18.7 7 12 9.2zM2 12l10 5 10-5-2.3-1.15L12 14.5 4.3 10.85 2 12zm0 5 10 5 10-5-2.3-1.15L12 19.5 4.3 15.85 2 17z" /></Svg>
export const IconSystem = (p) => <Svg {...p}><path d="M3 17v2h6v-2H3zM3 5v2h10V5H3zm10 16v-2h8v-2h-8v-2h-2v6h2zM7 9v2H3v2h4v2h2V9H7zm14 4v-2H11v2h10zm-6-4h2V7h4V5h-4V3h-2v6z" /></Svg>
export const IconApi = (p) => <Svg {...p}><path d="M9.4 16.6 4.8 12l4.6-4.6L8 6l-6 6 6 6 1.4-1.4zm5.2 0L19.2 12l-4.6-4.6L16 6l6 6-6 6-1.4-1.4z" /></Svg>
export const IconAnalytics = (p) => <Svg {...p}><path d="M4 9h4v11H4V9zm6-5h4v16h-4V4zm6 8h4v8h-4v-8z" /></Svg>
export const IconMemory = (p) => <Svg {...p}><path d="M17 3H7a2 2 0 0 0-2 2v16l7-3 7 3V5a2 2 0 0 0-2-2z" /></Svg>
export const IconImage = (p) => <Svg {...p} strokeIcon><rect x="3" y="4" width="18" height="16" rx="2.5" /><circle cx="8.5" cy="9" r="1.5" /><path d="m5 17 4.5-4.5 3.2 3.2 2.1-2.1L19 17" /></Svg>
export const IconSend = (p) => <Svg {...p}><path d="M12 4 6 10l1.4 1.4L11 7.8V20h2V7.8l3.6 3.6L18 10l-6-6z" /></Svg>
export const IconStop = (p) => <Svg {...p}><rect x="6" y="6" width="12" height="12" rx="2.5" /></Svg>
export const IconCopy = (p) => (
  <Svg {...p} strokeIcon>
    <rect x="4.5" y="8.5" width="11" height="11" rx="2.5" />
    <path d="M8.5 8.5V7a2.5 2.5 0 0 1 2.5-2.5h6A2.5 2.5 0 0 1 19.5 7v6a2.5 2.5 0 0 1-2.5 2.5H15.5" />
  </Svg>
)
export const IconCheck = (p) => (
  <Svg {...p} strokeIcon>
    <polyline points="20 6 9 17 4 12" />
  </Svg>
)
export const IconSun = (p) => <Svg {...p}><path d="M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10zm0-5 2 3h-4l2-3zm0 20-2-3h4l-2 3zM2 12l3-2v4l-3-2zm20 0-3 2v-4l3 2zM4.5 4.5 7 7 5.5 8.5 3 6l1.5-1.5zm15 15L17 17l1.5-1.5L21 18l-1.5 1.5zm0-15L18 6l-1.5-1.5L18 3l1.5 1.5zM4.5 19.5 6 17l1.5 1.5L6 21l-1.5-1.5z" /></Svg>
export const IconMoon = (p) => <Svg {...p}><path d="M12 3a9 9 0 1 0 9 9c0-.46-.04-.92-.1-1.36A6 6 0 0 1 11.36 3.1 9.4 9.4 0 0 0 12 3z" /></Svg>
export const IconMonitor = (p) => <Svg {...p}><path d="M21 3H3a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h7v2H7v2h10v-2h-3v-2h7a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2zm0 13H3V5h18v11z" /></Svg>
export const IconSearch = (p) => <Svg {...p} strokeIcon><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></Svg>
export const IconTrash = (p) => <Svg {...p}><path d="M6 7h12l-1 14H7L6 7zm3-3h6l1 2h4v2H4V6h4l1-2zm1 6v8h2v-8h-2zm4 0v8h2v-8h-2z" /></Svg>
export const IconClose = (p) => <Svg {...p}><path d="m18.3 5.7-1.4-1.4L12 9.2 7.1 4.3 5.7 5.7 10.6 10.6 5.7 15.5l1.4 1.4L12 12l4.9 4.9 1.4-1.4-4.9-4.9 4.9-4.9z" /></Svg>
export const IconFile = (p) => (
  <Svg {...p} strokeIcon>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
  </Svg>
)
export const IconScale = (p) => (
  <Svg {...p} strokeIcon>
    <path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z" />
    <path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z" />
    <path d="M7 21h10" />
    <path d="M12 3v18" />
    <path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2" />
  </Svg>
)
export const IconChevronDown = (p) => <Svg {...p} strokeIcon><path d="m6 9 6 6 6-6" /></Svg>
export const IconChevronRight = (p) => <Svg {...p} strokeIcon><path d="m9 6 6 6-6 6" /></Svg>
export const IconPlus = (p) => <Svg {...p}><path d="M11 5h2v6h6v2h-6v6h-2v-6H5v-2h6V5z" /></Svg>
export const IconDownload = (p) => <Svg {...p}><path d="M12 3v10.2l3.6-3.6L17 11l-5 5-5-5 1.4-1.4L11 13.2V3h2zM5 19h14v2H5z" /></Svg>
export const IconExternal = (p) => <Svg {...p}><path d="M14 3h7v7h-2V6.4l-9.3 9.3-1.4-1.4L17.6 5H14V3zM5 5h5v2H5v12h12v-5h2v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z" /></Svg>
export const IconRefresh = (p) => (
  <Svg {...p} strokeIcon>
    <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
    <path d="M3 3v5h5" />
    <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
    <path d="M16 21h5v-5" />
  </Svg>
)
export const IconDots = (p) => <Svg {...p}><path d="M12 8a2 2 0 1 0 0-4 2 2 0 0 0 0 4zm0 6a2 2 0 1 0 0-4 2 2 0 0 0 0 4zm0 6a2 2 0 1 0 0-4 2 2 0 0 0 0 4z" /></Svg>
export const IconPin = (p) => <Svg {...p}><path d="M16 4v2l-1 1v5l2 2v2h-4v5l-1 1-1-1v-5H7v-2l2-2V7L8 6V4h8z" /></Svg>
export const IconEdit = (p) => (
  <Svg {...p} strokeIcon>
    <path d="M17.5 3.5a2.121 2.121 0 0 1 3 3L7 20l-4 1 1-4L17.5 3.5z" />
    <path d="m15 6 3 3" />
  </Svg>
)
export const IconWarning = (p) => <Svg {...p}><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z" /></Svg>
export const IconInfo = (p) => <Svg {...p}><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" /></Svg>
export const IconCheckCircle = (p) => <Svg {...p}><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm-2 15-5-5 1.4-1.4L10 14.2l7.6-7.6L19 8l-9 9z" /></Svg>
export const IconError = (p) => <Svg {...p}><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" /></Svg>
export const IconBolt = (p) => <Svg {...p}><path d="M11 21 13 13H9l4-10h2l-2 6h4l-4 12h-2z" /></Svg>
export const IconReceipt = (p) => <Svg {...p}><path d="M5 2v20l2-1.5L9 22l2-1.5L13 22l2-1.5L17 22l2-1.5V2l-2 1.5L15 2l-2 1.5L11 2 9 3.5 7 2 5 3.5V2zm3 5h8v2H8V7zm0 4h8v2H8v-2zm0 4h5v2H8v-2z" /></Svg>
export const IconThinking = (p) => <Svg {...p} strokeIcon><path d="M9.5 18.5h5M10.5 21h3M12 3a6 6 0 0 0-3.6 10.8c.4.3.6.7.6 1.2v.5h6v-.5c0-.5.2-.9.6-1.2A6 6 0 0 0 12 3z" /></Svg>
export const IconChart = (p) => <Svg {...p}><path d="M3 3h2v18H3V3zm16 8h2v10h-2V11zm-4-6h2v16h-2V5zM7 13h2v8H7v-8z" /></Svg>
export const IconSettings = (p) => <Svg {...p}><path d="M19.14 12.94a7.5 7.5 0 0 0 .05-1.88l2.03-1.58a.5.5 0 0 0 .12-.64l-1.92-3.32a.5.5 0 0 0-.6-.22l-2.39.96a7 7 0 0 0-1.62-.94l-.36-2.54a.5.5 0 0 0-.5-.42h-3.84a.5.5 0 0 0-.5.42l-.36 2.54c-.59.24-1.13.56-1.62.94l-2.39-.96a.5.5 0 0 0-.6.22L2.32 8.84a.5.5 0 0 0 .12.64l2.03 1.58a7.5 7.5 0 0 0 0 1.88l-2.03 1.58a.5.5 0 0 0-.12.64l1.92 3.32c.13.22.39.31.6.22l2.39-.96c.49.38 1.03.7 1.62.94l.36 2.54a.5.5 0 0 0 .5.42h3.84a.5.5 0 0 0 .5-.42l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.09.47 0 .6-.22l1.92-3.32a.5.5 0 0 0-.12-.64l-2.03-1.58zM12 15.5A3.5 3.5 0 1 1 12 8.5a3.5 3.5 0 0 1 0 7z" /></Svg>
export const IconPlay = (p) => <Svg {...p}><path d="M8 5v14l11-7z" /></Svg>
export const IconServer = (p) => <Svg {...p}><path d="M4 4h16a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1zm0 9h16a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1zm3 2.5a1 1 0 1 0 0 2 1 1 0 0 0 0-2zM7 6.5a1 1 0 1 0 0 2 1 1 0 0 0 0-2z" /></Svg>
/* device types */
export const IconMac = (p) => <Svg {...p}><path d="M17.5 12.6c0-2 1.6-2.8 1.7-2.9-.9-1.3-2.4-1.5-2.9-1.5-1.2-.13-2.4.72-3 .72-.6 0-1.6-.7-2.6-.68-1.3.02-2.6.78-3.2 2-1.4 2.4-.35 5.9 1 7.9.66.9 1.45 2 2.5 1.96 1-.04 1.4-.65 2.6-.65s1.55.65 2.6.63c1.07-.02 1.75-.98 2.4-1.9.5-.74.74-1.46.76-1.5-.02-.01-2.04-.8-2.06-3.6zM15.6 6.7c.55-.66.92-1.6.82-2.5-.8.03-1.76.52-2.33 1.18-.5.58-.95 1.5-.83 2.4.9.07 1.8-.45 2.34-1.08z" /></Svg>
export const IconWindows = (p) => <Svg {...p}><path d="M3 5.5 10.5 4.4v7.1H3V5.5zm0 13L10.5 19.6v-7H3v6zM11.5 4.25 21 3v8.5h-9.5V4.25zm0 8.25H21V21l-9.5-1.3V12.5z" /></Svg>
export const IconLinux = (p) => <Svg {...p}><path d="M12 2c-2.2 0-3.5 1.9-3.5 4.2 0 1.4.4 2.2.4 3.3 0 .9-.7 1.6-1.6 3-1 1.6-2 3-2 4.7 0 1.4.9 2.1 2 2.4.4.9 1.2 1.9 2.4 2 .9.1 1.6-.3 2.3-.3s1.4.4 2.3.3c1.2-.1 2-1.1 2.4-2 1.1-.3 2-1 2-2.4 0-1.7-1-3.1-2-4.7-.9-1.4-1.6-2.1-1.6-3 0-1.1.4-1.9.4-3.3C15.5 3.9 14.2 2 12 2zm-1.5 4.1c.5 0 .9.5.9 1.1s-.4 1.1-.9 1.1-.9-.5-.9-1.1.4-1.1.9-1.1zm3 0c.5 0 .9.5.9 1.1s-.4 1.1-.9 1.1-.9-.5-.9-1.1.4-1.1.9-1.1zM12 9.4c.9 0 2 .6 2 1.2 0 .4-.6.7-1.1 1-.4.2-.7.5-.9.5s-.5-.3-.9-.5c-.5-.3-1.1-.6-1.1-1 0-.6 1.1-1.2 2-1.2z" /></Svg>
export const IconRaspberryPi = (p) => <Svg {...p} strokeIcon><rect x="3" y="6" width="18" height="12" rx="1.5" /><path d="M6 6V4M9 6V4M12 6V4M15 6V4M18 6V4" /><rect x="6.5" y="9.5" width="4" height="4" rx="0.5" /><path d="M14 10h4M14 12.5h4M14 15h4" /></Svg>
export const IconNetwork = (p) => <Svg {...p} strokeIcon><circle cx="12" cy="5" r="2.4" /><circle cx="5" cy="18" r="2.4" /><circle cx="19" cy="18" r="2.4" /><path d="M10.6 6.9 6.4 16M13.4 6.9 17.6 16M7.4 18h9.2" /></Svg>
/* observatory: an orbit ring around a core */
export const IconObservatory = (p) => <Svg {...p} strokeIcon><circle cx="12" cy="12" r="2.6" /><ellipse cx="12" cy="12" rx="9.5" ry="3.8" transform="rotate(-24 12 12)" /><circle cx="18.6" cy="8.3" r="1" fill="currentColor" stroke="none" /></Svg>
/* canvas controls */
export const IconZoomIn = (p) => <Svg {...p} strokeIcon><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5M11 8v6M8 11h6" /></Svg>
export const IconZoomOut = (p) => <Svg {...p} strokeIcon><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5M8 11h6" /></Svg>
export const IconFit = (p) => <Svg {...p} strokeIcon><path d="M4 9V5a1 1 0 0 1 1-1h4M20 9V5a1 1 0 0 0-1-1h-4M4 15v4a1 1 0 0 0 1 1h4M20 15v4a1 1 0 0 1-1 1h-4" /></Svg>
export const IconGrid = (p) => <Svg {...p}><path d="M3 3h7v7H3V3zm11 0h7v7h-7V3zM3 14h7v7H3v-7zm11 0h7v7h-7v-7z" /></Svg>
export const IconCpu = (p) => <Svg {...p} strokeIcon><rect x="7" y="7" width="10" height="10" rx="1.5" /><path d="M9.5 3v2M14.5 3v2M9.5 19v2M14.5 19v2M3 9.5h2M3 14.5h2M19 9.5h2M19 14.5h2" /></Svg>
export const IconWifi = (p) => <Svg {...p} strokeIcon><path d="M5 12.5a10 10 0 0 1 14 0M8 15.5a6 6 0 0 1 8 0" /><circle cx="12" cy="18.5" r="1" fill="currentColor" stroke="none" /></Svg>
export const IconLink = (p) => <Svg {...p} strokeIcon><path d="M9 12h6M8.5 8H7a4 4 0 0 0 0 8h1.5M15.5 8H17a4 4 0 0 1 0 8h-1.5" /></Svg>
