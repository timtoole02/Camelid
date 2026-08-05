import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
/* Self-hosted fonts (Fontsource) — bundled locally; the app must render fully
   offline, so no third-party font CDN imports anywhere. */
import '@fontsource-variable/inter'
import '@fontsource-variable/space-grotesk'
import '@fontsource/ibm-plex-mono/400.css'
import '@fontsource/ibm-plex-mono/500.css'
import '@fontsource/ibm-plex-mono/600.css'
import './styles.css'
import { installApiAuthFetch } from './lib/apiAuth'

installApiAuthFetch()

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
