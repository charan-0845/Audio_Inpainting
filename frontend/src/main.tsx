import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { ThemeProvider } from '@/features/backend-connection/ThemeContext'
import { BackendProvider } from '@/features/backend-connection/BackendContext'
import { App } from '@/app/App'
import '@/styles/globals.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <ThemeProvider>
        <BackendProvider>
          <App />
        </BackendProvider>
      </ThemeProvider>
    </BrowserRouter>
  </React.StrictMode>
)