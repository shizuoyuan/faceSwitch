import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { installBrowserMock } from './dev-mock'
import './styles.css'

installBrowserMock()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
