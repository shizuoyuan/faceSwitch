import { useEffect } from 'react'
import { useApp } from './store'
import { engineApi } from './api/client'
import { StepBar } from './components/StepBar'
import { ImportPage } from './pages/ImportPage'
import { ConfigurePage } from './pages/ConfigurePage'
import { ProcessPage } from './pages/ProcessPage'
import { ExportPage } from './pages/ExportPage'
import { SettingsModal } from './components/SettingsModal'

export default function App() {
  const step = useApp((s) => s.step)
  const settingsOpen = useApp((s) => s.settingsOpen)
  const setSettings = useApp((s) => s.setSettings)

  useEffect(() => {
    engineApi
      .getSettings()
      .then(setSettings)
      .catch(() => {})
  }, [setSettings])

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-title">
          <span className="app-logo">FS</span>
          <div>
            <div className="app-name">FaceSwitch</div>
            <div className="app-subtitle">AI 视频换脸 · 换背景 工作台</div>
          </div>
        </div>
        <button className="btn" onClick={() => useApp.getState().setSettingsOpen(true)}>
          ⚙ 设置
        </button>
      </header>

      <StepBar />

      <main className="app-content">
        {step === 'import' && <ImportPage />}
        {step === 'configure' && <ConfigurePage />}
        {step === 'process' && <ProcessPage />}
        {step === 'export' && <ExportPage />}
      </main>

      {settingsOpen && <SettingsModal />}
    </div>
  )
}
