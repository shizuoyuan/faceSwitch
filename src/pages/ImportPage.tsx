import { useState } from 'react'
import { useApp } from '../store'
import { engineApi } from '../api/client'
import type { ProbeResult } from '../types'

export function ImportPage() {
  const setStep = useApp((s) => s.setStep)
  const setProbe = useApp((s) => s.setProbe)
  const probe = useApp((s) => s.probe)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dragOver, setDragOver] = useState(false)

  async function loadVideo(path: string) {
    setError('')
    setLoading(true)
    try {
      const result = await engineApi.probe(path)
      setProbe(result)
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
    } finally {
      setLoading(false)
    }
  }

  async function onPick() {
    const path = await window.electron.pickVideo()
    if (path) await loadVideo(path)
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (!file) return
    const path = window.electron.pathForFile(file)
    if (path) void loadVideo(path)
  }

  const fmt = (sec: number) => {
    const m = Math.floor(sec / 60)
    const s = Math.round(sec % 60)
    return `${m}分${s.toString().padStart(2, '0')}秒`
  }

  return (
    <div className="import-page">
      <div
        className={`drop-zone ${dragOver ? 'over' : ''} ${probe ? 'filled' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
      >
        {!probe && !loading && (
          <div className="drop-empty">
            <div className="drop-icon">🎬</div>
            <div className="drop-title">拖拽视频到此处</div>
            <div className="drop-hint">或点击下方按钮选择文件 · 支持 MP4 / MOV / MKV / AVI / WEBM</div>
            <button className="btn btn-primary" onClick={onPick}>
              选择视频文件
            </button>
            <div className="drop-note">建议:1080p 及以下、2 分钟以内的视频处理最快</div>
          </div>
        )}
        {loading && <div className="drop-loading">正在读取视频信息…</div>}
        {probe && (
          <div className="probe-card">
            <img src={probe.thumbnail} alt="视频缩略图" className="probe-thumb" />
            <div className="probe-meta">
              <div className="probe-name">{probe.fileName}</div>
              <div className="probe-grid">
                <span>分辨率</span><b>{probe.width} × {probe.height}</b>
                <span>帧率</span><b>{probe.fps.toFixed(1)} fps</b>
                <span>时长</span><b>{fmt(probe.durationSec)}</b>
                <span>总帧数</span><b>{probe.frameCount}</b>
                <span>音频</span><b>{probe.hasAudio ? '有(将保留原声)' : '无'}</b>
              </div>
            </div>
          </div>
        )}
      </div>

      {error && <div className="error-box">{error}</div>}

      <div className="wizard-actions">
        <button className="btn" onClick={onPick}>
          {probe ? '换个视频' : '选择文件'}
        </button>
        <button
          className="btn btn-primary"
          disabled={!probe}
          onClick={() => setStep('configure')}
        >
          下一步:配置任务 →
        </button>
      </div>
    </div>
  )
}
