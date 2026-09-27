import { useEffect, useRef, useState } from 'react'
import { useApp } from '../store'
import { assetUrl, engineApi } from '../api/client'
import type { JobStatus } from '../types'

const STAGE_LABELS: Record<string, string> = {
  prepare: '准备与抽帧',
  matting: '人物抠像',
  background: '背景处理',
  faceswap: 'AI 换脸',
  compose: '合成与编码',
  done: '完成'
}

const STAGE_ORDER = ['prepare', 'matting', 'background', 'faceswap', 'compose', 'done']

export function ProcessPage() {
  const probe = useApp((s) => s.probe)
  const faceSwap = useApp((s) => s.faceSwap)
  const background = useApp((s) => s.background)
  const options = useApp((s) => s.options)
  const jobId = useApp((s) => s.jobId)
  const jobStatus = useApp((s) => s.jobStatus)
  const setJob = useApp((s) => s.setJob)
  const setStep = useApp((s) => s.setStep)
  const [previewUrl, setPreviewUrl] = useState('')
  const [error, setError] = useState('')
  const startedRef = useRef(false)

  useEffect(() => {
    if (startedRef.current) return
    startedRef.current = true
    ;(async () => {
      if (!probe) return
      try {
        const { jobId: id } = await engineApi.createJob({
          videoPath: probe.path,
          faceSwap,
          background,
          options
        })
        setJob(id)
      } catch (e) {
        setError(String(e instanceof Error ? e.message : e))
      }
    })()
  }, [probe, faceSwap, background, options, setJob])

  useEffect(() => {
    if (!jobId) return
    const timer = setInterval(async () => {
      try {
        const st: JobStatus = await engineApi.jobStatus(jobId)
        setJob(jobId, st)
        if (st.state === 'running' && st.frameIndex % 5 < 1) {
          setPreviewUrl(
            await assetUrl(`/api/jobs/${jobId}/preview.jpg?t=${st.frameIndex}`)
          )
        }
        if (st.state === 'done' || st.state === 'failed' || st.state === 'canceled') {
          clearInterval(timer)
          if (st.state === 'done') setTimeout(() => setStep('export'), 600)
          if (st.state === 'failed') setError(st.error ?? '处理失败')
          if (st.state === 'canceled') setError('已取消')
        }
      } catch (e) {
        clearInterval(timer)
        setError(String(e instanceof Error ? e.message : e))
      }
    }, 500)
    return () => clearInterval(timer)
  }, [jobId, setJob, setStep])

  async function onCancel() {
    if (jobId) await engineApi.cancelJob(jobId).catch(() => {})
    setJob(null, null)
    setStep('configure')
  }

  const running = jobStatus?.state === 'running' || jobStatus?.state === 'queued'

  const currentIdx = jobStatus ? STAGE_ORDER.indexOf(jobStatus.stage) : -1

  return (
    <div className="process-page">
      <div className="process-main">
        <div className="preview-panel panel">
          <div className="preview-title">实时预览</div>
          {previewUrl ? (
            <img className="preview-img" src={previewUrl} alt="处理预览" />
          ) : (
            <div className="preview-empty">
              {running ? '正在初始化模型…(首次运行需下载权重)' : '等待任务开始'}
            </div>
          )}
        </div>

        <div className="progress-panel panel">
          <div className="progress-percent">{Math.round(jobStatus?.progress ?? 0)}%</div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${jobStatus?.progress ?? 0}%` }}
            />
          </div>
          <div className="progress-meta">
            帧进度:{jobStatus?.frameIndex ?? 0} / {jobStatus?.frameTotal ?? probe?.frameCount ?? 0}
            {jobStatus?.elapsedSec ? ` · 已用 ${Math.round(jobStatus.elapsedSec)} 秒` : ''}
          </div>

          <div className="stage-list">
            {STAGE_ORDER.map((key) => {
              const idx = STAGE_ORDER.indexOf(key)
              const active = idx === currentIdx
              const done = currentIdx > idx
              return (
                <div key={key} className={`stage ${active ? 'active' : ''} ${done ? 'done' : ''}`}>
                  <span className="stage-dot">{done ? '✓' : active ? '●' : '○'}</span>
                  <span>{STAGE_LABELS[key]}</span>
                  {active && <span className="stage-pulse">处理中…</span>}
                </div>
              )
            })}
          </div>

          {jobStatus?.message && <div className="hint">{jobStatus.message}</div>}
          {error && <div className="error-box">{error}</div>}

          <div className="wizard-actions">
            <button className="btn btn-danger" onClick={onCancel} disabled={!running}>
              {running ? '取消任务' : '返回配置'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
