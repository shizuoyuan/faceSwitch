import { useEffect, useState } from 'react'
import { useApp } from '../store'
import { assetUrl } from '../api/client'

export function ExportPage() {
  const jobStatus = useApp((s) => s.jobStatus)
  const reset = useApp((s) => s.reset)
  const [videoUrl, setVideoUrl] = useState('')

  useEffect(() => {
    if (jobStatus?.outputPath) {
      assetUrl(`/api/file?path=${encodeURIComponent(jobStatus.outputPath)}`).then(setVideoUrl)
    }
  }, [jobStatus?.outputPath])

  if (!jobStatus?.outputPath) {
    return (
      <div className="export-page">
        <div className="error-box">没有可导出的成品,请重新处理。</div>
        <div className="wizard-actions">
          <button className="btn" onClick={reset}>返回首页</button>
        </div>
      </div>
    )
  }

  const fileName = jobStatus.outputPath.split(/[\\/]/).pop() ?? 'output.mp4'

  return (
    <div className="export-page">
      <div className="export-video-wrap panel">
        <video className="export-video" src={videoUrl} controls autoPlay />
      </div>
      <div className="export-info panel">
        <div className="export-done">🎉 处理完成</div>
        <div className="export-file" title={jobStatus.outputPath}>{fileName}</div>
        <div className="wizard-actions">
          <button
            className="btn btn-primary"
            onClick={() => window.electron.showInFolder(jobStatus.outputPath!)}
          >
            📂 打开所在文件夹
          </button>
          <button className="btn" onClick={reset}>
            ↺ 再处理一个
          </button>
        </div>
      </div>
    </div>
  )
}
