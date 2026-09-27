import { useEffect, useState } from 'react'
import { useApp } from '../store'
import { assetUrl } from '../api/client'

function FaceThumb({ path }: { path: string }) {
  const [url, setUrl] = useState('')
  useEffect(() => {
    assetUrl(`/api/thumb?path=${encodeURIComponent(path)}`).then(setUrl)
  }, [path])
  if (!url) return <div className="face-thumb placeholder" />
  return <img className="face-thumb" src={url} alt="源脸" />
}

const BG_MODES = [
  { key: 'keep', label: '保留原背景', desc: '只换脸,不动背景' },
  { key: 'image', label: '图片背景', desc: '上传一张图作为新背景' },
  { key: 'video', label: '视频背景', desc: '上传一段视频循环作为新背景' },
  { key: 'ai', label: 'AI 生成背景', desc: '调用配置的 AI 服务生成背景' }
] as const

export function ConfigurePage() {
  const probe = useApp((s) => s.probe)
  const faceSwap = useApp((s) => s.faceSwap)
  const setFaceSwap = useApp((s) => s.setFaceSwap)
  const background = useApp((s) => s.background)
  const setBackground = useApp((s) => s.setBackground)
  const options = useApp((s) => s.options)
  const setOptions = useApp((s) => s.setOptions)
  const settings = useApp((s) => s.settings)
  const setStep = useApp((s) => s.setStep)

  async function pickFaces() {
    const paths = await window.electron.pickImages()
    if (paths.length) setFaceSwap({ sourceImages: paths })
  }

  async function pickBgImage() {
    const paths = await window.electron.pickImages()
    if (paths.length) setBackground({ imagePath: paths[0], mode: 'image' })
  }

  async function pickBgVideo() {
    const path = await window.electron.pickVideo()
    if (path) setBackground({ videoPath: path, mode: 'video' })
  }

  const canStart =
    (!faceSwap.enabled || faceSwap.sourceImages.length > 0) &&
    (background.mode !== 'image' || background.imagePath) &&
    (background.mode !== 'video' || background.videoPath) &&
    (background.mode !== 'ai' || (background.providerId && background.prompt?.trim()))

  return (
    <div className="configure-page">
      <div className="config-grid">
        {/* 换脸配置 */}
        <section className="panel config-section">
          <h3>👤 换脸</h3>
          <label className="toggle-row">
            <input
              type="checkbox"
              checked={faceSwap.enabled}
              onChange={(e) => setFaceSwap({ enabled: e.target.checked })}
            />
            <span>启用换脸(在本地 GPU 处理,人脸数据不出本机)</span>
          </label>

          {faceSwap.enabled && (
            <>
              <label className="label">源脸图片(清晰正脸效果最好,可多选对应多张脸)</label>
              <div className="face-list">
                {faceSwap.sourceImages.map((p) => (
                  <div key={p} className="face-chip" title={p}>
                    <FaceThumb path={p} />
                    <span>{p.split(/[\\/]/).pop()}</span>
                    <button
                      className="face-remove"
                      onClick={() =>
                        setFaceSwap({
                          sourceImages: faceSwap.sourceImages.filter((x) => x !== p)
                        })
                      }
                    >
                      ×
                    </button>
                  </div>
                ))}
                <button className="btn" onClick={pickFaces}>
                  + 添加源脸
                </button>
              </div>

              <label className="label">
                面部修复强度:{faceSwap.restoreStrength}%(GFPGAN,越高越清晰但可能不像)
              </label>
              <input
                type="range"
                min={0}
                max={100}
                step={10}
                value={faceSwap.restoreStrength}
                onChange={(e) => setFaceSwap({ restoreStrength: Number(e.target.value) })}
                style={{ width: '100%' }}
              />
            </>
          )}
        </section>

        {/* 背景配置 */}
        <section className="panel config-section">
          <h3>🖼 背景</h3>
          <div className="bg-modes">
            {BG_MODES.map((m) => (
              <button
                key={m.key}
                className={`bg-mode ${background.mode === m.key ? 'selected' : ''}`}
                onClick={() => setBackground({ mode: m.key })}
              >
                <b>{m.label}</b>
                <span>{m.desc}</span>
              </button>
            ))}
          </div>

          {background.mode === 'image' && (
            <div className="bg-detail">
              <button className="btn" onClick={pickBgImage}>
                选择背景图片
              </button>
              {background.imagePath && <span className="chosen">{background.imagePath.split(/[\\/]/).pop()}</span>}
            </div>
          )}

          {background.mode === 'video' && (
            <div className="bg-detail">
              <button className="btn" onClick={pickBgVideo}>
                选择背景视频
              </button>
              {background.videoPath && <span className="chosen">{background.videoPath.split(/[\\/]/).pop()}</span>}
            </div>
          )}

          {background.mode === 'ai' && (
            <div className="bg-detail">
              {settings?.providers.length ? (
                <>
                  <label className="label">AI 服务</label>
                  <select
                    className="select"
                    value={background.providerId ?? ''}
                    onChange={(e) => setBackground({ providerId: e.target.value })}
                  >
                    <option value="">— 选择服务 —</option>
                    {settings.providers.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                </>
              ) : (
                <div className="error-box">
                  尚未配置 AI 服务。请到「设置 → AI 服务」添加你的本地 AI 或云端 API 地址。
                </div>
              )}
              <label className="label">背景描述(Prompt)</label>
              <textarea
                className="textarea"
                rows={3}
                placeholder="例如:夕阳下的雪山湖泊,电影感光线"
                value={background.prompt ?? ''}
                onChange={(e) => setBackground({ prompt: e.target.value })}
              />
            </div>
          )}

          {background.mode !== 'keep' && (
            <>
              <label className="label">背景适配方式</label>
              <select
                className="select"
                value={background.fit}
                onChange={(e) => setBackground({ fit: e.target.value as 'cover' | 'contain' | 'stretch' })}
              >
                <option value="cover">铺满裁切(cover)</option>
                <option value="contain">完整显示(contain)</option>
                <option value="stretch">拉伸(stretch)</option>
              </select>
            </>
          )}
        </section>
      </div>

      {/* 高级选项 */}
      <section className="panel config-section">
        <h3>⚙ 输出选项</h3>
        <div className="options-row">
          <label className="toggle-row">
            <input
              type="checkbox"
              checked={options.colorMatch}
              onChange={(e) => setOptions({ colorMatch: e.target.checked })}
            />
            <span>色彩匹配(前景与新背景光照融合,推荐开启)</span>
          </label>
          <div>
            <label className="label">输出质量</label>
            <select
              className="select"
              value={options.quality}
              onChange={(e) => setOptions({ quality: e.target.value as 'high' | 'balanced' | 'fast' })}
            >
              <option value="high">高质量(CRF 16,文件较大)</option>
              <option value="balanced">均衡(推荐)</option>
              <option value="fast">快速(CRF 26)</option>
            </select>
          </div>
        </div>
      </section>

      <div className="wizard-actions">
        <button className="btn" onClick={() => setStep('import')}>
          ← 上一步
        </button>
        <button className="btn btn-primary" disabled={!canStart} onClick={() => setStep('process')}>
          开始 AI 处理 →
        </button>
      </div>
      {probe && !canStart && (
        <div className="hint">请补全上方必填项后开始处理</div>
      )}
    </div>
  )
}
