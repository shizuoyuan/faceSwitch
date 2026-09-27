import { useState } from 'react'
import { useApp } from '../store'
import { engineApi } from '../api/client'
import type { Provider, ProviderType, Settings } from '../types'

const PRESET_TEMPLATES: Record<
  ProviderType,
  Omit<Provider, 'id' | 'name'>
> = {
  'sd-webui': {
    type: 'sd-webui',
    baseUrl: 'http://127.0.0.1:7860/sdapi/v1/txt2img',
    headersJson: '{}',
    method: 'POST',
    bodyTemplate:
      '{"prompt": "{prompt}", "width": {width}, "height": {height}, "steps": 25, "cfg_scale": 6.5}',
    responsePath: 'images.0',
    responseType: 'base64'
  },
  'openai-images': {
    type: 'openai-images',
    baseUrl: 'http://127.0.0.1:8000/v1/images/generations',
    headersJson: '{\n  "Authorization": "Bearer YOUR_KEY"\n}',
    method: 'POST',
    bodyTemplate:
      '{"model": "flux-schnell", "prompt": "{prompt}", "size": "{width}x{height}", "response_format": "b64_json"}',
    responsePath: 'data.0.b64_json',
    responseType: 'base64'
  },
  custom: {
    type: 'custom',
    baseUrl: 'http://127.0.0.1:8000/generate',
    headersJson: '{}',
    method: 'POST',
    bodyTemplate:
      '{"prompt": "{prompt}", "width": {width}, "height": {height}, "reference": "{first_frame_b64}"}',
    responsePath: 'image_url',
    responseType: 'url'
  }
}

const emptyProvider = (): Provider => ({
  id: crypto.randomUUID(),
  name: '新建 AI 服务',
  ...JSON.parse(JSON.stringify(PRESET_TEMPLATES.custom))
})

export function SettingsModal() {
  const settings = useApp((s) => s.settings)
  const setSettingsState = useApp((s) => s.setSettings)
  const setSettingsOpen = useApp((s) => s.setSettingsOpen)
  const [draft, setDraft] = useState<Settings | null>(settings)
  const [editing, setEditing] = useState<Provider | null>(null)
  const [saveMsg, setSaveMsg] = useState('')

  if (!draft) return null
  const d: Settings = draft

  async function save() {
    if (!draft) return
    const saved = await engineApi.saveSettings(draft)
    setSettingsState(saved)
    setSettingsOpen(false)
  }

  function addProvider() {
    const p = emptyProvider()
    setDraft({ ...d, providers: [...d.providers, p] })
    setEditing(p)
  }

  function updateProvider(patch: Partial<Provider>) {
    if (!editing) return
    const merged = { ...editing, ...patch }
    setEditing(merged)
    setDraft({
      ...d,
      providers: d.providers.map((p) => (p.id === merged.id ? merged : p))
    })
  }

  function removeProvider(id: string) {
    setDraft({ ...d, providers: d.providers.filter((p) => p.id !== id) })
    if (editing?.id === id) setEditing(null)
  }

  function applyPreset(type: ProviderType) {
    if (!editing) return
    const preset = JSON.parse(JSON.stringify(PRESET_TEMPLATES[type]))
    updateProvider({ ...preset, id: editing.id, name: editing.name })
  }

  return (
    <div className="modal-overlay" onClick={() => setSettingsOpen(false)}>
      <div className="modal panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2>设置</h2>
          <button className="btn" onClick={() => setSettingsOpen(false)}>✕</button>
        </div>

        <div className="modal-body">
          <h3>🤖 AI 服务(用于生成背景)</h3>
          {d.providers.length === 0 && (
            <div className="hint">尚未配置。使用「AI 生成背景」前需至少添加一个服务;本地背景/换脸功能不需要。</div>
          )}
          <div className="provider-list">
            {d.providers.map((p) => (
              <div key={p.id} className="provider-row">
                <b>{p.name}</b>
                <span className="provider-url">{p.baseUrl}</span>
                <button className="btn" onClick={() => setEditing(p)}>编辑</button>
                <button className="btn btn-danger" onClick={() => removeProvider(p.id)}>删除</button>
              </div>
            ))}
          </div>
          <button className="btn" onClick={addProvider}>+ 添加 AI 服务</button>

          {editing && (
            <div className="provider-form">
              <div className="form-row">
                <div>
                  <label className="label">名称</label>
                  <input
                    className="input"
                    value={editing.name}
                    onChange={(e) => updateProvider({ name: e.target.value })}
                  />
                </div>
                <div>
                  <label className="label">接口类型(点此套用模板)</label>
                  <select
                    className="select"
                    value={editing.type}
                    onChange={(e) => applyPreset(e.target.value as ProviderType)}
                  >
                    <option value="custom">自定义 HTTP</option>
                    <option value="sd-webui">SD WebUI(本地)</option>
                    <option value="openai-images">OpenAI 兼容 images 接口</option>
                  </select>
                </div>
              </div>

              <label className="label">请求地址(URL)</label>
              <input
                className="input"
                value={editing.baseUrl}
                onChange={(e) => updateProvider({ baseUrl: e.target.value })}
                placeholder="http://127.0.0.1:7860/..."
              />

              <div className="form-row">
                <div>
                  <label className="label">方法</label>
                  <select
                    className="select"
                    value={editing.method}
                    onChange={(e) => updateProvider({ method: e.target.value as 'POST' | 'GET' })}
                  >
                    <option value="POST">POST</option>
                    <option value="GET">GET</option>
                  </select>
                </div>
                <div>
                  <label className="label">响应类型</label>
                  <select
                    className="select"
                    value={editing.responseType}
                    onChange={(e) => updateProvider({ responseType: e.target.value as 'url' | 'base64' })}
                  >
                    <option value="base64">Base64 图片</option>
                    <option value="url">图片 URL</option>
                  </select>
                </div>
              </div>

              <label className="label">请求头(JSON,含鉴权)</label>
              <textarea
                className="textarea mono"
                rows={3}
                value={editing.headersJson}
                onChange={(e) => updateProvider({ headersJson: e.target.value })}
              />

              <label className="label">
                请求体模板(占位符:{'{prompt}'} {'{width}'} {'{height}'} {'{first_frame_b64}'})
              </label>
              <textarea
                className="textarea mono"
                rows={5}
                value={editing.bodyTemplate}
                onChange={(e) => updateProvider({ bodyTemplate: e.target.value })}
              />

              <label className="label">响应取值路径(如 images.0 或 data.0.b64_json)</label>
              <input
                className="input"
                value={editing.responsePath}
                onChange={(e) => updateProvider({ responsePath: e.target.value })}
              />
            </div>
          )}

          <h3 style={{ marginTop: 24 }}>🖥 运行环境</h3>
          <div className="form-row">
            <div>
              <label className="label">计算设备</label>
              <select
                className="select"
                value={draft.device}
                onChange={(e) => setDraft({ ...d, device: e.target.value as 'gpu' | 'cpu' })}
              >
                <option value="gpu">GPU(推荐)</option>
                <option value="cpu">CPU(很慢)</option>
              </select>
            </div>
            <div>
              <label className="label">输出目录</label>
              <input
                className="input"
                value={draft.outputDir}
                onChange={(e) => setDraft({ ...d, outputDir: e.target.value })}
              />
            </div>
          </div>
        </div>

        <div className="modal-foot">
          {saveMsg && <span className="hint">{saveMsg}</span>}
          <button className="btn btn-primary" onClick={save}>保存</button>
        </div>
      </div>
    </div>
  )
}
