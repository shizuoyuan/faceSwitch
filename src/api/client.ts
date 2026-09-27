import type { JobRequest, JobStatus, ProbeResult, Settings } from '../types'

let cachedInfo: { baseUrl: string; token: string } | null = null

async function info(): Promise<{ baseUrl: string; token: string }> {
  if (cachedInfo) return cachedInfo
  cachedInfo = await window.electron.engineInfo()
  return cachedInfo
}

async function api<T>(
  path: string,
  init?: RequestInit & { json?: unknown }
): Promise<T> {
  const { baseUrl, token } = await info()
  const headers: Record<string, string> = { 'X-Engine-Token': token }
  let body = init?.body
  if (init?.json !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(init.json)
  }
  const res = await fetch(`${baseUrl}${path}`, { ...init, body, headers })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`请求失败 ${res.status}: ${text.slice(0, 300)}`)
  }
  return res.json() as Promise<T>
}

/** 带鉴权的资源 URL(video/img 标签用,token 走 query) */
export async function assetUrl(path: string): Promise<string> {
  const { baseUrl, token } = await info()
  const sep = path.includes('?') ? '&' : '?'
  return `${baseUrl}${path}${sep}token=${encodeURIComponent(token)}`
}

export const engineApi = {
  health: () => api<{ ok: boolean; gpu: boolean; device: string }>('/api/health'),
  probe: (videoPath: string) =>
    api<ProbeResult>('/api/probe', { method: 'POST', json: { videoPath } }),
  createJob: (req: JobRequest) =>
    api<{ jobId: string }>('/api/jobs', { method: 'POST', json: req }),
  jobStatus: (id: string) => api<JobStatus>(`/api/jobs/${id}`),
  cancelJob: (id: string) => api<{ ok: boolean }>(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  getSettings: () => api<Settings>('/api/settings'),
  saveSettings: (s: Settings) => api<Settings>('/api/settings', { method: 'PUT', json: s }),
  modelStatus: () => api<Record<string, boolean>>('/api/models/status'),
  downloadModels: () => api<{ ok: boolean }>('/api/models/download', { method: 'POST' })
}
