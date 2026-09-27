export interface ProbeResult {
  path: string
  fileName: string
  width: number
  height: number
  fps: number
  frameCount: number
  durationSec: number
  hasAudio: boolean
  thumbnail: string // data URL (jpeg)
}

export interface FaceSwapConfig {
  enabled: boolean
  sourceImages: string[]
  restoreStrength: number // 0-100, GFPGAN 修复混合强度
}

export type BackgroundMode = 'keep' | 'image' | 'video' | 'ai'

export interface BackgroundConfig {
  mode: BackgroundMode
  imagePath?: string
  videoPath?: string
  providerId?: string
  prompt?: string
  fit: 'cover' | 'contain' | 'stretch'
}

export type Quality = 'high' | 'balanced' | 'fast'

export interface JobOptions {
  colorMatch: boolean
  quality: Quality
}

export interface JobRequest {
  videoPath: string
  faceSwap: FaceSwapConfig
  background: BackgroundConfig
  options: JobOptions
}

export type JobState = 'queued' | 'running' | 'done' | 'failed' | 'canceled'

export interface JobStatus {
  id: string
  state: JobState
  stage: string
  stageLabel: string
  progress: number
  frameIndex: number
  frameTotal: number
  message?: string
  outputPath?: string
  error?: string
  elapsedSec?: number
}

export type ProviderType = 'custom' | 'sd-webui' | 'openai-images'

export interface Provider {
  id: string
  name: string
  type: ProviderType
  baseUrl: string
  headersJson: string
  method: 'POST' | 'GET'
  bodyTemplate: string
  responsePath: string
  responseType: 'url' | 'base64'
}

export interface Settings {
  providers: Provider[]
  device: 'gpu' | 'cpu'
  modelsDir: string
  outputDir: string
}

export type WizardStep = 'import' | 'configure' | 'process' | 'export'
