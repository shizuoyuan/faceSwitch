import { create } from 'zustand'
import type {
  BackgroundConfig,
  FaceSwapConfig,
  JobOptions,
  JobStatus,
  ProbeResult,
  Settings,
  WizardStep
} from './types'

interface AppState {
  step: WizardStep
  probe: ProbeResult | null
  faceSwap: FaceSwapConfig
  background: BackgroundConfig
  options: JobOptions
  jobId: string | null
  jobStatus: JobStatus | null
  settings: Settings | null
  settingsOpen: boolean
  setStep: (s: WizardStep) => void
  setProbe: (p: ProbeResult | null) => void
  setFaceSwap: (patch: Partial<FaceSwapConfig>) => void
  setBackground: (patch: Partial<BackgroundConfig>) => void
  setOptions: (patch: Partial<JobOptions>) => void
  setJob: (id: string | null, status?: JobStatus | null) => void
  setSettings: (s: Settings | null) => void
  setSettingsOpen: (open: boolean) => void
  reset: () => void
}

const initialFaceSwap: FaceSwapConfig = {
  enabled: true,
  sourceImages: [],
  restoreStrength: 60
}

const initialBackground: BackgroundConfig = {
  mode: 'keep',
  fit: 'cover'
}

const initialOptions: JobOptions = {
  colorMatch: true,
  quality: 'balanced'
}

export const useApp = create<AppState>((set) => ({
  step: 'import',
  probe: null,
  faceSwap: initialFaceSwap,
  background: initialBackground,
  options: initialOptions,
  jobId: null,
  jobStatus: null,
  settings: null,
  settingsOpen: false,
  setStep: (step) => set({ step }),
  setProbe: (probe) => set({ probe }),
  setFaceSwap: (patch) =>
    set((st) => ({ faceSwap: { ...st.faceSwap, ...patch } })),
  setBackground: (patch) =>
    set((st) => ({ background: { ...st.background, ...patch } })),
  setOptions: (patch) => set((st) => ({ options: { ...st.options, ...patch } })),
  setJob: (jobId, jobStatus = null) => set({ jobId, jobStatus }),
  setSettings: (settings) => set({ settings }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
  reset: () =>
    set({
      step: 'import',
      probe: null,
      faceSwap: { ...initialFaceSwap },
      background: { ...initialBackground, mode: 'keep' },
      options: { ...initialOptions },
      jobId: null,
      jobStatus: null
    })
}))
