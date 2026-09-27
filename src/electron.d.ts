export interface ElectronBridge {
  engineInfo(): Promise<{ baseUrl: string; token: string }>
  pickVideo(): Promise<string | null>
  pickImages(): Promise<string[]>
  showInFolder(path: string): Promise<void>
  pathForFile(file: File): string
}

declare global {
  interface Window {
    electron: ElectronBridge
  }
}

export {}
