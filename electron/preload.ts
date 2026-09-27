import { contextBridge, ipcRenderer, webUtils } from 'electron'

contextBridge.exposeInMainWorld('electron', {
  engineInfo: (): Promise<{ baseUrl: string; token: string }> =>
    ipcRenderer.invoke('app:engineInfo'),
  pickVideo: (): Promise<string | null> => ipcRenderer.invoke('dialog:pickVideo'),
  pickImages: (): Promise<string[]> => ipcRenderer.invoke('dialog:pickImages'),
  showInFolder: (path: string): Promise<void> =>
    ipcRenderer.invoke('shell:showInFolder', path),
  pathForFile: (file: File): string => webUtils.getPathForFile(file)
})
