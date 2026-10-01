/**
 * 浏览器调试模式: 不在 Electron 内运行时注入 electron 桥的 mock,
 * 使 renderer 可通过 `vite dev` 直接在浏览器里预览与调试。
 */
import type { ElectronBridge } from './electron.d'

export function installBrowserMock() {
  if (typeof window === 'undefined' || window.electron) return

  const fakeVideo = 'C:/demo/sample.mp4'
  const fakeFace = 'C:/demo/face.jpg'

  const bridge: ElectronBridge = {
    async engineInfo() {
      return { baseUrl: 'http://127.0.0.1:8765', token: 'mock' }
    },
    async pickVideo() {
      return fakeVideo
    },
    async pickImages() {
      return [fakeFace]
    },
    async showInFolder() {},
    pathForFile() {
      return fakeVideo
    }
  }
  ;(window as unknown as { electron: ElectronBridge }).electron = bridge

  // mock probe 数据
  const origFetch = window.fetch.bind(window)
  window.fetch = async (input, init) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
    if (url.includes('/api/probe')) {
      return new Response(
        JSON.stringify({
          path: fakeVideo,
          fileName: 'sample.mp4',
          width: 1920,
          height: 1080,
          fps: 30,
          frameCount: 900,
          durationSec: 30,
          hasAudio: true,
          thumbnail:
            'data:image/svg+xml;utf8,' +
            encodeURIComponent(
              '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180"><rect width="320" height="180" fill="#1d242f"/><text x="160" y="95" fill="#5a6577" text-anchor="middle" font-size="14">视频缩略图</text></svg>'
            )
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    }
    if (url.includes('/api/settings')) {
      return new Response(
        JSON.stringify({
          providers: [],
          device: 'gpu',
          modelsDir: 'C:/models',
          outputDir: 'C:/output'
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    }
    return origFetch(input, init)
  }
}
