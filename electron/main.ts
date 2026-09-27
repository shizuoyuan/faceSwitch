import {
  app,
  BrowserWindow,
  ipcMain,
  dialog,
  shell,
  clipboard
} from 'electron'
import { join, dirname } from 'node:path'
import { spawn, type ChildProcess } from 'node:child_process'
import { createServer } from 'node:net'
import { randomUUID } from 'node:crypto'
import { createWriteStream, mkdirSync, existsSync } from 'node:fs'

// 开发模式:项目根目录;打包后:resources 目录
const isDev = !app.isPackaged
const PROJECT_ROOT = isDev
  ? dirname(dirname(dirname(__filename)))
  : process.resourcesPath!

let mainWindow: BrowserWindow | null = null
let engineProc: ChildProcess | null = null
let engineInfo = { baseUrl: '', token: '' }
let engineLog: ReturnType<typeof createWriteStream> | null = null

function findFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer()
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address()
      const port = typeof addr === 'object' && addr ? addr.port : 0
      srv.close(() => resolve(port))
    })
    srv.on('error', reject)
  })
}

function engineLogPath(): string {
  const dir = isDev
    ? join(PROJECT_ROOT, 'logs')
    : join(app.getPath('userData'), 'logs')
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  return join(dir, 'engine.log')
}

function spawnEngine(port: number, token: string): ChildProcess {
  const pythonExe = isDev
    ? join(PROJECT_ROOT, 'engine', '.venv', 'Scripts', 'python.exe')
    : join(process.resourcesPath!, 'engine', 'engine.exe')

  const proc = isDev
    ? spawn(pythonExe, ['-m', 'engine.api'], {
        cwd: PROJECT_ROOT,
        env: {
          ...process.env,
          ENGINE_PORT: String(port),
          ENGINE_TOKEN: token,
          PYTHONUNBUFFERED: '1',
          HF_ENDPOINT: 'https://hf-mirror.com'
        },
        windowsHide: true
      })
    : spawn(pythonExe, [], {
        cwd: join(process.resourcesPath!, 'engine'),
        env: {
          ...process.env,
          ENGINE_PORT: String(port),
          ENGINE_TOKEN: token,
          PYTHONUNBUFFERED: '1',
          HF_ENDPOINT: 'https://hf-mirror.com'
        },
        windowsHide: true
      })

  engineLog = createWriteStream(engineLogPath(), { flags: 'a' })
  engineLog.write(`\n===== engine start ${new Date().toISOString()} pid=${proc.pid} =====\n`)
  proc.stdout?.on('data', (d) => engineLog?.write(d))
  proc.stderr?.on('data', (d) => engineLog?.write(d))
  proc.on('exit', (code) => {
    engineLog?.write(`\n===== engine exit code=${code} =====\n`)
    if (code !== 0 && code !== null) {
      // 引擎崩溃时把日志路径给剪贴板,方便用户反馈
      try {
        clipboard.writeText(engineLogPath())
      } catch {
        /* ignore */
      }
    }
  })
  return proc
}

async function waitEngineHealthy(baseUrl: string, timeoutMs = 60000): Promise<void> {
  const started = Date.now()
  while (Date.now() - started < timeoutMs) {
    try {
      const res = await fetch(`${baseUrl}/api/health`, {
        headers: { 'X-Engine-Token': engineInfo.token }
      })
      if (res.ok) return
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 400))
  }
  throw new Error(`引擎启动超时,日志: ${engineLogPath()}`)
}

function killEngine() {
  if (!engineProc || engineProc.pid === undefined) return
  // Windows 下必须杀整个进程树,否则 uvicorn 子进程残留
  spawn('taskkill', ['/PID', String(engineProc.pid), '/T', '/F'], {
    windowsHide: true
  })
  engineProc = null
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 1080,
    minHeight: 720,
    backgroundColor: '#0e1116',
    title: 'FaceSwitch',
    autoHideMenuBar: true,
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  })

  // electron-vite dev 模式下加载开发服务器,否则加载构建产物
  if (process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

const VIDEO_EXTS = ['mp4', 'mov', 'mkv', 'avi', 'webm', 'flv', 'ts']
const IMAGE_EXTS = ['png', 'jpg', 'jpeg', 'webp', 'bmp']

function registerIpc() {
  ipcMain.handle('app:engineInfo', () => engineInfo)

  ipcMain.handle('dialog:pickVideo', async () => {
    const res = await dialog.showOpenDialog(mainWindow!, {
      title: '选择视频文件',
      properties: ['openFile'],
      filters: [
        {
          name: '视频',
          extensions: VIDEO_EXTS
        }
      ]
    })
    return res.canceled ? null : res.filePaths[0]
  })

  ipcMain.handle('dialog:pickImages', async () => {
    const res = await dialog.showOpenDialog(mainWindow!, {
      title: '选择图片(可多选)',
      properties: ['openFile', 'multiSelections'],
      filters: [
        {
          name: '图片',
          extensions: IMAGE_EXTS
        }
      ]
    })
    return res.canceled ? [] : res.filePaths
  })

  ipcMain.handle('shell:showInFolder', (_e, path: string) => shell.showItemInFolder(path))
}

app.whenReady().then(async () => {
  registerIpc()

  const port = await findFreePort()
  const token = randomUUID()
  engineInfo = { baseUrl: `http://127.0.0.1:${port}`, token }
  engineProc = spawnEngine(port, token)

  try {
    await waitEngineHealthy(engineInfo.baseUrl)
  } catch (err) {
    dialog.showErrorBox(
      'AI 引擎启动失败',
      `本地 AI 引擎未能启动。请查看日志:\n${engineLogPath()}\n\n${String(err)}`
    )
    app.quit()
    return
  }

  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  killEngine()
  app.quit()
})

app.on('before-quit', () => {
  killEngine()
})
