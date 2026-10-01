import {
  app,
  BrowserWindow,
  ipcMain,
  dialog,
  shell,
  clipboard
} from 'electron'
import { join, dirname } from 'node:path'
import { spawn, execFile, type ChildProcess } from 'node:child_process'
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

// ---- 本地 SDXL 生图服务 (sidecar): 随应用启停 ----
const SD_PORT = Number.parseInt(process.env.FACESWITCH_SD_PORT || '', 10) || 7860
const SD_HEALTH_URL = `http://127.0.0.1:${SD_PORT}/health`
const SD_HEALTH_MARKER = 'faceswitch-sdxl'
let sdProc: ChildProcess | null = null
let sdLog: ReturnType<typeof createWriteStream> | null = null

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

// ---- 本地 SDXL 生图服务管理 ----

function sdLogPath(): string {
  const dir = isDev
    ? join(PROJECT_ROOT, 'logs')
    : join(app.getPath('userData'), 'logs')
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  return join(dir, 'sd_server.log')
}

function sdNote(msg: string) {
  if (!sdLog) sdLog = createWriteStream(sdLogPath(), { flags: 'a' })
  sdLog.write(`${msg}\n`)
}

function resolveSdPython(): string | null {
  // 生图服务依赖 torch+diffusers, 不打进 engine.exe (打包版刻意排除 torch 减重),
  // 因此用机器上的 venv Python 运行随包分发的服务脚本。
  const candidates = [
    process.env.FACESWITCH_SD_PYTHON, // 显式指定: 含 torch+diffusers 的 python.exe
    join(process.resourcesPath!, 'python', 'Scripts', 'python.exe') // 随包分发的 venv (若打包时附带)
  ]
  // 从 resources 逐级向上找仓库树内的 venv (覆盖 release2/、FaceSwitch-release/ 等打包位置)
  let dir: string | undefined = process.resourcesPath
  for (let i = 0; i < 5 && dir; i++) {
    candidates.push(join(dir, 'engine', '.venv', 'Scripts', 'python.exe'))
    const up = dirname(dir)
    if (up === dir) break
    dir = up
  }
  // 部署机已知 venv 位置的兜底 (通用部署请用 FACESWITCH_SD_PYTHON 或随包 venv)
  candidates.push('D:\\face-switch\\engine\\.venv\\Scripts\\python.exe')
  candidates.push('D:\\FaceSwitchDev\\engine\\.venv\\Scripts\\python.exe')
  for (const py of candidates) {
    if (py && existsSync(py)) return py
  }
  return null
}

function resolveSdScript(): string {
  return isDev
    ? join(PROJECT_ROOT, 'engine', 'sd_server', 'local_sd_server.py')
    : join(process.resourcesPath!, 'sd_server', 'local_sd_server.py')
}

function findSdPids(): Promise<string[]> {
  // wmic 输出是 UTF-16, 重定向下带 \0 分隔, 先去掉再解析。
  // 只认 python.exe: LIKE 模式会同时匹配查询进程自身等无关命令行。
  return new Promise((resolve) => {
    execFile(
      'wmic',
      ['process', 'where', "CommandLine Like '%local_sd_server%'", 'get', 'Name,ProcessId', '/format:value'],
      { windowsHide: true, timeout: 8000, maxBuffer: 4 * 1024 * 1024 },
      (err, stdout) => {
        if (err) return resolve([])
        const text = String(stdout).replace(/\0/g, '')
        const pids: string[] = []
        let name = ''
        for (const line of text.split(/\r?\n/)) {
          const eq = line.indexOf('=')
          if (eq < 0) continue
          const key = line.slice(0, eq).trim()
          const val = line.slice(eq + 1).trim()
          if (key === 'Name') name = val.toLowerCase()
          else if (key === 'ProcessId' && name === 'python.exe') pids.push(val)
        }
        resolve(pids)
      }
    )
  })
}

async function waitSdHealthy(timeoutMs = 20000): Promise<boolean> {
  const started = Date.now()
  while (Date.now() - started < timeoutMs) {
    try {
      const res = await fetch(SD_HEALTH_URL, { signal: AbortSignal.timeout(1500) })
      if (res.ok) return true
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 500))
  }
  return false
}

// 返回 'free' 可以拉起; 'foreign' 表示端口被别的服务占用, 不要动它
async function reclaimSdPort(): Promise<'free' | 'foreign'> {
  let marker: string | undefined
  try {
    const res = await fetch(SD_HEALTH_URL, { signal: AbortSignal.timeout(1500) })
    if (res.ok) {
      const data = (await res.json().catch(() => ({}))) as { service?: string }
      marker = data.service
    }
  } catch {
    return 'free' // 端口空闲 (或被非 HTTP 程序占用, 拉起失败会留日志)
  }

  // 端口上有服务应答: 命令行含 local_sd_server 的就是我们自己的实例
  // (上次异常退出的残留, 或旧版/手动启动的), 一律接管清理后重新拉起
  const pids = await findSdPids()
  if (pids.length === 0) {
    if (marker === SD_HEALTH_MARKER) {
      sdNote('[sd] 发现残留实例但无法定位 PID, 跳过本次拉起 (现有实例继续服务)')
    } else {
      sdNote(`[sd] 端口 ${SD_PORT} 已被其他服务占用, 跳过本地生图服务拉起`)
    }
    return 'foreign'
  }
  sdNote(`[sd] 清理残留的本地生图服务实例 (pid: ${pids.join(', ')})...`)
  for (const pid of pids) {
    spawn('taskkill', ['/PID', pid, '/T', '/F'], { windowsHide: true })
  }
  const started = Date.now()
  while (Date.now() - started < 8000) {
    try {
      await fetch(SD_HEALTH_URL, { signal: AbortSignal.timeout(1000) })
    } catch {
      return 'free'
    }
    await new Promise((r) => setTimeout(r, 300))
  }
  return 'foreign'
}

function spawnSdServer(): void {
  const py = resolveSdPython()
  const script = resolveSdScript()
  if (!py || !existsSync(script)) {
    sdNote(
      `[sd] 未找到 Python 环境 (${py ?? '无'}) 或服务脚本 (${script}), 本地生图服务未启动。\n` +
        `[sd] 可设置环境变量 FACESWITCH_SD_PYTHON 指向含 torch+diffusers 的 python.exe`
    )
    return
  }

  sdNote(`\n===== sd server start ${new Date().toISOString()} py=${py} =====`)
  const proc = spawn(py, [script], {
    cwd: dirname(script),
    env: {
      ...process.env,
      FACESWITCH_SD_PORT: String(SD_PORT)
    },
    windowsHide: true
  })
  sdProc = proc
  proc.stdout?.on('data', (d) => sdLog?.write(d))
  proc.stderr?.on('data', (d) => sdLog?.write(d))
  proc.on('error', (err) => {
    sdNote(`[sd] spawn 失败: ${err}`)
    if (sdProc === proc) sdProc = null
  })
  proc.on('exit', (code) => {
    sdNote(`[sd] sd server exit code=${code}`)
    if (sdProc === proc) sdProc = null
  })
}

async function startSdService(): Promise<void> {
  const state = await reclaimSdPort()
  if (state === 'foreign') return
  spawnSdServer()
  const ok = await waitSdHealthy()
  if (ok) {
    sdNote(`[sd] 本地生图服务就绪: ${SD_HEALTH_URL}`)
  } else {
    sdNote(`[sd] 生图服务未在 20s 内就绪, 详情见日志: ${sdLogPath()}`)
  }
}

function killSdServer() {
  if (!sdProc || sdProc.pid === undefined) return
  spawn('taskkill', ['/PID', String(sdProc.pid), '/T', '/F'], {
    windowsHide: true
  })
  sdProc = null
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

// 单实例: 二次启动直接聚焦已有窗口, 也避免重复拉起引擎/生图服务
if (!app.requestSingleInstanceLock()) {
  app.quit()
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore()
      mainWindow.focus()
    }
  })
}

app.whenReady().then(async () => {
  registerIpc()

  // 生图服务与引擎并行拉起, 越早可用; 若引擎失败退出, before-quit 会一并杀掉它
  startSdService().catch((err) => sdNote(`[sd] 启动流程异常: ${err}`))

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
  killSdServer()
  app.quit()
})

app.on('before-quit', () => {
  killEngine()
  killSdServer()
})
