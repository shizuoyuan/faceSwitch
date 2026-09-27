import { useApp } from '../store'
import type { WizardStep } from '../types'

const STEPS: { key: WizardStep; label: string; index: number }[] = [
  { key: 'import', label: '导入视频', index: 1 },
  { key: 'configure', label: '配置任务', index: 2 },
  { key: 'process', label: 'AI 处理', index: 3 },
  { key: 'export', label: '导出成品', index: 4 }
]

export function StepBar() {
  const step = useApp((s) => s.step)
  const current = STEPS.find((s) => s.key === step)!

  return (
    <div className="stepbar">
      {STEPS.map((s) => {
        const active = s.key === step
        const done = s.index < current.index
        return (
          <div key={s.key} className={`step ${active ? 'active' : ''} ${done ? 'done' : ''}`}>
            <span className="step-num">{done ? '✓' : s.index}</span>
            <span>{s.label}</span>
            {s.index < STEPS.length && <span className="step-line" />}
          </div>
        )
      })}
    </div>
  )
}
