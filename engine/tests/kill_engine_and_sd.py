"""按命令行精确清理 engine.api / local_sd_server / engine.exe 进程 (含 venv 启动器对)。"""
import os
import re
import subprocess


def wmic_text(query: str) -> str:
    out = subprocess.run(
        ["wmic", "process", "where", query, "get", "ProcessId,CommandLine"],
        capture_output=True,
    )
    raw = out.stdout
    if raw.startswith(b"\xff\xfe"):  # UTF-16 LE BOM
        text = raw.decode("utf-16-le", errors="ignore")
    else:
        text = raw.decode("gbk", errors="ignore")
    return text.replace("\0", "")


def target_pids() -> list[int]:
    me = os.getpid()
    pids = []
    for query in ("Name='python.exe'", "Name='engine.exe'"):
        for line in wmic_text(query).splitlines():
            line = line.strip()
            if not line or line.lower().startswith("commandline"):
                continue
            m = re.search(r"(\d+)\s*$", line)
            if not m:
                continue
            pid = int(m.group(1))
            low = line.lower()
            if ("engine.api" in low or "local_sd_server" in low) and pid != me:
                pids.append(pid)
    return pids


if __name__ == "__main__":
    for pid in target_pids():
        print("kill", pid)
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    print("done")
