$os = Get-CimInstance Win32_OperatingSystem
"TotalVM(MB): " + [math]::Round($os.TotalVirtualMemorySize/1KB)
"FreeVM(MB):  " + [math]::Round($os.FreeVirtualMemory/1KB)
"FreePhys(MB): " + [math]::Round($os.FreePhysicalMemory/1KB)
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 10 Name,
  @{n='WS_MB';e={[math]::Round($_.WorkingSet64/1MB)}},
  @{n='Commit_MB';e={[math]::Round($_.PagedMemorySize64/1MB)}} |
  Format-Table -AutoSize
