#!/opt/arthur-img-env/bin/python
"""
Auto-status checker for Ideogram 4.
Checks VRAM, service status, test results, and logs.
Run: python3 /tmp/check_status.py [--watch]
"""
import subprocess, sys, time, json

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

def check():
    vram  = run("nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader")
    svc   = run("sudo systemctl is-active arthur-imglab.service")
    proc  = run("ps aux | grep image_lab | grep -v grep | awk '{printf \"CPU:%s%% RSS:%.0fGB\",$3,$6/1024/1024}'")
    
    result_file = sys.argv[2] if len(sys.argv) > 2 else "/tmp/newspaper_result.txt"
    result = run(f"cat {result_file} 2>/dev/null | head -c 300")
    
    loading = run("sudo journalctl -u arthur-imglab.service --no-pager -n 3 | grep 'Loading Ideogram' | tail -1 | grep -o 'quant=[^ ]*' || echo ''")
    
    print(f"{'='*50}")
    print(f"Time:    {time.strftime('%H:%M:%S')}")
    print(f"Service: {svc}")
    print(f"VRAM:    {vram}")
    print(f"Process: {proc}")
    print(f"Loading: {loading if loading else 'idle'}")
    print(f"Result:  {result[:150] if result else '(empty)'}")
    print(f"{'='*50}")

if __name__ == "__main__":
    if "--watch" in sys.argv:
        interval = int(sys.argv[sys.argv.index("--watch")+1]) if len(sys.argv) > sys.argv.index("--watch")+1 else 30
        while True:
            check()
            time.sleep(interval)
    else:
        check()
