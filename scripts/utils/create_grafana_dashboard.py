"""
create_grafana_dashboard.py
Run this on the VM (or locally with appropriate host) to push a
"Model Load Monitor" dashboard to Grafana via the HTTP API.

Usage:
    python3 create_grafana_dashboard.py
"""

import json, sys
import urllib.request, urllib.error

GRAFANA_URL  = "http://localhost:3000"
GRAFANA_USER = "admin"
GRAFANA_PASS = "newpass2026"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

PROM_DS_UID = "ffjmsi0wmmpdsf"   # Prometheus datasource UID (from /api/datasources)


def _api(method: str, path: str, body: dict | None = None):
    url  = GRAFANA_URL + path
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(url, data=data, method=method)
    import base64
    creds = base64.b64encode(f"{GRAFANA_USER}:{GRAFANA_PASS}".encode()).decode()
    req.add_header("Authorization", f"Basic {creds}")
    req.add_header("Content-Type",  "application/json")
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
        raise


def _ts_panel(pid, title, exprs, x, y, w=12, h=8,
              unit="short", color=None, fill=True):
    """Build a timeseries panel dict."""
    targets = []
    colors  = color or ["#73BF69", "#F2CC0C", "#FF7383", "#5794F2",
                        "#FADE2A", "#B877D9", "#37872D", "#FBAD37"]
    for i, (legend, expr) in enumerate(exprs):
        targets.append({
            "datasource": {"type": "prometheus", "uid": PROM_DS_UID},
            "expr": expr,
            "legendFormat": legend,
            "refId": chr(65 + i),
        })

    override_colors = []
    if color:
        for i, (legend, _) in enumerate(exprs):
            override_colors.append({
                "matcher": {"id": "byName", "options": legend},
                "properties": [{"id": "color", "value": {"fixedColor": color[i % len(color)], "mode": "fixed"}}],
            })

    return {
        "id": pid, "type": "timeseries", "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "datasource": {"type": "prometheus", "uid": PROM_DS_UID},
        "targets": targets,
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "custom": {
                    "fillOpacity": 20 if fill else 0,
                    "lineWidth": 2,
                    "spanNulls": True,
                },
                "color": {"mode": "palette-classic"},
            },
            "overrides": override_colors,
        },
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom"},
            "tooltip":  {"mode": "multi"},
        },
    }


def _stat_panel(pid, title, expr, x, y, unit="short", color_mode="thresholds",
                thresholds=None, w=6, h=4):
    """Build a stat panel dict."""
    th = thresholds or [
        {"color": "green",  "value": None},
        {"color": "yellow", "value": 70},
        {"color": "red",    "value": 90},
    ]
    return {
        "id": pid, "type": "stat", "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "datasource": {"type": "prometheus", "uid": PROM_DS_UID},
        "targets": [{
            "datasource": {"type": "prometheus", "uid": PROM_DS_UID},
            "expr": expr, "refId": "A",
        }],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "color": {"mode": color_mode},
                "thresholds": {"mode": "percentage", "steps": th},
                "mappings": [],
            }
        },
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"]},
            "orientation": "auto",
            "colorMode": "background",
            "textMode": "auto",
            "graphMode": "area",
        },
    }


# ---------------------------------------------------------------------------
# GPU UUID variable
# ---------------------------------------------------------------------------

GPU_VAR = {
    "current": {},
    "datasource": {"type": "prometheus", "uid": PROM_DS_UID},
    "definition": "label_values(nvidia_smi_index, uuid)",
    "hide": 0,
    "includeAll": False,
    "label": "GPU",
    "multi": False,
    "name": "gpu",
    "options": [],
    "query": {
        "query": "label_values(nvidia_smi_index, uuid)",
        "refId": "GPU-var",
    },
    "refresh": 1,
    "regex": "",
    "sort": 1,
    "type": "query",
}

# Disk device variable
DISK_VAR = {
    "current": {"value": "sda", "text": "sda"},
    "datasource": {"type": "prometheus", "uid": PROM_DS_UID},
    "definition": "label_values(node_disk_read_bytes_total, device)",
    "hide": 0,
    "includeAll": False,
    "label": "Disk",
    "multi": False,
    "name": "disk",
    "options": [],
    "query": {
        "query": "label_values(node_disk_read_bytes_total, device)",
        "refId": "Disk-var",
    },
    "refresh": 1,
    "regex": "^sd[a-z]$",   # only real disks, skip loop/dm
    "sort": 1,
    "type": "query",
}

# ---------------------------------------------------------------------------
# Panel definitions
# ---------------------------------------------------------------------------

GPU_UUID = 'uuid="$gpu"'

PANELS = [
    # ── Row: GPU ──────────────────────────────────────────────────────────
    {
        "id": 100, "type": "row", "title": "🖥  GPU",
        "gridPos": {"x": 0, "y": 0, "w": 24, "h": 1},
        "collapsed": False,
    },

    # VRAM used – full width, most important
    _ts_panel(1, "VRAM Used (GB)",
              [("Used", f"nvidia_smi_memory_used_bytes{{{GPU_UUID}}} / 1073741824"),
               ("Total", f"nvidia_smi_memory_total_bytes{{{GPU_UUID}}} / 1073741824")],
              x=0, y=1, w=24, h=8,
              unit="GB",
              color=["#FF7383", "#5794F2"]),

    _ts_panel(2, "GPU Utilization %",
              [("GPU %", f"nvidia_smi_utilization_gpu_ratio{{{GPU_UUID}}} * 100")],
              x=0, y=9, w=12, h=7, unit="percent",
              color=["#73BF69"]),

    _ts_panel(3, "GPU Power Draw (W)",
              [("Power", f"nvidia_smi_power_draw_instant_watts{{{GPU_UUID}}}")],
              x=12, y=9, w=12, h=7, unit="watt",
              color=["#FADE2A"]),

    _stat_panel(4, "GPU Temp °C",
                f"nvidia_smi_temperature_gpu{{{GPU_UUID}}}",
                x=0, y=16, unit="celsius",
                thresholds=[
                    {"color": "green",  "value": None},
                    {"color": "yellow", "value": 70},
                    {"color": "red",    "value": 85},
                ], w=6, h=4),

    _stat_panel(5, "VRAM Free (GB)",
                f"nvidia_smi_memory_free_bytes{{{GPU_UUID}}} / 1073741824",
                x=6, y=16, unit="GB",
                color_mode="thresholds",
                thresholds=[
                    {"color": "red",    "value": None},
                    {"color": "yellow", "value": 20},
                    {"color": "green",  "value": 40},
                ], w=6, h=4),

    _stat_panel(6, "GPU Utilization %",
                f"nvidia_smi_utilization_gpu_ratio{{{GPU_UUID}}} * 100",
                x=12, y=16, unit="percent",
                thresholds=[
                    {"color": "green",  "value": None},
                    {"color": "yellow", "value": 70},
                    {"color": "red",    "value": 90},
                ], w=6, h=4),

    _stat_panel(7, "Power Draw (W)",
                f"nvidia_smi_power_draw_instant_watts{{{GPU_UUID}}}",
                x=18, y=16, unit="watt",
                thresholds=[
                    {"color": "green",  "value": None},
                    {"color": "yellow", "value": 250},
                    {"color": "red",    "value": 350},
                ], w=6, h=4),

    # ── Row: Disk I/O ─────────────────────────────────────────────────────
    {
        "id": 200, "type": "row", "title": "💽  Disk I/O  (watch during model load)",
        "gridPos": {"x": 0, "y": 20, "w": 24, "h": 1},
        "collapsed": False,
    },

    _ts_panel(8, "Disk Read MB/s  ($disk)",
              [("Read MB/s", 'rate(node_disk_read_bytes_total{device="$disk"}[15s]) / 1048576')],
              x=0, y=21, w=12, h=7, unit="MBs",
              color=["#5794F2"]),

    _ts_panel(9, "Disk Write MB/s  ($disk)",
              [("Write MB/s", 'rate(node_disk_written_bytes_total{device="$disk"}[15s]) / 1048576')],
              x=12, y=21, w=12, h=7, unit="MBs",
              color=["#FF7383"]),

    # ── Row: System ───────────────────────────────────────────────────────
    {
        "id": 300, "type": "row", "title": "🖧  System (CPU / RAM)",
        "gridPos": {"x": 0, "y": 28, "w": 24, "h": 1},
        "collapsed": False,
    },

    _ts_panel(10, "CPU Utilization %",
              [("CPU %", '(1 - avg(rate(node_cpu_seconds_total{mode="idle"}[15s]))) * 100')],
              x=0, y=29, w=12, h=7, unit="percent",
              color=["#F2CC0C"]),

    _ts_panel(11, "RAM Used (GB)",
              [("Used", "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / 1073741824"),
               ("Total", "node_memory_MemTotal_bytes / 1073741824")],
              x=12, y=29, w=12, h=7, unit="GB",
              color=["#FF7383", "#5794F2"]),
]

# ---------------------------------------------------------------------------
# Dashboard payload
# ---------------------------------------------------------------------------

DASHBOARD = {
    "id":       None,
    "uid":      "model-load-monitor",
    "title":    "Model Load Monitor",
    "tags":     ["gpu", "ai", "model-loading"],
    "refresh":  "5s",
    "time":     {"from": "now-10m", "to": "now"},
    "timepicker": {},
    "timezone": "browser",
    "schemaVersion": 38,
    "version":  0,
    "panels":   PANELS,
    "templating": {"list": [GPU_VAR, DISK_VAR]},
    "annotations": {"list": []},
    "links": [],
}

# ---------------------------------------------------------------------------
# Push to Grafana
# ---------------------------------------------------------------------------

def main():
    payload = {
        "dashboard": DASHBOARD,
        "folderUid": "",
        "overwrite": True,
        "message":   "Created by create_grafana_dashboard.py",
    }

    print("Pushing dashboard to Grafana …")
    try:
        result = _api("POST", "/api/dashboards/db", payload)
        print(f"  ✓ Dashboard created: {result.get('url', '?')}")
        print(f"    uid = {result.get('uid', '?')}")
        print(f"    Open: http://192.168.0.87:3000{result.get('url', '')}")
    except Exception as e:
        print(f"  ✗ Failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
