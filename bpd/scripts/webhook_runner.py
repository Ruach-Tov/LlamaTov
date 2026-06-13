#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""webhook_runner.py - GitHub push webhook triggers CI pipeline on enclave.

Listens on port 9099 for POST from GitHub.
On push to main: pull → build → test → update dashboard.

The complete pipeline is in bpd/scripts/ci_build_test_dashboard.py.

Start: nohup python3 bpd/scripts/webhook_runner.py --port 9099 &
"""
import http.server, json, subprocess, os, sys, threading, time

PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 9099
REPO = os.path.expanduser("~/Ruach-Tov")
PY = sys.executable  # use same python as webhook runner
PIPELINE = os.path.join(REPO, "bpd/scripts/ci_build_test_dashboard.py")
LOG_DIR = "/tmp/bpd-generated/logs"

def run_pipeline():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, time.strftime("ci-%Y%m%d-%H%M%S.log"))
    
    print(f"[webhook] Starting pipeline, log: {log_file}", flush=True)
    
    with open(log_file, 'w') as log:
        r = subprocess.run(
            [PY, PIPELINE],
            cwd=REPO,
            stdout=log, stderr=subprocess.STDOUT,
            timeout=300
        )
    
    print(f"[webhook] Pipeline done (exit {r.returncode})", flush=True)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            p = json.loads(body)
            if p.get("ref") == "refs/heads/main":
                print("[webhook] Push to main — running pipeline...", flush=True)
                threading.Thread(target=run_pipeline, daemon=True).start()
        except:
            pass
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
    
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        # Status page
        msg = f"webhook alive, repo={REPO}, pipeline={PIPELINE}\n"
        self.wfile.write(msg.encode())
    
    def log_message(self, format, *args):
        pass  # quiet


if __name__ == "__main__":
    s = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[webhook] Listening on :{PORT}", flush=True)
    print(f"[webhook] Repo: {REPO}", flush=True)
    print(f"[webhook] Pipeline: {PIPELINE}", flush=True)
    s.serve_forever()

# ─── Table(10011) Divergence Dashboard CI ───
DIVERGENCE_PIPELINE = os.path.join(REPO, "bpd/model_coverage/run_divergence_oracles.py")
DIVERGENCE_OUTPUT = "/tmp/bpd-generated/div_fixtures.o.pl"

def run_divergence_pipeline():
    """Run Table(10011) divergence oracles and generate facts."""
    if not os.path.exists(DIVERGENCE_PIPELINE):
        print("[webhook] No divergence pipeline found, skipping Table(10011)", flush=True)
        return
    
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, time.strftime("ci-div-%Y%m%d-%H%M%S.log"))
    
    print(f"[webhook] Starting divergence pipeline, log: {log_file}", flush=True)
    
    with open(log_file, "w") as log:
        r = subprocess.run(
            [PY, DIVERGENCE_PIPELINE, "--output", DIVERGENCE_OUTPUT],
            cwd=REPO,
            stdout=log, stderr=subprocess.STDOUT,
            timeout=600  # oracles may take longer (builds + runs)
        )
    
    print(f"[webhook] Divergence pipeline done (exit {r.returncode})", flush=True)
