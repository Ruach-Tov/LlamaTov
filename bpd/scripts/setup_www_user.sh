#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# Setup www user on NixOS enclave for internet-facing services.
# Run as root: sudo bash setup_www_user.sh
#
# NOTE: NixOS manages users/services declaratively via configuration.nix.
# This script creates the imperative minimum. For persistence across
# nixos-rebuild, add these to configuration.nix.
#
# Author: medayek

set -e

echo "=== Creating www user ==="
useradd -r -s /usr/sbin/nologin -d /srv/www -m www 2>/dev/null || echo "User www already exists"

echo "=== Creating directory structure ==="
mkdir -p /srv/www/static
mkdir -p /srv/www/hooks
mkdir -p /srv/www/logs
mkdir -p /srv/www/repo

echo "=== Cloning repo for www user ==="
if [ ! -d /srv/www/repo/Ruach-Tov/.git ]; then
    git clone https://github.com/Ruach-Tov/Ruach-Tov.git /srv/www/repo/Ruach-Tov
fi

echo "=== Setting ownership ==="
chown -R www:www /srv/www

echo "=== Creating systemd services ==="

# Webhook runner
cat > /etc/systemd/system/bpd-webhook.service << 'EOF'
[Unit]
Description=BPD LLVM Match Webhook Runner
After=network.target

[Service]
Type=simple
User=www
Group=www
WorkingDirectory=/srv/www/repo/Ruach-Tov/bpd
ExecStart=/nix/store/h3q2g9wq4x3q84164qsfm3lz5djj0bf3-python3-3.12.13/bin/python3 /srv/www/repo/Ruach-Tov/bpd/scripts/webhook_runner.py --port 9099
Environment=PYTHONPATH=/nix/store/r3m9fwhp3fmp0zwi32d8a31yi4a1pkqf-python3.12-torch-2.11.0/lib/python3.12/site-packages:/nix/store/m8zsv491f72nfm3c41j5sif1c5kgbksj-python3-3.12.11-env/lib/python3.12/site-packages
Environment=LD_LIBRARY_PATH=/nix/store/84jwqlpfchwgg5ky26qzhg9zh4ybdw0j-python3.12-torch-2.11.0-lib/lib
Restart=always
RestartSec=5
StandardOutput=append:/srv/www/logs/webhook.log
StandardError=append:/srv/www/logs/webhook.log

[Install]
WantedBy=multi-user.target
EOF

# Prolog diviner
cat > /etc/systemd/system/bpd-diviner.service << 'EOF'
[Unit]
Description=BPD Prolog Diviner (live SVG from Prolog)
After=network.target

[Service]
Type=simple
User=www
Group=www
WorkingDirectory=/srv/www/repo/Ruach-Tov
ExecStart=/nix/store/bbyp6vkdszn6a14gqnfx8l5j3mhfcnfs-python3-3.12.11/bin/python3 /srv/www/repo/Ruach-Tov/continuity-guardian/prolog_diviner.py --port 8099
Restart=always
RestartSec=5
StandardOutput=append:/srv/www/logs/diviner.log
StandardError=append:/srv/www/logs/diviner.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

echo ""
echo "=== DONE ==="
echo ""
echo "Directory structure:"
echo "  /srv/www/static/  - Caddy serves static files"
echo "  /srv/www/repo/    - git clone (www user)"
echo "  /srv/www/logs/    - service logs"
echo ""
echo "Next steps:"
echo "  1. Update Caddy root to /srv/www/static/"
echo "  2. systemctl enable --now bpd-webhook"
echo "  3. systemctl enable --now bpd-diviner"
echo "  4. Add GitHub webhook: POST to http://enclave:9099/"
echo "  5. For NixOS persistence: add to configuration.nix"
