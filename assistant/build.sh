#!/usr/bin/env bash
# Собирает Assistant.app — нативный macOS-бандл (нужен для доступа к микрофону
# через TCC и чтобы приложение не висело иконкой в доке, LSUIElement).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

APP="Assistant.app"
BIN="$APP/Contents/MacOS/Assistant"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp Info.plist "$APP/Contents/Info.plist"

echo "Компилирую…" >&2
swiftc -O -swift-version 5 main.swift -o "$BIN" \
    -framework AppKit -framework AVFoundation -framework QuartzCore

# Ad-hoc подпись — стабилизирует запись разрешения микрофона в TCC.
codesign --force --sign - "$APP" >/dev/null 2>&1 || true

echo "Готово: $APP" >&2
