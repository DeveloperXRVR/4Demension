#!/bin/bash

# Create macOS app bundle for 4Demension

APP_NAME="4Demension"
APP_DIR="/Users/macos/Desktop/Developer/4Demension/${APP_NAME}.app"
CONTENTS_DIR="${APP_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"

echo "Creating macOS app bundle for 4Demension..."

# Create directory structure
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Create Info.plist
cat > "${CONTENTS_DIR}/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>4Demension</string>
    <key>CFBundleName</key>
    <string>4Demension</string>
    <key>CFBundleIdentifier</key>
    <string>com.vlasovai.4demension</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>4demension-launcher</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSUIElement</key>
    <false/>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# Create launcher script
cat > "${MACOS_DIR}/4demension-launcher" << 'EOF'
#!/bin/bash

# 4Demension Launcher Script
# Starts the webapp and opens browser

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBAPP_DIR="$SCRIPT_DIR/../../webapp"

# Check if webapp directory exists
if [ ! -d "$WEBAPP_DIR" ]; then
    osascript -e 'display dialog "4Demension webapp directory not found!" buttons={"OK"} default button "OK"'
    exit 1
fi

# Change to webapp directory
cd "$WEBAPP_DIR"

# Check if node_modules exists, if not install dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start the development server in background
echo "Starting 4Demension webapp..."
npm run dev > /tmp/4demension.log 2>&1 &
DEV_PID=$!

# Wait a moment for server to start
sleep 3

# Open browser to localhost:3000
open http://localhost:3000

# Show notification
osascript -e 'display notification "4Demension webapp started!" with title "4Demension"'

# Keep the launcher running (you can close this window)
echo "4Demension is running at http://localhost:3000"
echo "Close this window to stop the server."
echo "Process ID: $DEV_PID"

# Wait for user to close terminal or interrupt
trap "kill $DEV_PID 2>/dev/null; exit" INT TERM
wait $DEV_PID
EOF

# Make launcher executable
chmod +x "${MACOS_DIR}/4demension-launcher"

echo "✅ macOS app bundle created: ${APP_DIR}"
echo "You can now double-click 4Demension.app to start the webapp!"
