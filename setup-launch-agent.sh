#!/bin/bash

# Create a LaunchAgent for 4Demension (auto-start at login)
# This makes 4Demension behave like a real macOS app

LAUNCH_AGENT_DIR="$HOME/Library/LaunchAgents"
PLIST_FILE="$LAUNCH_AGENT_DIR/com.vlasovai.4demension.plist"
SERVICE_SCRIPT="/Users/macos/Desktop/Developer/4Demension/4demension-service"

echo "Setting up 4Demension LaunchAgent..."

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$LAUNCH_AGENT_DIR"

# Create the LaunchAgent plist
cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.vlasovai.4demension</string>
    <key>ProgramArguments</key>
    <array>
        <string>$SERVICE_SCRIPT</string>
        <string>start</string>
    </array>
    <key>RunAtLoad</key>
    <false/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardErrorPath</key>
    <string>/tmp/4demension.error.log</string>
    <key>StandardOutPath</key>
    <string>/tmp/4demension.output.log</string>
    <key>WorkingDirectory</key>
    <string>/Users/macos/Desktop/Developer/4Demension/webapp</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
EOF

echo "✅ LaunchAgent created: $PLIST_FILE"
echo ""
echo "To enable auto-start at login:"
echo "  launchctl load $PLIST_FILE"
echo ""
echo "To disable auto-start:"
echo "  launchctl unload $PLIST_FILE"
echo ""
echo "To start manually (without auto-start):"
echo "  launchctl start com.vlasovai.4demension"
echo ""
echo "To stop:"
echo "  launchctl stop com.vlasovai.4demension"
