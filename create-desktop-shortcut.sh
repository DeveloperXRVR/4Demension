#!/bin/bash

# Create Desktop Shortcut for 4Demension
# Creates a clickable shortcut on the Desktop

DESKTOP_DIR="$HOME/Desktop"
APP_NAME="4Demension"
APP_PATH="/Users/macos/Desktop/Developer/4Demension/${APP_NAME}.app"
SHORTCUT_PATH="${DESKTOP_DIR}/${APP_NAME}"

echo "Creating desktop shortcut for 4Demension..."

# Create a symbolic link (works like an alias)
ln -sf "$APP_PATH" "$SHORTCUT_PATH"

echo "✅ Desktop shortcut created: ${SHORTCUT_PATH}"
echo "You can now double-click the 4Demension icon on your Desktop!"
