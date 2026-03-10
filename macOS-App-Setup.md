# 4Demension macOS App Setup

## 🚀 Quick Start

### Option 1: Double-click the App
1. Navigate to `/Users/macos/Desktop/Developer/4Demension/`
2. **Double-click `4Demension.app`** to start the webapp
3. The app will:
   - Install dependencies if needed
   - Start the development server
   - Open your browser at `http://localhost:3000`
   - Show a notification when ready

### Option 2: Use Desktop Shortcut
1. **Double-click the `4Demension` icon on your Desktop**
2. This launches the same app bundle

### Option 3: Add to Dock (Permanent)
```bash
# Run this script to add 4Demension to your Dock permanently
osascript /Users/macos/Desktop/Developer/4Demension/add-to-dock.scpt
```

## 📁 What Was Created

### App Bundle Structure
```
4Demension.app/
├── Contents/
│   ├── Info.plist          # App metadata
│   ├── MacOS/
│   │   └── 4demension-launcher  # Main launcher script
│   └── Resources/
└── (icon files - optional)
```

### Desktop Shortcuts
- **`Start 4Demension.command`** - Simple command file
- **`4Demension` (Desktop symlink)** - Direct app shortcut

## 🎯 How It Works

1. **Auto-dependency check** - Installs npm packages if missing
2. **Server startup** - Runs `npm run dev` in background
3. **Browser launch** - Automatically opens `http://localhost:3000`
4. **Notification** - macOS notification when ready
5. **Clean shutdown** - Stops server when you close the terminal

## 🔧 Customization

### Change Port
Edit the launcher script to use a different port:
```bash
# In 4Demension.app/Contents/MacOS/4demension-launcher
open http://localhost:3001  # Change port here
npm run dev -- -p 3001      # And here
```

### Add Custom Icon
1. Create a 512x512 PNG icon
2. Convert to `.icns` format:
   ```bash
   # Install iconutil if needed
   mkdir -p AppIcon.iconset
   sips -z 16 16 icon.png --out AppIcon.iconset/icon_16x16.png
   sips -z 32 32 icon.png --out AppIcon.iconset/icon_16x16@2x.png
   # ... repeat for all sizes (32, 64, 128, 256, 512)
   iconutil -c icns AppIcon.iconset
   ```
3. Place `AppIcon.icns` in `Contents/Resources/`

## 🗂️ File Management

- **App location**: `/Users/macos/Desktop/Developer/4Demension/4Demension.app`
- **Desktop shortcut**: `/Users/macos/Desktop/4Demension`
- **Logs**: `/tmp/4demension.log`

## 🚨 Troubleshooting

### App doesn't start
```bash
# Check permissions
chmod +x /Users/macos/Desktop/Developer/4Demension/4Demension.app/Contents/MacOS/4demension-launcher
```

### Port already in use
```bash
# Kill existing process
lsof -ti:3000 | xargs kill -9
```

### Dependencies missing
```bash
cd /Users/macos/Desktop/Developer/4Demension/webapp
npm install
```

## 🎉 Enjoy!

Now you can start 4Demension just like any other macOS app - no command line needed!
