#!/usr/bin/osascript

# AppleScript to create a Dock icon for 4Demension
# This adds the app to the Dock permanently

tell application "Finder"
    set appPath to (POSIX file "/Users/macos/Desktop/Developer/4Demension/4Demension.app") as alias
    try
        add appPath to dock
        display notification "4Demension added to Dock!" with title "Success"
    on error errMsg
        display dialog "Failed to add to Dock: " & errMsg buttons {"OK"} default button "OK"
    end try
end tell
