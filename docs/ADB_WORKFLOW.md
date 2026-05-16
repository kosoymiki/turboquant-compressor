# ADB Workflow for Termux Development

## Overview

This guide explains how to use ADB (Android Debug Bridge) for debugging and developing TurboQuant Compressor on Android devices via Termux.

## Prerequisites

### Install ADB

**On Linux/macOS:**
```bash
# Using package manager
sudo apt install android-tools-adb android-tools-fastboot  # Debian/Ubuntu
brew install android-platform-tools                        # macOS
```

**On Windows:**
Download Android SDK Platform Tools from Google.

### Enable USB Debugging

1. Enable Developer Options on Android:
   - Go to Settings → About Phone
   - Tap Build Number 7 times

2. Enable USB Debugging:
   - Settings → Developer Options → USB Debugging
   - Connect device via USB
   - Authorize the computer

## Basic ADB Commands

### Check Device Connection

```bash
adb devices
```

Should show:
```
List of devices attached
XXXXXXXX    device
```

### Open Termux Shell

```bash
adb shell
```

### Transfer Files

**Push to device:**
```bash
adb push local/path/file /data/data/com.termux/files/home/path/
```

**Pull from device:**
```bash
adb pull /data/data/com.termux/files/home/path/file local/path/
```

## Development Workflow

### 1. Edit Code Locally

Edit files on your development machine.

### 2. Push to Device

```bash
# Push entire project
adb push turboquant-compressor/ /data/data/com.termux/files/home/claude/tools/blackdiamond/

# Or push specific files
adb push turboquant-compressor/dist/server.js /data/data/com.termux/files/home/claude/tools/blackdiamond/turboquant-compressor/dist/
```

### 3. Run on Device

```bash
adb shell
cd /data/data/com.termux/files/home/claude/tools/blackdiamond/turboquant-compressor
npm run build
npm test
npm run smoke:stdio
```

### 4. Pull Results

```bash
# Pull test results
adb pull /data/data/com.termux/files/home/claude/tools/blackdiamond/turboquant-compressor/test-results/ ./

# Pull logs
adb pull /data/data/com.termux/files/home/claude/tools/blackdiamond/turboquant-compressor/server.log ./
```

## Debugging Tips

### View Logs in Real-Time

```bash
adb logcat | grep -E "turboquant|node|Termux"
```

### Check Memory Usage

```bash
adb shell
top -m 10
```

### Monitor CPU Usage

```bash
adb shell
top -b -n 1 | head -20
```

## Troubleshooting

### Device Not Detected

1. Check USB cable
2. Try different USB port
3. Restart ADB server:
```bash
adb kill-server
adb start-server
```

### Permission Denied

```bash
# Grant storage permission
adb shell pm grant com.termux android.permission.WRITE_EXTERNAL_STORAGE
adb shell pm grant com.termux android.permission.READ_EXTERNAL_STORAGE
```

### Termux Not Starting

```bash
adb shell
am start -n com.termux/.MainActivity
```

## Performance Testing

### Benchmark on Device

```bash
adb shell
cd /data/data/com.termux/files/home/claude/tools/blackdiamond/turboquant-compressor
npm run bench:termux
```

### Capture Performance Data

```bash
adb shell
cd /data/data/com.termux/files/home/claude/tools/blackdiamond/turboquant-compressor
time npm test
```

## Script Automation

Create a deployment script:

```bash
#!/bin/bash
# deploy.sh - Deploy and test on Android

echo "Building project..."
npm run build

echo "Pushing to device..."
adb push dist/ /data/data/com.termux/files/home/claude/tools/blackdiamond/turboquant-compressor/dist/

echo "Running tests on device..."
adb shell "cd /data/data/com.termux/files/home/claude/tools/blackdiamond/turboquant-compressor && npm test"

echo "Pulling results..."
adb pull /data/data/com.termux/files/home/claude/tools/blackdiamond/turboquant-compressor/test-results/ ./test-results/

echo "Done!"
```

## Security Notes

- ADB debugging should be disabled when not in use
- Only connect trusted devices
- Review permissions before granting