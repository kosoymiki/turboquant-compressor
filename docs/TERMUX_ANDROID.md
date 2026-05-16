# Termux and Android Deployment

## Overview

TurboQuant Compressor is designed for mobile-first deployment using Termux on Android devices.

## Termux Installation

### Install Termux

1. Download Termux from F-Droid or Google Play Store
2. Update packages:
```bash
pkg update && pkg upgrade
```

### Install Node.js

```bash
# Option 1: Node.js (current)
pkg install nodejs

# Option 2: Node.js LTS (more stable)
pkg install nodejs-lts
```

### Verify Installation

```bash
node --version
npm --version
```

## Project Setup

### Clone and Install

```bash
# Clone the repository
git clone <repository-url>
cd turboquant-compressor

# Install dependencies
npm ci

# Build the project
npm run build
```

## Running on Termux

### Basic Usage

```bash
# Start the MCP server
node dist/server.js

# Run tests
npm test

# Run smoke tests
npm run smoke:stdio
```

### Background Execution

Use Termux's background execution feature or `nohup`:

```bash
nohup node dist/server.js > server.log 2>&1 &
```

## Performance Considerations

### Memory Constraints

Termux has limited memory on Android devices. The project includes safeguards:

- Default max vectors: 4096
- Default max dimensions: 8192
- Memory estimation: `estimateCompressionMemory()`

### Optimization Tips

1. Use lower bits per value (2-4) for memory-constrained devices
2. Compress in batches for large datasets
3. Monitor memory usage with `free` command

### Battery Considerations

- Avoid长时间运行 compression on battery
- Use smaller batch sizes for large operations

## Android-Specific Notes

### File Storage

- Use Termux home directory for project files
- Access shared storage via `/sdcard/`
- Example: `/sdcard/Download/`

### Permissions

Termux may need storage permission:
```bash
termux-setup-storage
```

### ADB Debugging

See [ADB_WORKFLOW.md](ADB_WORKFLOW.md) for debugging instructions.

## Troubleshooting

### Node.js Issues

If Node.js crashes with memory errors:
```bash
# Increase Node.js memory limit
export NODE_OPTIONS="--max-old-space-size=4096"
```

### Performance Issues

- Reduce vector dimensions if too slow
- Use fewer bits per value
- Process vectors in smaller batches

## Security Notes

- The server has no network access by default
- All input is validated before processing
- CRC32 validation catches data corruption