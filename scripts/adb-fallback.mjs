#!/usr/bin/env node
/**
 * adb-fallback.mjs
 * Maintains ADB TCP/IP connection to device even when WiFi disconnects
 * Uses periodic reconnection to 192.168.50.54:40439
 */

import { spawn } from 'node:child_process';
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const DEVICE_IP = '192.168.50.54';
const DEVICE_PORT = '40439';
const RECONNECT_INTERVAL = 30000; // 30 seconds
const MAX_RECONNECT_ATTEMPTS = 5;

let reconnectAttempts = 0;

function log(msg) {
  console.log(`[ADB-FALLBACK] ${msg}`);
}

function connectToDevice() {
  return new Promise((resolve, reject) => {
    const cmd = spawn('adb', ['connect', `${DEVICE_IP}:${DEVICE_PORT}`], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let output = '';
    cmd.stdout.on('data', (data) => { output += data.toString(); });
    cmd.stderr.on('data', (data) => { output += data.toString(); });

    cmd.on('close', (code) => {
      if (code === 0) {
        reconnectAttempts = 0;
        resolve(true);
      } else {
        reject(new Error(`adb connect failed: ${output}`));
      }
    });

    cmd.on('error', (err) => {
      reject(err);
    });

    setTimeout(() => {
      if (!cmd.killed) {
        cmd.kill('SIGTERM');
        reject(new Error('adb connect timeout'));
      }
    }, 10000);
  });
}

async function ensureConnection() {
  try {
    const result = await connectToDevice();
    if (result) {
      log(`Connected to ${DEVICE_IP}:${DEVICE_PORT}`);
    }
  } catch (err) {
    reconnectAttempts++;
    log(`Connection attempt ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS} failed: ${err.message}`);

    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      log('Max reconnect attempts reached. Entering fallback mode.');
      // Fallback: try to restart ADB server
      spawn('adb', ['kill-server'], { stdio: 'ignore' });
      setTimeout(() => {
        reconnectAttempts = 0;
        ensureConnection();
      }, 5000);
    } else {
      setTimeout(ensureConnection, RECONNECT_INTERVAL);
    }
  }
}

// Start the fallback loop
log('Starting ADB fallback service...');
ensureConnection();

// Periodic health check
setInterval(() => {
  const check = spawn('adb', ['shell', 'echo', 'healthcheck'], { stdio: 'ignore' });
  check.on('error', () => {
    log('Health check failed. Reconnecting...');
    ensureConnection();
  });
}, RECONNECT_INTERVAL);

// Handle graceful shutdown
process.on('SIGTERM', () => {
  log('Shutting down ADB fallback service...');
  process.exit(0);
});

process.on('SIGINT', () => {
  log('Shutting down ADB fallback service...');
  process.exit(0);
});
