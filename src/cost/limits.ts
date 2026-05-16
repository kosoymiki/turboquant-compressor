// Context and payload limits for Termux/mobile safety
export const MAX_CONTEXT_FILES = 512;
export const MAX_CONTEXT_TOTAL_BYTES = 8 * 1024 * 1024; // 8 MiB
export const MAX_CONTEXT_FILE_BYTES = 512 * 1024; // 512 KiB
export const MAX_CONTEXT_PATH_LENGTH = 512;
export const MIN_CHUNK_BYTES = 256;
export const MAX_CHUNK_BYTES = 65536; // 64 KiB
export const MAX_CONTEXT_TOP_K = 100;
export const MAX_COST_SNAPSHOTS = 10000;
export const MAX_CACHE_BLOCKS = 512;
export const MAX_INLINE_CHUNK_BYTES = 4096; // Default max inline text per chunk

// Validation functions
export function validateContextFiles(files: unknown): void {
  if (!Array.isArray(files)) {
    throw new Error('files must be an array');
  }
  if (files.length < 1) {
    throw new Error('files must have at least 1 item');
  }
  if (files.length > MAX_CONTEXT_FILES) {
    throw new Error(`files exceeds maximum of ${MAX_CONTEXT_FILES}`);
  }
}

export function validateContextTotalBytes(bytes: number): void {
  if (bytes > MAX_CONTEXT_TOTAL_BYTES) {
    throw new Error(`total context size exceeds ${MAX_CONTEXT_TOTAL_BYTES} bytes`);
  }
}

export function validateContextFileBytes(bytes: number): void {
  if (bytes > MAX_CONTEXT_FILE_BYTES) {
    throw new Error(`file size exceeds ${MAX_CONTEXT_FILE_BYTES} bytes`);
  }
}

export function validateContextPathLength(path: string): void {
  if (path.length > MAX_CONTEXT_PATH_LENGTH) {
    throw new Error(`path length exceeds ${MAX_CONTEXT_PATH_LENGTH} characters`);
  }
}

export function validateChunkBytes(bytes: number): void {
  if (bytes < MIN_CHUNK_BYTES) {
    throw new Error(`chunkBytes must be at least ${MIN_CHUNK_BYTES}`);
  }
  if (bytes > MAX_CHUNK_BYTES) {
    throw new Error(`chunkBytes must not exceed ${MAX_CHUNK_BYTES}`);
  }
}

export function validateTopK(k: number): void {
  if (k < 1) {
    throw new Error('top_k must be at least 1');
  }
  if (k > MAX_CONTEXT_TOP_K) {
    throw new Error(`top_k must not exceed ${MAX_CONTEXT_TOP_K}`);
  }
}

export function validateCostSnapshots(snapshots: unknown): void {
  if (!Array.isArray(snapshots)) {
    throw new Error('usage must be an array');
  }
  if (snapshots.length < 1) {
    throw new Error('usage must have at least 1 item');
  }
  if (snapshots.length > MAX_COST_SNAPSHOTS) {
    throw new Error(`usage exceeds maximum of ${MAX_COST_SNAPSHOTS} items`);
  }
}

export function validateCacheBlocks(blocks: unknown): void {
  if (!Array.isArray(blocks)) {
    throw new Error('blocks must be an array');
  }
  if (blocks.length < 1) {
    throw new Error('blocks must have at least 1 item');
  }
  if (blocks.length > MAX_CACHE_BLOCKS) {
    throw new Error(`blocks exceeds maximum of ${MAX_CACHE_BLOCKS}`);
  }
}

export function validateInlineChunkBytes(bytes: number): void {
  if (bytes < 0) {
    throw new Error('maxInlineChunkBytes must be non-negative');
  }
  if (bytes > MAX_CHUNK_BYTES) {
    throw new Error(`maxInlineChunkBytes must not exceed ${MAX_CHUNK_BYTES}`);
  }
}