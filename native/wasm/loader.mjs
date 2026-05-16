// turboquant-wasm Node.js loader
// Auto-generated bindings from wasm-bindgen
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

let wasmInstance = null;
let wasmExports = null;

export async function initWasm() {
  if (wasmExports) return wasmExports;

  // Try wasm-bindgen output first, fallback to raw .wasm
  let wasmPath;
  try {
    // wasm-bindgen generated
    const bg = await import('./pkg/turboquant_wasm.js');
    await bg.default();
    wasmExports = bg;
    return wasmExports;
  } catch {
    // Raw wasm fallback
    wasmPath = join(__dirname, 'pkg', 'turboquant_wasm_bg.wasm');
  }

  const wasmBuffer = readFileSync(wasmPath);
  const { instance } = await WebAssembly.instantiate(wasmBuffer);
  wasmExports = instance.exports;
  return wasmExports;
}

export { wasmExports };
