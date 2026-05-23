# Typed Array Roadmap

Current LEVEL_1 public implementation uses number[] for readability and MCP JSON compatibility.

For Termux/Android and future WASM backend, internal numeric code should migrate to:

- Float32Array for vectors
- Uint8Array for packed payload
- DataView for binary format

Migration plan:

1. Keep MCP boundary as number[] / number[][].
2. Convert once in validation layer.
3. Use Float32Array internally in rotation, quantizer and search.
4. Keep public API backward compatible.
5. Add benchmark comparing number[] vs Float32Array on Termux.
