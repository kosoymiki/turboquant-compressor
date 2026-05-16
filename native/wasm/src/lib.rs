use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

#[allow(dead_code)]

/// SIMD f32x4 dot product (4 elements at a time)
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn dot_f32x4(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let n4 = (n / 4) * 4;
    let mut acc = f32x4_splat(0.0);
    for i in (0..n4).step_by(4) {
        unsafe {
            let va = v128_load(a[i..].as_ptr() as *const v128);
            let vb = v128_load(b[i..].as_ptr() as *const v128);
            acc = f32x4_add(acc, f32x4_mul(va, vb));
        }
    }
    let mut sum = f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);
    for i in n4..n {
        sum += a[i] * b[i];
    }
    sum
}

/// SIMD f32x4 scale: out[i] = data[i] * scalar
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn scale_f32x4(data: &mut [f32], scalar: f32) {
    let n = data.len();
    let n4 = (n / 4) * 4;
    let vs = f32x4_splat(scalar);
    for i in (0..n4).step_by(4) {
        unsafe {
            let v = v128_load(data[i..].as_ptr() as *const v128);
            let r = f32x4_mul(v, vs);
            v128_store(data[i..].as_mut_ptr() as *mut v128, r);
        }
    }
    for i in n4..n {
        data[i] *= scalar;
    }
}

#[allow(dead_code)]
/// SIMD f32x4 add-scaled: out[i] += data[i] * scalar
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn add_scaled_f32x4(out: &mut [f32], data: &[f32], scalar: f32) {
    let n = out.len().min(data.len());
    let n4 = (n / 4) * 4;
    let vs = f32x4_splat(scalar);
    for i in (0..n4).step_by(4) {
        unsafe {
            let vo = v128_load(out[i..].as_ptr() as *const v128);
            let vd = v128_load(data[i..].as_ptr() as *const v128);
            let r = f32x4_add(vo, f32x4_mul(vd, vs));
            v128_store(out[i..].as_mut_ptr() as *mut v128, r);
        }
    }
    for i in n4..n {
        out[i] += data[i] * scalar;
    }
}

/// Exposed SIMD dot product for JS
#[wasm_bindgen]
pub fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "wasm32")]
    { dot_f32x4(a, b) }
    #[cfg(not(target_arch = "wasm32"))]
    { a.iter().zip(b.iter()).map(|(x,y)| x*y).sum() }
}

/// Fast Walsh-Hadamard Transform (in-place, power-of-2)
/// Butterfly is data-dependent (not SIMD-able), normalize uses f32x4.
#[wasm_bindgen]
pub fn fwht(data: &mut [f32]) {
    let n = data.len();
    if n == 0 || (n & (n - 1)) != 0 {
        return;
    }
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h *= 2;
    }
    let norm = 1.0 / (n as f32).sqrt();
    #[cfg(target_arch = "wasm32")]
    { scale_f32x4(data, norm); }
    #[cfg(not(target_arch = "wasm32"))]
    { for v in data.iter_mut() { *v *= norm; } }
}

/// Lloyd-Max quantize: find nearest centroid index for each value
/// centroids: sorted array of centroid values
/// Returns packed indices (bits_per_value per element)
#[wasm_bindgen]
pub fn lloyd_max_quantize(values: &[f32], centroids: &[f32], bits: u8) -> Vec<u8> {
    let _n_centroids = centroids.len();
    let vals_per_byte = 8 / bits as usize;
    let packed_len = (values.len() + vals_per_byte - 1) / vals_per_byte;
    let mut packed = vec![0u8; packed_len];
    let mask = (1u8 << bits) - 1;

    for (i, &val) in values.iter().enumerate() {
        // Binary search for nearest centroid
        let mut best = 0usize;
        let mut best_dist = f32::MAX;
        for (c, &cv) in centroids.iter().enumerate() {
            let d = (val - cv).abs();
            if d < best_dist {
                best_dist = d;
                best = c;
            }
        }
        let idx = (best as u8) & mask;
        let byte_pos = i / vals_per_byte;
        let bit_offset = (i % vals_per_byte) * bits as usize;
        packed[byte_pos] |= idx << bit_offset;
    }
    packed
}

/// Dequantize packed indices back to f32 using centroids
#[wasm_bindgen]
pub fn lloyd_max_dequantize(packed: &[u8], centroids: &[f32], bits: u8, count: usize) -> Vec<f32> {
    let vals_per_byte = 8 / bits as usize;
    let mask = (1u8 << bits) - 1;
    let mut out = Vec::with_capacity(count);

    for i in 0..count {
        let byte_pos = i / vals_per_byte;
        let bit_offset = (i % vals_per_byte) * bits as usize;
        let idx = (packed[byte_pos] >> bit_offset) & mask;
        out.push(centroids[idx as usize]);
    }
    out
}

/// QJL 1-bit sign projection: dot product signs with random matrix row
#[wasm_bindgen]
pub fn qjl_sign_project(vector: &[f32], random_matrix: &[f32], proj_dim: usize) -> Vec<u8> {
    let dim = vector.len();
    let packed_len = (proj_dim + 7) / 8;
    let mut bits = vec![0u8; packed_len];

    for p in 0..proj_dim {
        let row_offset = p * dim;
        let mut dot = 0.0f32;
        for d in 0..dim {
            dot += vector[d] * random_matrix[row_offset + d];
        }
        if dot >= 0.0 {
            bits[p / 8] |= 1 << (p % 8);
        }
    }
    bits
}

/// Batch FWHT: apply to N vectors of same dimension
#[wasm_bindgen]
pub fn fwht_batch(data: &mut [f32], dim: usize) {
    if dim == 0 || (dim & (dim - 1)) != 0 {
        return;
    }
    let n_vecs = data.len() / dim;
    for v in 0..n_vecs {
        let offset = v * dim;
        let slice = &mut data[offset..offset + dim];
        let mut h = 1;
        while h < dim {
            for i in (0..dim).step_by(h * 2) {
                for j in i..i + h {
                    let x = slice[j];
                    let y = slice[j + h];
                    slice[j] = x + y;
                    slice[j + h] = x - y;
                }
            }
            h *= 2;
        }
        let norm = 1.0 / (dim as f32).sqrt();
        for val in slice.iter_mut() {
            *val *= norm;
        }
    }
}

/// Bit-pack 4 values (2-bit each) into 1 byte — SIMD-friendly layout
#[wasm_bindgen]
pub fn pack_2bit(indices: &[u8]) -> Vec<u8> {
    let packed_len = (indices.len() + 3) / 4;
    let mut out = vec![0u8; packed_len];
    for (i, &idx) in indices.iter().enumerate() {
        out[i / 4] |= (idx & 0x03) << ((i % 4) * 2);
    }
    out
}

/// Unpack 2-bit values
#[wasm_bindgen]
pub fn unpack_2bit(packed: &[u8], count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let byte = packed[i / 4];
        out.push((byte >> ((i % 4) * 2)) & 0x03);
    }
    out
}
