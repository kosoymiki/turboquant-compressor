/**
 * Rotation engine using FWHT-based sign flip pattern.
 * Mobile-friendly rotation without floating-point operations.
 */
export declare const FWHT_SIGN = "FWHT_SIGN";
export declare const DENSE_QR_DEBUG = "DENSE_QR_DEBUG";
export declare const NONE = "NONE";
export type RotationMode = typeof FWHT_SIGN | typeof DENSE_QR_DEBUG | typeof NONE;
export declare class RotationEngine {
    private dimensions;
    private paddedDimensions;
    private seed;
    private signFlipPattern;
    private mode;
    constructor(dimensions: number, seed?: number, mode?: RotationMode);
    private computeSignFlipPattern;
    rotate(vector: Float32Array): Float32Array;
    inverseRotate(rotated: Float32Array): Float32Array;
}
