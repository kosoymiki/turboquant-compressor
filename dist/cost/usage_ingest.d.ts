import type { UsageSnapshot } from './types.js';
export declare function normalizeUsageSnapshots(snapshots: UsageSnapshot[]): UsageSnapshot[];
export declare function sanitizeUsageSnapshots(snapshots: UsageSnapshot[]): UsageSnapshot[];
export declare function validateUsageSnapshot(snapshot: UsageSnapshot): void;
