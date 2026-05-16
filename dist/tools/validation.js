import * as z from 'zod/v4';
export const CompressInputSchema = z.object({
    vectors: z.array(z.array(z.number())).min(1),
    dimensions: z.number().int().positive().optional(),
    seed: z.number().int().optional(),
    bitsPerValue: z.union([z.literal(2), z.literal(3), z.literal(4), z.literal(8)]).optional(),
    includeQJL: z.boolean().optional(),
    // legacy aliases
    rotationSeed: z.number().int().optional(),
});
export const SearchInputSchema = z.object({
    compressed_database_b64: z.string().optional(),
    query_vector: z.array(z.number()).min(1).optional(),
    dimensions: z.number().int().min(1).max(8192).optional(),
    top_k: z.number().int().min(1).max(100).optional(),
    metric: z.enum(['cosine', 'dot']).optional(),
    // legacy aliases
    compressedData: z.string().optional(),
    queryVector: z.array(z.number()).min(1).optional(),
    k: z.number().int().min(1).max(100).optional(),
    useQjl: z.boolean().optional(),
}).superRefine((value, ctx) => {
    if (!value.compressed_database_b64 && !value.compressedData) {
        ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: 'compressed_database_b64 is required',
            path: ['compressed_database_b64'],
        });
    }
    if (!value.query_vector && !value.queryVector) {
        ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: 'query_vector is required',
            path: ['query_vector'],
        });
    }
});
export function validateFiniteNumber(value, name) {
    if (!Number.isFinite(value)) {
        throw new Error(`${name} must be finite, got ${value}`);
    }
}
export function validateRectangular2D(vectors, name) {
    if (vectors.length === 0) {
        throw new Error(`${name} must not be empty`);
    }
    const firstLength = vectors[0].length;
    if (firstLength === 0) {
        throw new Error(`${name} rows must not be empty`);
    }
    for (let i = 1; i < vectors.length; i++) {
        if (vectors[i].length !== firstLength) {
            throw new Error(`${name} must be rectangular: row ${i} has length ${vectors[i].length}, expected ${firstLength}`);
        }
    }
}
export function validateStrictBase64(value, name) {
    if (value.length === 0) {
        throw new Error(`${name} must not be empty`);
    }
    if (value.length % 4 !== 0) {
        throw new Error(`${name} has invalid base64 length`);
    }
    const base64Regex = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;
    if (!base64Regex.test(value)) {
        throw new Error(`${name} must be valid base64`);
    }
}
export function parseCompressInput(input) {
    const parsed = CompressInputSchema.parse(input);
    validateRectangular2D(parsed.vectors, 'vectors');
    for (const vector of parsed.vectors) {
        for (const v of vector) {
            validateFiniteNumber(v, 'vector value');
        }
    }
    if (parsed.dimensions !== undefined && parsed.dimensions !== parsed.vectors[0].length) {
        throw new Error(`dimensions ${parsed.dimensions} does not match vector length ${parsed.vectors[0].length}`);
    }
    return parsed;
}
export function parseSearchInput(input) {
    const parsed = SearchInputSchema.parse(input);
    const compressed = parsed.compressed_database_b64 ?? parsed.compressedData;
    const query = parsed.query_vector ?? parsed.queryVector;
    if (!compressed) {
        throw new Error('compressed_database_b64 is required');
    }
    if (!query) {
        throw new Error('query_vector is required');
    }
    validateStrictBase64(compressed, 'compressed_database_b64');
    for (const v of query) {
        validateFiniteNumber(v, 'query value');
    }
    return parsed;
}
