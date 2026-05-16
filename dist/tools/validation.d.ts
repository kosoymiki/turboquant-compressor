import * as z from 'zod/v4';
export declare const CompressInputSchema: z.ZodObject<{
    vectors: z.ZodArray<z.ZodArray<z.ZodNumber>>;
    dimensions: z.ZodOptional<z.ZodNumber>;
    seed: z.ZodOptional<z.ZodNumber>;
    bitsPerValue: z.ZodOptional<z.ZodUnion<readonly [z.ZodLiteral<2>, z.ZodLiteral<3>, z.ZodLiteral<4>, z.ZodLiteral<8>]>>;
    includeQJL: z.ZodOptional<z.ZodBoolean>;
    rotationSeed: z.ZodOptional<z.ZodNumber>;
}, z.core.$strip>;
export declare const SearchInputSchema: z.ZodObject<{
    compressed_database_b64: z.ZodOptional<z.ZodString>;
    query_vector: z.ZodOptional<z.ZodArray<z.ZodNumber>>;
    dimensions: z.ZodOptional<z.ZodNumber>;
    top_k: z.ZodOptional<z.ZodNumber>;
    metric: z.ZodOptional<z.ZodEnum<{
        cosine: "cosine";
        dot: "dot";
    }>>;
    compressedData: z.ZodOptional<z.ZodString>;
    queryVector: z.ZodOptional<z.ZodArray<z.ZodNumber>>;
    k: z.ZodOptional<z.ZodNumber>;
    useQjl: z.ZodOptional<z.ZodBoolean>;
}, z.core.$strip>;
export type CompressInput = z.infer<typeof CompressInputSchema>;
export type SearchInput = z.infer<typeof SearchInputSchema>;
export declare function validateFiniteNumber(value: number, name: string): void;
export declare function validateRectangular2D(vectors: number[][], name: string): void;
export declare function validateStrictBase64(value: string, name: string): void;
export declare function parseCompressInput(input: unknown): CompressInput;
export declare function parseSearchInput(input: unknown): SearchInput;
