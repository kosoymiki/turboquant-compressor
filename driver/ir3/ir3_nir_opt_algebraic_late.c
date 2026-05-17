#include "ir3_nir.h"

#include "nir.h"
#include "nir_builder.h"
#include "nir_search.h"
#include "nir_search_helpers.h"

/* What follows is NIR algebraic transform code for the following 8
 * transforms:
 *    ('fadd@16', ('fmul(is_only_used_by_fadd)', 'a', 'b'), 'c') => ('ffma', 'a', 'b', 'c')
 *    ('fadd@16', ('fneg(is_only_used_by_fadd)', ('fmul(is_only_used_by_fadd)', 'a', 'b')), 'c') => ('ffma', ('fneg', 'a'), 'b', 'c')
 *    ('fadd@16', ('fabs(is_only_used_by_fadd)', ('fmul(is_only_used_by_fadd)', 'a', 'b')), 'c') => ('ffma', ('fabs', 'a'), ('fabs', 'b'), 'c')
 *    ('fadd@16', ('fneg(is_only_used_by_fadd)', ('fabs', ('fmul(is_only_used_by_fadd)', 'a', 'b'))), 'c') => ('ffma', ('fneg', ('fabs', 'a')), ('fabs', 'b'), 'c')
 *    ('fadd@32', ('fmul(is_only_used_by_fadd)', 'a', 'b'), 'c') => ('ffma', 'a', 'b', 'c')
 *    ('fadd@32', ('fneg(is_only_used_by_fadd)', ('fmul(is_only_used_by_fadd)', 'a', 'b')), 'c') => ('ffma', ('fneg', 'a'), 'b', 'c')
 *    ('fadd@32', ('fabs(is_only_used_by_fadd)', ('fmul(is_only_used_by_fadd)', 'a', 'b')), 'c') => ('ffma', ('fabs', 'a'), ('fabs', 'b'), 'c')
 *    ('fadd@32', ('fneg(is_only_used_by_fadd)', ('fabs', ('fmul(is_only_used_by_fadd)', 'a', 'b'))), 'c') => ('ffma', ('fneg', ('fabs', 'a')), ('fabs', 'b'), 'c')
 */


static const nir_search_value_union ir3_nir_opt_algebraic_late_values[] = {
   /* ('fadd@16', ('fmul(is_only_used_by_fadd)', 'a', 'b'), 'c') => ('ffma', 'a', 'b', 'c') */
   { .variable = {
      { nir_search_value_variable, 16 },
      0, /* a */
      false,
      -1,
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
   } },
   { .variable = {
      { nir_search_value_variable, 16 },
      1, /* b */
      false,
      -1,
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
   } },
   { .expression = {
      { nir_search_value_expression, 16 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fmul,
      1, 1,
      { 0, 1 },
      0,
   } },
   { .variable = {
      { nir_search_value_variable, 16 },
      2, /* c */
      false,
      -1,
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
   } },
   { .expression = {
      { nir_search_value_expression, 16 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fadd,
      0, 2,
      { 2, 3 },
      -1,
   } },

   /* replace0_0 -> 0 in the cache */
   /* replace0_1 -> 1 in the cache */
   /* replace0_2 -> 3 in the cache */
   { .expression = {
      { nir_search_value_expression, 16 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_ffma,
      0, 1,
      { 0, 1, 3 },
      -1,
   } },

   /* ('fadd@16', ('fneg(is_only_used_by_fadd)', ('fmul(is_only_used_by_fadd)', 'a', 'b')), 'c') => ('ffma', ('fneg', 'a'), 'b', 'c') */
   /* search1_0_0_0 -> 0 in the cache */
   /* search1_0_0_1 -> 1 in the cache */
   /* search1_0_0 -> 2 in the cache */
   { .expression = {
      { nir_search_value_expression, 16 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fneg,
      -1, 1,
      { 2 },
      0,
   } },
   /* search1_1 -> 3 in the cache */
   { .expression = {
      { nir_search_value_expression, 16 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fadd,
      0, 2,
      { 6, 3 },
      -1,
   } },

   /* replace1_0_0 -> 0 in the cache */
   { .expression = {
      { nir_search_value_expression, 16 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fneg,
      -1, 0,
      { 0 },
      -1,
   } },
   /* replace1_1 -> 1 in the cache */
   /* replace1_2 -> 3 in the cache */
   { .expression = {
      { nir_search_value_expression, 16 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_ffma,
      0, 1,
      { 8, 1, 3 },
      -1,
   } },

   /* ('fadd@16', ('fabs(is_only_used_by_fadd)', ('fmul(is_only_used_by_fadd)', 'a', 'b')), 'c') => ('ffma', ('fabs', 'a'), ('fabs', 'b'), 'c') */
   /* search2_0_0_0 -> 0 in the cache */
   /* search2_0_0_1 -> 1 in the cache */
   /* search2_0_0 -> 2 in the cache */
   { .expression = {
      { nir_search_value_expression, 16 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fabs,
      -1, 1,
      { 2 },
      0,
   } },
   /* search2_1 -> 3 in the cache */
   { .expression = {
      { nir_search_value_expression, 16 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fadd,
      0, 2,
      { 10, 3 },
      -1,
   } },

   /* replace2_0_0 -> 0 in the cache */
   { .expression = {
      { nir_search_value_expression, 16 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fabs,
      -1, 0,
      { 0 },
      -1,
   } },
   /* replace2_1_0 -> 1 in the cache */
   { .expression = {
      { nir_search_value_expression, 16 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fabs,
      -1, 0,
      { 1 },
      -1,
   } },
   /* replace2_2 -> 3 in the cache */
   { .expression = {
      { nir_search_value_expression, 16 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_ffma,
      0, 1,
      { 12, 13, 3 },
      -1,
   } },

   /* ('fadd@16', ('fneg(is_only_used_by_fadd)', ('fabs', ('fmul(is_only_used_by_fadd)', 'a', 'b'))), 'c') => ('ffma', ('fneg', ('fabs', 'a')), ('fabs', 'b'), 'c') */
   /* search3_0_0_0_0 -> 0 in the cache */
   /* search3_0_0_0_1 -> 1 in the cache */
   /* search3_0_0_0 -> 2 in the cache */
   { .expression = {
      { nir_search_value_expression, 16 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fabs,
      -1, 1,
      { 2 },
      -1,
   } },
   { .expression = {
      { nir_search_value_expression, 16 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fneg,
      -1, 1,
      { 15 },
      0,
   } },
   /* search3_1 -> 3 in the cache */
   { .expression = {
      { nir_search_value_expression, 16 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fadd,
      0, 2,
      { 16, 3 },
      -1,
   } },

   /* replace3_0_0_0 -> 0 in the cache */
   /* replace3_0_0 -> 12 in the cache */
   { .expression = {
      { nir_search_value_expression, 16 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fneg,
      -1, 0,
      { 12 },
      -1,
   } },
   /* replace3_1_0 -> 1 in the cache */
   /* replace3_1 -> 13 in the cache */
   /* replace3_2 -> 3 in the cache */
   { .expression = {
      { nir_search_value_expression, 16 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_ffma,
      0, 1,
      { 18, 13, 3 },
      -1,
   } },

   /* ('fadd@32', ('fmul(is_only_used_by_fadd)', 'a', 'b'), 'c') => ('ffma', 'a', 'b', 'c') */
   { .variable = {
      { nir_search_value_variable, 32 },
      0, /* a */
      false,
      -1,
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
   } },
   { .variable = {
      { nir_search_value_variable, 32 },
      1, /* b */
      false,
      -1,
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
   } },
   { .expression = {
      { nir_search_value_expression, 32 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fmul,
      1, 1,
      { 20, 21 },
      0,
   } },
   { .variable = {
      { nir_search_value_variable, 32 },
      2, /* c */
      false,
      -1,
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
   } },
   { .expression = {
      { nir_search_value_expression, 32 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fadd,
      0, 2,
      { 22, 23 },
      -1,
   } },

   /* replace4_0 -> 20 in the cache */
   /* replace4_1 -> 21 in the cache */
   /* replace4_2 -> 23 in the cache */
   { .expression = {
      { nir_search_value_expression, 32 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_ffma,
      0, 1,
      { 20, 21, 23 },
      -1,
   } },

   /* ('fadd@32', ('fneg(is_only_used_by_fadd)', ('fmul(is_only_used_by_fadd)', 'a', 'b')), 'c') => ('ffma', ('fneg', 'a'), 'b', 'c') */
   /* search5_0_0_0 -> 20 in the cache */
   /* search5_0_0_1 -> 21 in the cache */
   /* search5_0_0 -> 22 in the cache */
   { .expression = {
      { nir_search_value_expression, 32 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fneg,
      -1, 1,
      { 22 },
      0,
   } },
   /* search5_1 -> 23 in the cache */
   { .expression = {
      { nir_search_value_expression, 32 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fadd,
      0, 2,
      { 26, 23 },
      -1,
   } },

   /* replace5_0_0 -> 20 in the cache */
   { .expression = {
      { nir_search_value_expression, 32 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fneg,
      -1, 0,
      { 20 },
      -1,
   } },
   /* replace5_1 -> 21 in the cache */
   /* replace5_2 -> 23 in the cache */
   { .expression = {
      { nir_search_value_expression, 32 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_ffma,
      0, 1,
      { 28, 21, 23 },
      -1,
   } },

   /* ('fadd@32', ('fabs(is_only_used_by_fadd)', ('fmul(is_only_used_by_fadd)', 'a', 'b')), 'c') => ('ffma', ('fabs', 'a'), ('fabs', 'b'), 'c') */
   /* search6_0_0_0 -> 20 in the cache */
   /* search6_0_0_1 -> 21 in the cache */
   /* search6_0_0 -> 22 in the cache */
   { .expression = {
      { nir_search_value_expression, 32 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fabs,
      -1, 1,
      { 22 },
      0,
   } },
   /* search6_1 -> 23 in the cache */
   { .expression = {
      { nir_search_value_expression, 32 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fadd,
      0, 2,
      { 30, 23 },
      -1,
   } },

   /* replace6_0_0 -> 20 in the cache */
   { .expression = {
      { nir_search_value_expression, 32 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fabs,
      -1, 0,
      { 20 },
      -1,
   } },
   /* replace6_1_0 -> 21 in the cache */
   { .expression = {
      { nir_search_value_expression, 32 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fabs,
      -1, 0,
      { 21 },
      -1,
   } },
   /* replace6_2 -> 23 in the cache */
   { .expression = {
      { nir_search_value_expression, 32 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_ffma,
      0, 1,
      { 32, 33, 23 },
      -1,
   } },

   /* ('fadd@32', ('fneg(is_only_used_by_fadd)', ('fabs', ('fmul(is_only_used_by_fadd)', 'a', 'b'))), 'c') => ('ffma', ('fneg', ('fabs', 'a')), ('fabs', 'b'), 'c') */
   /* search7_0_0_0_0 -> 20 in the cache */
   /* search7_0_0_0_1 -> 21 in the cache */
   /* search7_0_0_0 -> 22 in the cache */
   { .expression = {
      { nir_search_value_expression, 32 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fabs,
      -1, 1,
      { 22 },
      -1,
   } },
   { .expression = {
      { nir_search_value_expression, 32 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fneg,
      -1, 1,
      { 35 },
      0,
   } },
   /* search7_1 -> 23 in the cache */
   { .expression = {
      { nir_search_value_expression, 32 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fadd,
      0, 2,
      { 36, 23 },
      -1,
   } },

   /* replace7_0_0_0 -> 20 in the cache */
   /* replace7_0_0 -> 32 in the cache */
   { .expression = {
      { nir_search_value_expression, 32 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_fneg,
      -1, 0,
      { 32 },
      -1,
   } },
   /* replace7_1_0 -> 21 in the cache */
   /* replace7_1 -> 33 in the cache */
   /* replace7_2 -> 23 in the cache */
   { .expression = {
      { nir_search_value_expression, 32 },
      nir_fp_fast_math,
      nir_fp_fast_math,
      false,
      -1,
      nir_op_ffma,
      0, 1,
      { 38, 33, 23 },
      -1,
   } },

};

UNUSED static const nir_search_expression_cond ir3_nir_opt_algebraic_late_expression_cond[] = {
   is_only_used_by_fadd,
};


static const struct transform ir3_nir_opt_algebraic_late_transforms[] = {
   { ~0, ~0, ~0 }, /* Sentinel */

   { 4, 5, 0 },
   { 24, 25, 0 },
   { ~0, ~0, ~0 }, /* Sentinel */

   { 7, 9, 0 },
   { 27, 29, 0 },
   { ~0, ~0, ~0 }, /* Sentinel */

   { 11, 14, 0 },
   { 31, 34, 0 },
   { ~0, ~0, ~0 }, /* Sentinel */

   { 4, 5, 0 },
   { 7, 9, 0 },
   { 24, 25, 0 },
   { 27, 29, 0 },
   { ~0, ~0, ~0 }, /* Sentinel */

   { 4, 5, 0 },
   { 11, 14, 0 },
   { 24, 25, 0 },
   { 31, 34, 0 },
   { ~0, ~0, ~0 }, /* Sentinel */

   { 7, 9, 0 },
   { 11, 14, 0 },
   { 27, 29, 0 },
   { 31, 34, 0 },
   { ~0, ~0, ~0 }, /* Sentinel */

   { 17, 19, 0 },
   { 37, 39, 0 },
   { ~0, ~0, ~0 }, /* Sentinel */

   { 4, 5, 0 },
   { 17, 19, 0 },
   { 24, 25, 0 },
   { 37, 39, 0 },
   { ~0, ~0, ~0 }, /* Sentinel */

   { 7, 9, 0 },
   { 17, 19, 0 },
   { 27, 29, 0 },
   { 37, 39, 0 },
   { ~0, ~0, ~0 }, /* Sentinel */

   { 11, 14, 0 },
   { 17, 19, 0 },
   { 31, 34, 0 },
   { 37, 39, 0 },
   { ~0, ~0, ~0 }, /* Sentinel */

};

static const struct per_op_table ir3_nir_opt_algebraic_late_pass_op_table[nir_num_search_ops] = {
   [nir_op_fadd] = {
      .filter = (const uint16_t []) {
         0,
         0,
         1,
         0,
         2,
         3,
         0,
         0,
         0,
         0,
         0,
         4,
         0,
         0,
         0,
         0,
      },
      
      .num_filtered_states = 5,
      .table = (const uint16_t []) {
      
         0,
         3,
         6,
         7,
         12,
         3,
         3,
         8,
         9,
         13,
         6,
         8,
         6,
         10,
         14,
         7,
         9,
         10,
         7,
         15,
         12,
         13,
         14,
         15,
         12,
      },
   },
   [nir_op_fmul] = {
      .filter = NULL,
      
      .num_filtered_states = 1,
      .table = (const uint16_t []) {
      
         2,
      },
   },
   [nir_op_fneg] = {
      .filter = (const uint16_t []) {
         0,
         0,
         1,
         0,
         0,
         2,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
      },
      
      .num_filtered_states = 3,
      .table = (const uint16_t []) {
      
         0,
         4,
         11,
      },
   },
   [nir_op_fabs] = {
      .filter = (const uint16_t []) {
         0,
         0,
         1,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
      },
      
      .num_filtered_states = 2,
      .table = (const uint16_t []) {
      
         0,
         5,
      },
   },
};

/* Mapping from state index to offset in transforms (0 being no transforms) */
static const uint16_t ir3_nir_opt_algebraic_late_transform_offsets[] = {
   0,
   0,
   0,
   1,
   0,
   0,
   4,
   7,
   10,
   15,
   20,
   0,
   25,
   28,
   33,
   38,
};

static const nir_algebraic_table ir3_nir_opt_algebraic_late_table = {
   .transforms = ir3_nir_opt_algebraic_late_transforms,
   .transform_offsets = ir3_nir_opt_algebraic_late_transform_offsets,
   .pass_op_table = ir3_nir_opt_algebraic_late_pass_op_table,
   .values = ir3_nir_opt_algebraic_late_values,
   .expression_cond = ir3_nir_opt_algebraic_late_expression_cond,
   .variable_cond = NULL,
};

bool
ir3_nir_opt_algebraic_late(
   nir_shader *shader
) {
   bool progress = false;
   bool condition_flags[1];
   const nir_shader_compiler_options *options = shader->options;
   const shader_info *info = &shader->info;
   (void) options;
   (void) info;

   STATIC_ASSERT(40 == ARRAY_SIZE(ir3_nir_opt_algebraic_late_values));
   condition_flags[0] = true;

   nir_foreach_function_impl(impl, shader) {
     progress |= nir_algebraic_impl(impl, condition_flags, &ir3_nir_opt_algebraic_late_table);
   }

   return progress;
}

