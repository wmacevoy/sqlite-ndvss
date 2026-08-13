#ifndef SIMILARITY_FUNCTIONS_WASMSIMD_H_INCLUDED
#define SIMILARITY_FUNCTIONS_WASMSIMD_H_INCLUDED
/**
 * This file contains the WebAssembly SIMD128 versions of the similarity
 * function definitions. Unlike the x86/ARM backends, WASM has no runtime
 * CPU-feature dispatch: a .wasm module either was compiled with -msimd128
 * (Emscripten/clang) or wasn't, so this whole translation unit must be
 * built with that flag - there is no per-function target attribute to
 * request it selectively, as sse41.h/avx.h do on x86.
 */
#include <wasm_simd128.h>



//----------------------------------------------------------------------------------------
// Name: cosine_similarity_f_wasmsimd
// Desc: Calculates the cosine similarity using two given float arrays. WASM SIMD128 version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB,
//       Number of dimensions INTEGER
//       Pointer to divider_a FLOAT
//       Pointer to divider_b FLOAT
// Returns: Similarity as an angle float
//----------------------------------------------------------------------------------------
float cosine_similarity_f_wasmsimd(
     const float*   searched_array
    ,const float*   column_array
    ,const int      vector_size
    ,float*         divider_a
    ,float*         divider_b )
{
    int   i = 0;
    float dividerA   = 0.0f
         ,dividerB   = 0.0f
         ,similarity = 0.0f;
    v128_t A
          ,B
          ,AA
          ,BB
          ,AB
          ,mmdividerA   = wasm_f32x4_splat(0.0f)
          ,mmdividerB   = wasm_f32x4_splat(0.0f)
          ,mmsimilarity = wasm_f32x4_splat(0.0f);

    for( ; i + 3 < vector_size; i += 4 ) {
        A = wasm_v128_load(&searched_array[i]);
        B = wasm_v128_load(&column_array[i]);
        AA = wasm_f32x4_mul(A, A);
        BB = wasm_f32x4_mul(B, B);
        AB = wasm_f32x4_mul(A, B);
        mmdividerA = wasm_f32x4_add(AA, mmdividerA);
        mmdividerB = wasm_f32x4_add(BB, mmdividerB);
        mmsimilarity = wasm_f32x4_add(AB, mmsimilarity);
    }//endfor i+4

    dividerA = wasm_f32x4_extract_lane(mmdividerA, 0) + wasm_f32x4_extract_lane(mmdividerA, 1)
             + wasm_f32x4_extract_lane(mmdividerA, 2) + wasm_f32x4_extract_lane(mmdividerA, 3);
    dividerB = wasm_f32x4_extract_lane(mmdividerB, 0) + wasm_f32x4_extract_lane(mmdividerB, 1)
             + wasm_f32x4_extract_lane(mmdividerB, 2) + wasm_f32x4_extract_lane(mmdividerB, 3);
    similarity = wasm_f32x4_extract_lane(mmsimilarity, 0) + wasm_f32x4_extract_lane(mmsimilarity, 1)
               + wasm_f32x4_extract_lane(mmsimilarity, 2) + wasm_f32x4_extract_lane(mmsimilarity, 3);

    // Calculate the remaining elements.
    for(; i < vector_size; ++i ) {
        float A = searched_array[i];
        float B = column_array[i];
        similarity += (A*B);
        dividerA   += (A*A);
        dividerB   += (B*B);
    }

    *divider_a = dividerA;
    *divider_b = dividerB;
    return similarity;
}


//----------------------------------------------------------------------------------------
// Name: cosine_similarity_d_wasmsimd
// Desc: Calculates the cosine similarity using two given double arrays. WASM SIMD128 version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB,
//       Number of dimensions INTEGER
//       Pointer to divider_a DOUBLE
//       Pointer to divider_b DOUBLE
// Returns: Similarity as an angle DOUBLE
//----------------------------------------------------------------------------------------
double cosine_similarity_d_wasmsimd(
     const double*   searched_array
    ,const double*   column_array
    ,const int       vector_size
    ,double*         divider_a
    ,double*         divider_b )
{
    int   i = 0;
    double dividerA   = 0.0
          ,dividerB   = 0.0
          ,similarity = 0.0;
    v128_t A
          ,B
          ,AA
          ,BB
          ,AB
          ,mmdividerA   = wasm_f64x2_splat(0.0)
          ,mmdividerB   = wasm_f64x2_splat(0.0)
          ,mmsimilarity = wasm_f64x2_splat(0.0);

    for( ; i + 1 < vector_size; i += 2 ) {
        A = wasm_v128_load(&searched_array[i]);
        B = wasm_v128_load(&column_array[i]);
        AA = wasm_f64x2_mul(A, A);
        BB = wasm_f64x2_mul(B, B);
        AB = wasm_f64x2_mul(A, B);
        mmdividerA = wasm_f64x2_add(AA, mmdividerA);
        mmdividerB = wasm_f64x2_add(BB, mmdividerB);
        mmsimilarity = wasm_f64x2_add(AB, mmsimilarity);
    }//endfor i+2

    dividerA   = wasm_f64x2_extract_lane(mmdividerA, 0)   + wasm_f64x2_extract_lane(mmdividerA, 1);
    dividerB   = wasm_f64x2_extract_lane(mmdividerB, 0)   + wasm_f64x2_extract_lane(mmdividerB, 1);
    similarity = wasm_f64x2_extract_lane(mmsimilarity, 0) + wasm_f64x2_extract_lane(mmsimilarity, 1);

    // Calculate the remaining elements.
    for(; i < vector_size; ++i ) {
        double A = searched_array[i];
        double B = column_array[i];
        similarity += (A*B);
        dividerA   += (A*A);
        dividerB   += (B*B);
    }

    *divider_a = dividerA;
    *divider_b = dividerB;
    return similarity;
}


//----------------------------------------------------------------------------------------
// Name: euclidean_distance_similarity_f_wasmsimd
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of floats.
//       WASM SIMD128 version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB,
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
float euclidean_distance_similarity_f_wasmsimd( const float* searched_array
                                                ,const float* column_array
                                                ,const int    vector_size )
{
    float similarity = 0.0f;
    int i = 0;
    // WASM SIMD128 can handle 4 floats at a time.
    v128_t A, B, AB, ABAB, sumAB = wasm_f32x4_splat(0.0f);
    for( ; i + 3 < vector_size; i += 4 ) {
        A = wasm_v128_load(&searched_array[i]);
        B = wasm_v128_load(&column_array[i]);
        AB = wasm_f32x4_sub(A, B);

        ABAB = wasm_f32x4_mul(AB, AB);
        sumAB = wasm_f32x4_add(ABAB, sumAB);
    }//endfor i+4

    similarity = wasm_f32x4_extract_lane(sumAB, 0) + wasm_f32x4_extract_lane(sumAB, 1)
               + wasm_f32x4_extract_lane(sumAB, 2) + wasm_f32x4_extract_lane(sumAB, 3);

    // Handle the remaining elements.
    for( ; i < vector_size; ++i ) {
        float AB = (searched_array[i] - column_array[i]);
        similarity += (AB * AB);
    }

    return similarity;
}


//----------------------------------------------------------------------------------------
// Name: euclidean_distance_similarity_d_wasmsimd
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of doubles.
//       WASM SIMD128 version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB,
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
double euclidean_distance_similarity_d_wasmsimd( const double* searched_array
                                                 ,const double* column_array
                                                 ,const int     vector_size )
{
    double similarity = 0.0;
    int i = 0;
    // WASM SIMD128 can handle 2 doubles at a time.
    v128_t A, B, AB, ABAB, sumAB = wasm_f64x2_splat(0.0);
    for( ; i + 1 < vector_size; i += 2 ) {
        A = wasm_v128_load(&searched_array[i]);
        B = wasm_v128_load(&column_array[i]);
        AB = wasm_f64x2_sub(A, B);

        ABAB = wasm_f64x2_mul(AB, AB);
        sumAB = wasm_f64x2_add(ABAB, sumAB);
    }//endfor i+2

    similarity = wasm_f64x2_extract_lane(sumAB, 0) + wasm_f64x2_extract_lane(sumAB, 1);

    // Handle the remaining elements.
    for( ; i < vector_size; ++i ) {
        double AB = (searched_array[i] - column_array[i]);
        similarity += (AB * AB);
    }

    return similarity;
}


//----------------------------------------------------------------------------------------
// Name: dot_product_similarity_f_wasmsimd
// Desc: Calculates the dot product similarity to a BLOB-converted array of floats.
//       WASM SIMD128 version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB,
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product FLOAT
//----------------------------------------------------------------------------------------
float dot_product_similarity_f_wasmsimd( const float* searched_array
                                        ,const float* column_array
                                        ,const int    vector_size )
{
    float similarity = 0.0f;
    int i = 0;
    v128_t A, B, AB, sumAB = wasm_f32x4_splat(0.0f);
    for( ; i + 3 < vector_size; i += 4 ) {
        A = wasm_v128_load(&searched_array[i]);
        B = wasm_v128_load(&column_array[i]);
        AB = wasm_f32x4_mul(A, B);
        sumAB = wasm_f32x4_add(AB, sumAB);
    }// endfor i+4

    similarity = wasm_f32x4_extract_lane(sumAB, 0) + wasm_f32x4_extract_lane(sumAB, 1)
               + wasm_f32x4_extract_lane(sumAB, 2) + wasm_f32x4_extract_lane(sumAB, 3);

    for( ; i < vector_size; ++i ) {
        similarity += ((searched_array[i]) * (column_array[i]));
    }

    return similarity;
}


//----------------------------------------------------------------------------------------
// Name: dot_product_similarity_d_wasmsimd
// Desc: Calculates the dot product similarity to a BLOB-converted array of doubles.
//       WASM SIMD128 version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB,
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product DOUBLE
//----------------------------------------------------------------------------------------
double dot_product_similarity_d_wasmsimd(  const double* searched_array
                                          ,const double* column_array
                                          ,const int     vector_size )
{
    double similarity = 0.0;
    int i = 0;
    v128_t A, B, AB, sumAB = wasm_f64x2_splat(0.0);
    for( ; i + 1 < vector_size; i += 2 ) {
        A = wasm_v128_load(&searched_array[i]);
        B = wasm_v128_load(&column_array[i]);
        AB = wasm_f64x2_mul(A, B);
        sumAB = wasm_f64x2_add(AB, sumAB);
    }//endfor i + 2

    similarity = wasm_f64x2_extract_lane(sumAB, 0) + wasm_f64x2_extract_lane(sumAB, 1);

    for( ; i < vector_size; ++i ) {
        similarity += ((searched_array[i]) * (column_array[i]));
    }

    return similarity;
}


#endif
