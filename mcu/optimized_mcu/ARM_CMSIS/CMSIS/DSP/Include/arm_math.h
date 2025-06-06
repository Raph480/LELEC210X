/******************************************************************************
 * @file     arm_math.h
 * @brief    Public header file for CMSIS DSP Library
 * @version  V1.7.0
 * @date     18. March 2019
 ******************************************************************************/
/*
 * Copyright (c) 2010-2019 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
   \mainpage CMSIS DSP Software Library
   *
   * Introduction
   * ------------
   *
   * This user manual describes the CMSIS DSP software library,
   * a suite of common signal processing functions for use on Cortex-M and Cortex-A processor 
   * based devices.
   *
   * The library is divided into a number of functions each covering a specific category:
   * - Basic math functions
   * - Fast math functions
   * - Complex math functions
   * - Filtering functions
   * - Matrix functions
   * - Transform functions
   * - Motor control functions
   * - Statistical functions
   * - Support functions
   * - Interpolation functions
   * - Support Vector Machine functions (SVM)
   * - Bayes classifier functions
   * - Distance functions
   *
   * The library has generally separate functions for operating on 8-bit integers, 16-bit integers,
   * 32-bit integer and 32-bit floating-point values.
   *
   * Using the Library
   * ------------
   *
   * The library installer contains prebuilt versions of the libraries in the <code>Lib</code> folder.
   *
   * Here is the list of pre-built libraries :
   * - arm_cortexM7lfdp_math.lib (Cortex-M7, Little endian, Double Precision Floating Point Unit)
   * - arm_cortexM7bfdp_math.lib (Cortex-M7, Big endian, Double Precision Floating Point Unit)
   * - arm_cortexM7lfsp_math.lib (Cortex-M7, Little endian, Single Precision Floating Point Unit)
   * - arm_cortexM7bfsp_math.lib (Cortex-M7, Big endian and Single Precision Floating Point Unit on)
   * - arm_cortexM7l_math.lib (Cortex-M7, Little endian)
   * - arm_cortexM7b_math.lib (Cortex-M7, Big endian)
   * - arm_cortexM4lf_math.lib (Cortex-M4, Little endian, Floating Point Unit)
   * - arm_cortexM4bf_math.lib (Cortex-M4, Big endian, Floating Point Unit)
   * - arm_cortexM4l_math.lib (Cortex-M4, Little endian)
   * - arm_cortexM4b_math.lib (Cortex-M4, Big endian)
   * - arm_cortexM3l_math.lib (Cortex-M3, Little endian)
   * - arm_cortexM3b_math.lib (Cortex-M3, Big endian)
   * - arm_cortexM0l_math.lib (Cortex-M0 / Cortex-M0+, Little endian)
   * - arm_cortexM0b_math.lib (Cortex-M0 / Cortex-M0+, Big endian)
   * - arm_ARMv8MBLl_math.lib (Armv8-M Baseline, Little endian)
   * - arm_ARMv8MMLl_math.lib (Armv8-M Mainline, Little endian)
   * - arm_ARMv8MMLlfsp_math.lib (Armv8-M Mainline, Little endian, Single Precision Floating Point Unit)
   * - arm_ARMv8MMLld_math.lib (Armv8-M Mainline, Little endian, DSP instructions)
   * - arm_ARMv8MMLldfsp_math.lib (Armv8-M Mainline, Little endian, DSP instructions, Single Precision Floating Point Unit)
   *
   * The library functions are declared in the public file <code>arm_math.h</code> which is placed in the <code>Include</code> folder.
   * Simply include this file and link the appropriate library in the application and begin calling the library functions. The Library supports single
   * public header file <code> arm_math.h</code> for Cortex-M cores with little endian and big endian. Same header file will be used for floating point unit(FPU) variants.
   *
   *
   * Examples
   * --------
   *
   * The library ships with a number of examples which demonstrate how to use the library functions.
   *
   * Toolchain Support
   * ------------
   *
   * The library is now tested on Fast Models building with cmake.
   * Core M0, M7, A5 are tested.
   * 
   * 
   *
   * Building the Library
   * ------------
   *
   * The library installer contains a project file to rebuild libraries on MDK toolchain in the <code>CMSIS\\DSP\\Projects\\ARM</code> folder.
   * - arm_cortexM_math.uvprojx
   *
   *
   * The libraries can be built by opening the arm_cortexM_math.uvprojx project in MDK-ARM, selecting a specific target, and defining the optional preprocessor macros detailed above.
   *
   * There is also a work in progress cmake build. The README file is giving more details.
   *
   * Preprocessor Macros
   * ------------
   *
   * Each library project have different preprocessor macros.
   *
   * - ARM_MATH_BIG_ENDIAN:
   *
   * Define macro ARM_MATH_BIG_ENDIAN to build the library for big endian targets. By default library builds for little endian targets.
   *
   * - ARM_MATH_MATRIX_CHECK:
   *
   * Define macro ARM_MATH_MATRIX_CHECK for checking on the input and output sizes of matrices
   *
   * - ARM_MATH_ROUNDING:
   *
   * Define macro ARM_MATH_ROUNDING for rounding on support functions
   *
   * - ARM_MATH_LOOPUNROLL:
   *
   * Define macro ARM_MATH_LOOPUNROLL to enable manual loop unrolling in DSP functions
   *
   * - ARM_MATH_NEON:
   *
   * Define macro ARM_MATH_NEON to enable Neon versions of the DSP functions.
   * It is not enabled by default when Neon is available because performances are 
   * dependent on the compiler and target architecture.
   *
   * - ARM_MATH_NEON_EXPERIMENTAL:
   *
   * Define macro ARM_MATH_NEON_EXPERIMENTAL to enable experimental Neon versions of 
   * of some DSP functions. Experimental Neon versions currently do not have better
   * performances than the scalar versions.
   *
   * - ARM_MATH_HELIUM:
   *
   * It implies the flags ARM_MATH_MVEF and ARM_MATH_MVEI and ARM_MATH_FLOAT16.
   *
   * - ARM_MATH_MVEF:
   *
   * Select Helium versions of the f32 algorithms.
   * It implies ARM_MATH_FLOAT16 and ARM_MATH_MVEI.
   *
   * - ARM_MATH_MVEI:
   *
   * Select Helium versions of the int and fixed point algorithms.
   *
   * - ARM_MATH_FLOAT16:
   *
   * Float16 implementations of some algorithms (Requires MVE extension).
   *
   * <hr>
   * CMSIS-DSP in ARM::CMSIS Pack
   * -----------------------------
   *
   * The following files relevant to CMSIS-DSP are present in the <b>ARM::CMSIS</b> Pack directories:
   * |File/Folder                      |Content                                                                 |
   * |---------------------------------|------------------------------------------------------------------------|
   * |\b CMSIS\\Documentation\\DSP     | This documentation                                                     |
   * |\b CMSIS\\DSP\\DSP_Lib_TestSuite | DSP_Lib test suite                                                     |
   * |\b CMSIS\\DSP\\Examples          | Example projects demonstrating the usage of the library functions      |
   * |\b CMSIS\\DSP\\Include           | DSP_Lib include files                                                  |
   * |\b CMSIS\\DSP\\Lib               | DSP_Lib binaries                                                       |
   * |\b CMSIS\\DSP\\Projects          | Projects to rebuild DSP_Lib binaries                                   |
   * |\b CMSIS\\DSP\\Source            | DSP_Lib source files                                                   |
   *
   * <hr>
   * Revision History of CMSIS-DSP
   * ------------
   * Please refer to \ref ChangeLog_pg.
   */


/**
 * @defgroup groupMath Basic Math Functions
 */

/**
 * @defgroup groupFastMath Fast Math Functions
 * This set of functions provides a fast approximation to sine, cosine, and square root.
 * As compared to most of the other functions in the CMSIS math library, the fast math functions
 * operate on individual values and not arrays.
 * There are separate functions for Q15, Q31, and floating-point data.
 *
 */

/**
 * @defgroup groupCmplxMath Complex Math Functions
 * This set of functions operates on complex data vectors.
 * The data in the complex arrays is stored in an interleaved fashion
 * (real, imag, real, imag, ...).
 * In the API functions, the number of samples in a complex array refers
 * to the number of complex values; the array contains twice this number of
 * real values.
 */

/**
 * @defgroup groupFilters Filtering Functions
 */

/**
 * @defgroup groupMatrix Matrix Functions
 *
 * This set of functions provides basic matrix math operations.
 * The functions operate on matrix data structures.  For example,
 * the type
 * definition for the floating-point matrix structure is shown
 * below:
 * <pre>
 *     typedef struct
 *     {
 *       uint16_t numRows;     // number of rows of the matrix.
 *       uint16_t numCols;     // number of columns of the matrix.
 *       float32_t *pData;     // points to the data of the matrix.
 *     } arm_matrix_instance_f32;
 * </pre>
 * There are similar definitions for Q15 and Q31 data types.
 *
 * The structure specifies the size of the matrix and then points to
 * an array of data.  The array is of size <code>numRows X numCols</code>
 * and the values are arranged in row order.  That is, the
 * matrix element (i, j) is stored at:
 * <pre>
 *     pData[i*numCols + j]
 * </pre>
 *
 * \par Init Functions
 * There is an associated initialization function for each type of matrix
 * data structure.
 * The initialization function sets the values of the internal structure fields.
 * Refer to \ref arm_mat_init_f32(), \ref arm_mat_init_q31() and \ref arm_mat_init_q15()
 * for floating-point, Q31 and Q15 types,  respectively.
 *
 * \par
 * Use of the initialization function is optional. However, if initialization function is used
 * then the instance structure cannot be placed into a const data section.
 * To place the instance structure in a const data
 * section, manually initialize the data structure.  For example:
 * <pre>
 * <code>arm_matrix_instance_f32 S = {nRows, nColumns, pData};</code>
 * <code>arm_matrix_instance_q31 S = {nRows, nColumns, pData};</code>
 * <code>arm_matrix_instance_q15 S = {nRows, nColumns, pData};</code>
 * </pre>
 * where <code>nRows</code> specifies the number of rows, <code>nColumns</code>
 * specifies the number of columns, and <code>pData</code> points to the
 * data array.
 *
 * \par Size Checking
 * By default all of the matrix functions perform size checking on the input and
 * output matrices. For example, the matrix addition function verifies that the
 * two input matrices and the output matrix all have the same number of rows and
 * columns. If the size check fails the functions return:
 * <pre>
 *     ARM_MATH_SIZE_MISMATCH
 * </pre>
 * Otherwise the functions return
 * <pre>
 *     ARM_MATH_SUCCESS
 * </pre>
 * There is some overhead associated with this matrix size checking.
 * The matrix size checking is enabled via the \#define
 * <pre>
 *     ARM_MATH_MATRIX_CHECK
 * </pre>
 * within the library project settings.  By default this macro is defined
 * and size checking is enabled. By changing the project settings and
 * undefining this macro size checking is eliminated and the functions
 * run a bit faster. With size checking disabled the functions always
 * return <code>ARM_MATH_SUCCESS</code>.
 */

/**
 * @defgroup groupTransforms Transform Functions
 */

/**
 * @defgroup groupController Controller Functions
 */

/**
 * @defgroup groupStats Statistics Functions
 */

/**
 * @defgroup groupSupport Support Functions
 */

/**
 * @defgroup groupInterpolation Interpolation Functions
 * These functions perform 1- and 2-dimensional interpolation of data.
 * Linear interpolation is used for 1-dimensional data and
 * bilinear interpolation is used for 2-dimensional data.
 */

/**
 * @defgroup groupExamples Examples
 */

/**
 * @defgroup groupSVM SVM Functions
 * This set of functions is implementing SVM classification on 2 classes.
 * The training must be done from scikit-learn. The parameters can be easily
 * generated from the scikit-learn object. Some examples are given in
 * DSP/Testing/PatternGeneration/SVM.py
 *
 * If more than 2 classes are needed, the functions in this folder 
 * will have to be used, as building blocks, to do multi-class classification.
 *
 * No multi-class classification is provided in this SVM folder.
 * 
 */


/**
 * @defgroup groupBayes Bayesian estimators
 *
 * Implement the naive gaussian Bayes estimator.
 * The training must be done from scikit-learn.
 *
 * The parameters can be easily
 * generated from the scikit-learn object. Some examples are given in
 * DSP/Testing/PatternGeneration/Bayes.py
 */

/**
 * @defgroup groupDistance Distance functions
 *
 * Distance functions for use with clustering algorithms.
 * There are distance functions for float vectors and boolean vectors.
 *
 */


#ifndef _ARM_MATH_H
#define _ARM_MATH_H

#ifdef   __cplusplus
extern "C"
{
#endif

/* Compiler specific diagnostic adjustment */
#if   defined ( __CC_ARM )

#elif defined ( __ARMCC_VERSION ) && ( __ARMCC_VERSION >= 6010050 )

#elif defined ( __GNUC__ )
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wsign-conversion"
  #pragma GCC diagnostic ignored "-Wconversion"
  #pragma GCC diagnostic ignored "-Wunused-parameter"

#elif defined ( __ICCARM__ )

#elif defined ( __TI_ARM__ )

#elif defined ( __CSMC__ )

#elif defined ( __TASKING__ )

#elif defined ( _MSC_VER )

#else
  #error Unknown compiler
#endif


/* Included for instrinsics definitions */
#if defined (_MSC_VER ) 
#include <stdint.h>
#define __STATIC_FORCEINLINE static __forceinline
#define __STATIC_INLINE static __inline
#define __ALIGNED(x) __declspec(align(x))

#elif defined (__GNUC_PYTHON__)
#include <stdint.h>
#define  __ALIGNED(x) __attribute__((aligned(x)))
#define __STATIC_FORCEINLINE static __attribute__((inline))
#define __STATIC_INLINE static __attribute__((inline))
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wattributes"

#else
#include "cmsis_compiler.h"
#endif



#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>


#define F64_MAX   ((float64_t)DBL_MAX)
#define F32_MAX   ((float32_t)FLT_MAX)

#if defined(ARM_MATH_FLOAT16)
#define F16_MAX   ((float16_t)FLT_MAX)
#endif

#define F64_MIN   (-DBL_MAX)
#define F32_MIN   (-FLT_MAX)

#if defined(ARM_MATH_FLOAT16)
#define F16_MIN   (-(float16_t)FLT_MAX)
#endif

#define F64_ABSMAX   ((float64_t)DBL_MAX)
#define F32_ABSMAX   ((float32_t)FLT_MAX)

#if defined(ARM_MATH_FLOAT16)
#define F16_ABSMAX   ((float16_t)FLT_MAX)
#endif

#define F64_ABSMIN   ((float64_t)0.0)
#define F32_ABSMIN   ((float32_t)0.0)

#if defined(ARM_MATH_FLOAT16)
#define F16_ABSMIN   ((float16_t)0.0)
#endif

#define Q31_MAX   ((q31_t)(0x7FFFFFFFL))
#define Q15_MAX   ((q15_t)(0x7FFF))
#define Q7_MAX    ((q7_t)(0x7F))
#define Q31_MIN   ((q31_t)(0x80000000L))
#define Q15_MIN   ((q15_t)(0x8000))
#define Q7_MIN    ((q7_t)(0x80))

#define Q31_ABSMAX   ((q31_t)(0x7FFFFFFFL))
#define Q15_ABSMAX   ((q15_t)(0x7FFF))
#define Q7_ABSMAX    ((q7_t)(0x7F))
#define Q31_ABSMIN   ((q31_t)0)
#define Q15_ABSMIN   ((q15_t)0)
#define Q7_ABSMIN    ((q7_t)0)

/* evaluate ARM DSP feature */
#if (defined (__ARM_FEATURE_DSP) && (__ARM_FEATURE_DSP == 1))
  #define ARM_MATH_DSP                   1
#endif

#if defined(ARM_MATH_NEON)
#include <arm_neon.h>
#endif

#if defined (ARM_MATH_HELIUM)
  #define ARM_MATH_MVEF
  #define ARM_MATH_FLOAT16
#endif

#if defined (ARM_MATH_MVEF)
  #define ARM_MATH_MVEI
  #define ARM_MATH_FLOAT16
#endif

#if defined (ARM_MATH_HELIUM) || defined(ARM_MATH_MVEF) || defined(ARM_MATH_MVEI)
#include <arm_mve.h>
#endif


  /**
   * @brief Macros required for reciprocal calculation in Normalized LMS
   */

#define DELTA_Q31          ((q31_t)(0x100))
#define DELTA_Q15          ((q15_t)0x5)
#define INDEX_MASK         0x0000003F
#ifndef PI
  #define PI               3.14159265358979f
#endif

  /**
   * @brief Macros required for SINE and COSINE Fast math approximations
   */

#define FAST_MATH_TABLE_SIZE  512
#define FAST_MATH_Q31_SHIFT   (32 - 10)
#define FAST_MATH_Q15_SHIFT   (16 - 10)
#define CONTROLLER_Q31_SHIFT  (32 - 9)
#define TABLE_SPACING_Q31     0x400000
#define TABLE_SPACING_Q15     0x80

  /**
   * @brief Macros required for SINE and COSINE Controller functions
   */
  /* 1.31(q31) Fixed value of 2/360 */
  /* -1 to +1 is divided into 360 values so total spacing is (2/360) */
#define INPUT_SPACING         0xB60B61

  /**
   * @brief Macros for complex numbers
   */

  /* Dimension C vector space */
  #define CMPLX_DIM 2

  /**
   * @brief Error status returned by some functions in the library.
   */

  typedef enum
  {
    ARM_MATH_SUCCESS        =  0,        /**< No error */
    ARM_MATH_ARGUMENT_ERROR = -1,        /**< One or more arguments are incorrect */
    ARM_MATH_LENGTH_ERROR   = -2,        /**< Length of data buffer is incorrect */
    ARM_MATH_SIZE_MISMATCH  = -3,        /**< Size of matrices is not compatible with the operation */
    ARM_MATH_NANINF         = -4,        /**< Not-a-number (NaN) or infinity is generated */
    ARM_MATH_SINGULAR       = -5,        /**< Input matrix is singular and cannot be inverted */
    ARM_MATH_TEST_FAILURE   = -6         /**< Test Failed */
  } arm_status;

  /**
   * @brief 8-bit fractional data type in 1.7 format.
   */
  typedef int8_t q7_t;

  /**
   * @brief 16-bit fractional data type in 1.15 format.
   */
  typedef int16_t q15_t;

  /**
   * @brief 32-bit fractional data type in 1.31 format.
   */
  typedef int32_t q31_t;

  /**
   * @brief 64-bit fractional data type in 1.63 format.
   */
  typedef int64_t q63_t;

  /**
   * @brief 32-bit floating-point type definition.
   */
  typedef float float32_t;

  /**
   * @brief 64-bit floating-point type definition.
   */
  typedef double float64_t;

  /**
   * @brief vector types
   */
#if defined(ARM_MATH_NEON) || defined (ARM_MATH_MVEI)
  /**
   * @brief 64-bit fractional 128-bit vector data type in 1.63 format
   */
  typedef int64x2_t q63x2_t;

  /**
   * @brief 32-bit fractional 128-bit vector data type in 1.31 format.
   */
  typedef int32x4_t q31x4_t;

  /**
   * @brief 16-bit fractional 128-bit vector data type with 16-bit alignement in 1.15 format.
   */
  typedef __ALIGNED(2) int16x8_t q15x8_t;

 /**
   * @brief 8-bit fractional 128-bit vector data type with 8-bit alignement in 1.7 format.
   */
  typedef __ALIGNED(1) int8x16_t q7x16_t;

    /**
   * @brief 32-bit fractional 128-bit vector pair data type in 1.31 format.
   */
  typedef int32x4x2_t q31x4x2_t;

  /**
   * @brief 32-bit fractional 128-bit vector quadruplet data type in 1.31 format.
   */
  typedef int32x4x4_t q31x4x4_t;

  /**
   * @brief 16-bit fractional 128-bit vector pair data type in 1.15 format.
   */
  typedef int16x8x2_t q15x8x2_t;

  /**
   * @brief 16-bit fractional 128-bit vector quadruplet data type in 1.15 format.
   */
  typedef int16x8x4_t q15x8x4_t;

  /**
   * @brief 8-bit fractional 128-bit vector pair data type in 1.7 format.
   */
  typedef int8x16x2_t q7x16x2_t;

  /**
   * @brief 8-bit fractional 128-bit vector quadruplet data type in 1.7 format.
   */
   typedef int8x16x4_t q7x16x4_t;

  /**
   * @brief 32-bit fractional data type in 9.23 format.
   */
  typedef int32_t q23_t;

  /**
   * @brief 32-bit fractional 128-bit vector data type in 9.23 format.
   */
  typedef int32x4_t q23x4_t;

  /**
   * @brief 64-bit status 128-bit vector data type.
   */
  typedef int64x2_t status64x2_t;

  /**
   * @brief 32-bit status 128-bit vector data type.
   */
  typedef int32x4_t status32x4_t;

  /**
   * @brief 16-bit status 128-bit vector data type.
   */
  typedef int16x8_t status16x8_t;

  /**
   * @brief 8-bit status 128-bit vector data type.
   */
  typedef int8x16_t status8x16_t;


#endif

#if defined(ARM_MATH_NEON) || defined(ARM_MATH_MVEF) /* floating point vector*/
  /**
   * @brief 32-bit floating-point 128-bit vector type
   */
  typedef float32x4_t f32x4_t;

#if defined(ARM_MATH_FLOAT16)
  /**
   * @brief 16-bit floating-point 128-bit vector data type
   */
  typedef __ALIGNED(2) float16x8_t f16x8_t;
#endif

  /**
   * @brief 32-bit floating-point 128-bit vector pair data type
   */
  typedef float32x4x2_t f32x4x2_t;

  /**
   * @brief 32-bit floating-point 128-bit vector quadruplet data type
   */
  typedef float32x4x4_t f32x4x4_t;

#if defined(ARM_MATH_FLOAT16)
  /**
   * @brief 16-bit floating-point 128-bit vector pair data type
   */
  typedef float16x8x2_t f16x8x2_t;

  /**
   * @brief 16-bit floating-point 128-bit vector quadruplet data type
   */
  typedef float16x8x4_t f16x8x4_t;
#endif

  /**
   * @brief 32-bit ubiquitous 128-bit vector data type
   */
  typedef union _any32x4_t
  {
      float32x4_t     f;
      int32x4_t       i;
  } any32x4_t;

#if defined(ARM_MATH_FLOAT16)
  /**
   * @brief 16-bit ubiquitous 128-bit vector data type
   */
  typedef union _any16x8_t
  {
      float16x8_t     f;
      int16x8_t       i;
  } any16x8_t;
#endif

#endif

#if defined(ARM_MATH_NEON)
  /**
   * @brief 32-bit fractional 64-bit vector data type in 1.31 format.
   */
  typedef int32x2_t  q31x2_t;

  /**
   * @brief 16-bit fractional 64-bit vector data type in 1.15 format.
   */
  typedef  __ALIGNED(2) int16x4_t q15x4_t;

  /**
   * @brief 8-bit fractional 64-bit vector data type in 1.7 format.
   */
  typedef  __ALIGNED(1) int8x8_t q7x8_t;

  /**
   * @brief 32-bit float 64-bit vector data type.
   */
  typedef float32x2_t  f32x2_t;

#if defined(ARM_MATH_FLOAT16)
  /**
   * @brief 16-bit float 64-bit vector data type.
   */
  typedef  __ALIGNED(2) float16x4_t f16x4_t;
#endif 

  /**
   * @brief 32-bit floating-point 128-bit vector triplet data type
   */
  typedef float32x4x3_t f32x4x3_t;

#if defined(ARM_MATH_FLOAT16)
  /**
   * @brief 16-bit floating-point 128-bit vector triplet data type
   */
  typedef float16x8x3_t f16x8x3_t;
#endif

  /**
   * @brief 32-bit fractional 128-bit vector triplet data type in 1.31 format
   */
  typedef int32x4x3_t q31x4x3_t;

  /**
   * @brief 16-bit fractional 128-bit vector triplet data type in 1.15 format
   */
  typedef int16x8x3_t q15x8x3_t;

  /**
   * @brief 8-bit fractional 128-bit vector triplet data type in 1.7 format
   */
  typedef int8x16x3_t q7x16x3_t;

  /**
   * @brief 32-bit floating-point 64-bit vector pair data type
   */
  typedef float32x2x2_t f32x2x2_t;

  /**
   * @brief 32-bit floating-point 64-bit vector triplet data type
   */
  typedef float32x2x3_t f32x2x3_t;

  /**
   * @brief 32-bit floating-point 64-bit vector quadruplet data type
   */
  typedef float32x2x4_t f32x2x4_t;

#if defined(ARM_MATH_FLOAT16)
  /**
   * @brief 16-bit floating-point 64-bit vector pair data type
   */
  typedef float16x4x2_t f16x4x2_t;

  /**
   * @brief 16-bit floating-point 64-bit vector triplet data type
   */
  typedef float16x4x3_t f16x4x3_t;

  /**
   * @brief 16-bit floating-point 64-bit vector quadruplet data type
   */
  typedef float16x4x4_t f16x4x4_t;
#endif 

  /**
   * @brief 32-bit fractional 64-bit vector pair data type in 1.31 format
   */
  typedef int32x2x2_t q31x2x2_t;

  /**
   * @brief 32-bit fractional 64-bit vector triplet data type in 1.31 format
   */
  typedef int32x2x3_t q31x2x3_t;

  /**
   * @brief 32-bit fractional 64-bit vector quadruplet data type in 1.31 format
   */
  typedef int32x4x3_t q31x2x4_t;

  /**
   * @brief 16-bit fractional 64-bit vector pair data type in 1.15 format
   */
  typedef int16x4x2_t q15x4x2_t;

  /**
   * @brief 16-bit fractional 64-bit vector triplet data type in 1.15 format
   */
  typedef int16x4x2_t q15x4x3_t;

  /**
   * @brief 16-bit fractional 64-bit vector quadruplet data type in 1.15 format
   */
  typedef int16x4x3_t q15x4x4_t;

  /**
   * @brief 8-bit fractional 64-bit vector pair data type in 1.7 format
   */
  typedef int8x8x2_t q7x8x2_t;

  /**
   * @brief 8-bit fractional 64-bit vector triplet data type in 1.7 format
   */
  typedef int8x8x3_t q7x8x3_t;

  /**
   * @brief 8-bit fractional 64-bit vector quadruplet data type in 1.7 format
   */
  typedef int8x8x4_t q7x8x4_t;

  /**
   * @brief 32-bit ubiquitous 64-bit vector data type
   */
  typedef union _any32x2_t
  {
      float32x2_t     f;
      int32x2_t       i;
  } any32x2_t;

#if defined(ARM_MATH_FLOAT16)
  /**
   * @brief 16-bit ubiquitous 64-bit vector data type
   */
  typedef union _any16x4_t
  {
      float16x4_t     f;
      int16x4_t       i;
  } any16x4_t;
#endif 

  /**
   * @brief 32-bit status 64-bit vector data type.
   */
  typedef int32x4_t status32x2_t;

  /**
   * @brief 16-bit status 64-bit vector data type.
   */
  typedef int16x8_t status16x4_t;

  /**
   * @brief 8-bit status 64-bit vector data type.
   */
  typedef int8x16_t status8x8_t;

#endif



/**
  @brief definition to read/write two 16 bit values.
  @deprecated
 */
#if   defined ( __CC_ARM )
  #define __SIMD32_TYPE int32_t __packed
#elif defined ( __ARMCC_VERSION ) && ( __ARMCC_VERSION >= 6010050 )
  #define __SIMD32_TYPE int32_t
#elif defined ( __GNUC__ )
  #define __SIMD32_TYPE int32_t
#elif defined ( __ICCARM__ )
  #define __SIMD32_TYPE int32_t __packed
#elif defined ( __TI_ARM__ )
  #define __SIMD32_TYPE int32_t
#elif defined ( __CSMC__ )
  #define __SIMD32_TYPE int32_t
#elif defined ( __TASKING__ )
  #define __SIMD32_TYPE __un(aligned) int32_t
#elif defined(_MSC_VER )
  #define __SIMD32_TYPE int32_t
#else
  #error Unknown compiler
#endif

#define __SIMD32(addr)        (*(__SIMD32_TYPE **) & (addr))
#define __SIMD32_CONST(addr)  ( (__SIMD32_TYPE * )   (addr))
#define _SIMD32_OFFSET(addr)  (*(__SIMD32_TYPE * )   (addr))
#define __SIMD64(addr)        (*(      int64_t **) & (addr))

#define STEP(x) (x) <= 0 ? 0 : 1
#define SQ(x) ((x) * (x))

/* SIMD replacement */


/**
  @brief         Read 2 Q15 from Q15 pointer.
  @param[in]     pQ15      points to input value
  @return        Q31 value
 */
__STATIC_FORCEINLINE q31_t read_q15x2 (
  q15_t * pQ15)
{
  q31_t val;

#ifdef __ARM_FEATURE_UNALIGNED
  memcpy (&val, pQ15, 4);
#else
  val = (pQ15[1] << 16) | (pQ15[0] & 0x0FFFF) ;
#endif

  return (val);
}

/**
  @brief         Read 2 Q15 from Q15 pointer and increment pointer afterwards.
  @param[in]     pQ15      points to input value
  @return        Q31 value
 */
__STATIC_FORCEINLINE q31_t read_q15x2_ia (
  q15_t ** pQ15)
{
  q31_t val;

#ifdef __ARM_FEATURE_UNALIGNED
  memcpy (&val, *pQ15, 4);
#else
  val = ((*pQ15)[1] << 16) | ((*pQ15)[0] & 0x0FFFF);
#endif

 *pQ15 += 2;
 return (val);
}

/**
  @brief         Read 2 Q15 from Q15 pointer and decrement pointer afterwards.
  @param[in]     pQ15      points to input value
  @return        Q31 value
 */
__STATIC_FORCEINLINE q31_t read_q15x2_da (
  q15_t ** pQ15)
{
  q31_t val;

#ifdef __ARM_FEATURE_UNALIGNED
  memcpy (&val, *pQ15, 4);
#else
  val = ((*pQ15)[1] << 16) | ((*pQ15)[0] & 0x0FFFF);
#endif

  *pQ15 -= 2;
  return (val);
}

/**
  @brief         Write 2 Q15 to Q15 pointer and increment pointer afterwards.
  @param[in]     pQ15      points to input value
  @param[in]     value     Q31 value
  @return        none
 */
__STATIC_FORCEINLINE void write_q15x2_ia (
  q15_t ** pQ15,
  q31_t    value)
{
  q31_t val = value;
#ifdef __ARM_FEATURE_UNALIGNED
  memcpy (*pQ15, &val, 4);
#else
  (*pQ15)[0] = (val & 0x0FFFF);
  (*pQ15)[1] = (val >> 16) & 0x0FFFF;
#endif

 *pQ15 += 2;
}

/**
  @brief         Write 2 Q15 to Q15 pointer.
  @param[in]     pQ15      points to input value
  @param[in]     value     Q31 value
  @return        none
 */
__STATIC_FORCEINLINE void write_q15x2 (
  q15_t * pQ15,
  q31_t   value)
{
  q31_t val = value;

#ifdef __ARM_FEATURE_UNALIGNED
  memcpy (pQ15, &val, 4);
#else
  pQ15[0] = val & 0x0FFFF;
  pQ15[1] = val >> 16;
#endif
}


/**
  @brief         Read 4 Q7 from Q7 pointer and increment pointer afterwards.
  @param[in]     pQ7       points to input value
  @return        Q31 value
 */
__STATIC_FORCEINLINE q31_t read_q7x4_ia (
  q7_t ** pQ7)
{
  q31_t val;


#ifdef __ARM_FEATURE_UNALIGNED
  memcpy (&val, *pQ7, 4);
#else
  val =(((*pQ7)[3] & 0x0FF) << 24)  | (((*pQ7)[2] & 0x0FF) << 16)  | (((*pQ7)[1] & 0x0FF) << 8)  | ((*pQ7)[0] & 0x0FF);
#endif 

  *pQ7 += 4;

  return (val);
}

/**
  @brief         Read 4 Q7 from Q7 pointer and decrement pointer afterwards.
  @param[in]     pQ7       points to input value
  @return        Q31 value
 */
__STATIC_FORCEINLINE q31_t read_q7x4_da (
  q7_t ** pQ7)
{
  q31_t val;
#ifdef __ARM_FEATURE_UNALIGNED
  memcpy (&val, *pQ7, 4);
#else
  val = ((((*pQ7)[3]) & 0x0FF) << 24) | ((((*pQ7)[2]) & 0x0FF) << 16)   | ((((*pQ7)[1]) & 0x0FF) << 8)  | ((*pQ7)[0] & 0x0FF);
#endif 
  *pQ7 -= 4;

  return (val);
}

/**
  @brief         Write 4 Q7 to Q7 pointer and increment pointer afterwards.
  @param[in]     pQ7       points to input value
  @param[in]     value     Q31 value
  @return        none
 */
__STATIC_FORCEINLINE void write_q7x4_ia (
  q7_t ** pQ7,
  q31_t   value)
{
  q31_t val = value;
#ifdef __ARM_FEATURE_UNALIGNED
  memcpy (*pQ7, &val, 4);
#else
  (*pQ7)[0] = val & 0x0FF;
  (*pQ7)[1] = (val >> 8) & 0x0FF;
  (*pQ7)[2] = (val >> 16) & 0x0FF;
  (*pQ7)[3] = (val >> 24) & 0x0FF;

#endif
  *pQ7 += 4;
}

/*

Normally those kind of definitions are in a compiler file
in Core or Core_A.

But for MSVC compiler it is a bit special. The goal is very specific
to CMSIS-DSP and only to allow the use of this library from other
systems like Python or Matlab.

MSVC is not going to be used to cross-compile to ARM. So, having a MSVC
compiler file in Core or Core_A would not make sense.

*/
#if defined ( _MSC_VER ) || defined(__GNUC_PYTHON__)
    __STATIC_FORCEINLINE uint8_t __CLZ(uint32_t data)
    {
      if (data == 0U) { return 32U; }

      uint32_t count = 0U;
      uint32_t mask = 0x80000000U;

      while ((data & mask) == 0U)
      {
        count += 1U;
        mask = mask >> 1U;
      }
      return count;
    }

  __STATIC_FORCEINLINE int32_t __SSAT(int32_t val, uint32_t sat)
  {
    if ((sat >= 1U) && (sat <= 32U))
    {
      const int32_t max = (int32_t)((1U << (sat - 1U)) - 1U);
      const int32_t min = -1 - max ;
      if (val > max)
      {
        return max;
      }
      else if (val < min)
      {
        return min;
      }
    }
    return val;
  }

  __STATIC_FORCEINLINE uint32_t __USAT(int32_t val, uint32_t sat)
  {
    if (sat <= 31U)
    {
      const uint32_t max = ((1U << sat) - 1U);
      if (val > (int32_t)max)
      {
        return max;
      }
      else if (val < 0)
      {
        return 0U;
      }
    }
    return (uint32_t)val;
  }
#endif

#ifndef ARM_MATH_DSP
  /**
   * @brief definition to pack two 16 bit values.
   */
  #define __PKHBT(ARG1, ARG2, ARG3) ( (((int32_t)(ARG1) <<    0) & (int32_t)0x0000FFFF) | \
                                      (((int32_t)(ARG2) << ARG3) & (int32_t)0xFFFF0000)  )
  #define __PKHTB(ARG1, ARG2, ARG3) ( (((int32_t)(ARG1) <<    0) & (int32_t)0xFFFF0000) | \
                                      (((int32_t)(ARG2) >> ARG3) & (int32_t)0x0000FFFF)  )
#endif

   /**
   * @brief definition to pack four 8 bit values.
   */
#ifndef ARM_MATH_BIG_ENDIAN
  #define __PACKq7(v0,v1,v2,v3) ( (((int32_t)(v0) <<  0) & (int32_t)0x000000FF) | \
                                  (((int32_t)(v1) <<  8) & (int32_t)0x0000FF00) | \
                                  (((int32_t)(v2) << 16) & (int32_t)0x00FF0000) | \
                                  (((int32_t)(v3) << 24) & (int32_t)0xFF000000)  )
#else
  #define __PACKq7(v0,v1,v2,v3) ( (((int32_t)(v3) <<  0) & (int32_t)0x000000FF) | \
                                  (((int32_t)(v2) <<  8) & (int32_t)0x0000FF00) | \
                                  (((int32_t)(v1) << 16) & (int32_t)0x00FF0000) | \
                                  (((int32_t)(v0) << 24) & (int32_t)0xFF000000)  )
#endif


  /**
   * @brief Clips Q63 to Q31 values.
   */
  __STATIC_FORCEINLINE q31_t clip_q63_to_q31(
  q63_t x)
  {
    return ((q31_t) (x >> 32) != ((q31_t) x >> 31)) ?
      ((0x7FFFFFFF ^ ((q31_t) (x >> 63)))) : (q31_t) x;
  }

  /**
   * @brief Clips Q63 to Q15 values.
   */
  __STATIC_FORCEINLINE q15_t clip_q63_to_q15(
  q63_t x)
  {
    return ((q31_t) (x >> 32) != ((q31_t) x >> 31)) ?
      ((0x7FFF ^ ((q15_t) (x >> 63)))) : (q15_t) (x >> 15);
  }

  /**
   * @brief Clips Q31 to Q7 values.
   */
  __STATIC_FORCEINLINE q7_t clip_q31_to_q7(
  q31_t x)
  {
    return ((q31_t) (x >> 24) != ((q31_t) x >> 23)) ?
      ((0x7F ^ ((q7_t) (x >> 31)))) : (q7_t) x;
  }

  /**
   * @brief Clips Q31 to Q15 values.
   */
  __STATIC_FORCEINLINE q15_t clip_q31_to_q15(
  q31_t x)
  {
    return ((q31_t) (x >> 16) != ((q31_t) x >> 15)) ?
      ((0x7FFF ^ ((q15_t) (x >> 31)))) : (q15_t) x;
  }

  /**
   * @brief Multiplies 32 X 64 and returns 32 bit result in 2.30 format.
   */
  __STATIC_FORCEINLINE q63_t mult32x64(
  q63_t x,
  q31_t y)
  {
    return ((((q63_t) (x & 0x00000000FFFFFFFF) * y) >> 32) +
            (((q63_t) (x >> 32)                * y)      )  );
  }

  /**
   * @brief Function to Calculates 1/in (reciprocal) value of Q31 Data type.
   */
  __STATIC_FORCEINLINE uint32_t arm_recip_q31(
        q31_t in,
        q31_t * dst,
  const q31_t * pRecipTable)
  {
    q31_t out;
    uint32_t tempVal;
    uint32_t index, i;
    uint32_t signBits;

    if (in > 0)
    {
      signBits = ((uint32_t) (__CLZ( in) - 1));
    }
    else
    {
      signBits = ((uint32_t) (__CLZ(-in) - 1));
    }

    /* Convert input sample to 1.31 format */
    in = (in << signBits);

    /* calculation of index for initial approximated Val */
    index = (uint32_t)(in >> 24);
    index = (index & INDEX_MASK);

    /* 1.31 with exp 1 */
    out = pRecipTable[index];

    /* calculation of reciprocal value */
    /* running approximation for two iterations */
    for (i = 0U; i < 2U; i++)
    {
      tempVal = (uint32_t) (((q63_t) in * out) >> 31);
      tempVal = 0x7FFFFFFFu - tempVal;
      /*      1.31 with exp 1 */
      /* out = (q31_t) (((q63_t) out * tempVal) >> 30); */
      out = clip_q63_to_q31(((q63_t) out * tempVal) >> 30);
    }

    /* write output */
    *dst = out;

    /* return num of signbits of out = 1/in value */
    return (signBits + 1U);
  }


  /**
   * @brief Function to Calculates 1/in (reciprocal) value of Q15 Data type.
   */
  __STATIC_FORCEINLINE uint32_t arm_recip_q15(
        q15_t in,
        q15_t * dst,
  const q15_t * pRecipTable)
  {
    q15_t out = 0;
    uint32_t tempVal = 0;
    uint32_t index = 0, i = 0;
    uint32_t signBits = 0;

    if (in > 0)
    {
      signBits = ((uint32_t)(__CLZ( in) - 17));
    }
    else
    {
      signBits = ((uint32_t)(__CLZ(-in) - 17));
    }

    /* Convert input sample to 1.15 format */
    in = (in << signBits);

    /* calculation of index for initial approximated Val */
    index = (uint32_t)(in >>  8);
    index = (index & INDEX_MASK);

    /*      1.15 with exp 1  */
    out = pRecipTable[index];

    /* calculation of reciprocal value */
    /* running approximation for two iterations */
    for (i = 0U; i < 2U; i++)
    {
      tempVal = (uint32_t) (((q31_t) in * out) >> 15);
      tempVal = 0x7FFFu - tempVal;
      /*      1.15 with exp 1 */
      out = (q15_t) (((q31_t) out * tempVal) >> 14);
      /* out = clip_q31_to_q15(((q31_t) out * tempVal) >> 14); */
    }

    /* write output */
    *dst = out;

    /* return num of signbits of out = 1/in value */
    return (signBits + 1);
  }

/**
 * @brief Integer exponentiation
 * @param[in]    x           value
 * @param[in]    nb          integer exponent >= 1
 * @return x^nb
 *
 */
__STATIC_INLINE float32_t arm_exponent_f32(float32_t x, int32_t nb)
{
    float32_t r = x;
    nb --;
    while(nb > 0)
    {
        r = r * x;
        nb--;
    }
    return(r);
}

/**
 * @brief  64-bit to 32-bit unsigned normalization
 * @param[in]  in           is input unsigned long long value
 * @param[out] normalized   is the 32-bit normalized value
 * @param[out] norm         is norm scale
 */
__STATIC_INLINE  void arm_norm_64_to_32u(uint64_t in, int32_t * normalized, int32_t *norm)
{
    int32_t     n1;
    int32_t     hi = (int32_t) (in >> 32);
    int32_t     lo = (int32_t) ((in << 32) >> 32);

    n1 = __CLZ(hi) - 32;
    if (!n1)
    {
        /*
         * input fits in 32-bit
         */
        n1 = __CLZ(lo);
        if (!n1)
        {
            /*
             * MSB set, need to scale down by 1
             */
            *norm = -1;
            *normalized = (((uint32_t) lo) >> 1);
        } else
        {
            if (n1 == 32)
            {
                /*
                 * input is zero
                 */
                *norm = 0;
                *normalized = 0;
            } else
            {
                /*
                 * 32-bit normalization
                 */
                *norm = n1 - 1;
                *normalized = lo << *norm;
            }
        }
    } else
    {
        /*
         * input fits in 64-bit
         */
        n1 = 1 - n1;
        *norm = -n1;
        /*
         * 64 bit normalization
         */
        *normalized = (((uint32_t) lo) >> n1) | (hi << (32 - n1));
    }
}

__STATIC_INLINE q31_t arm_div_q63_to_q31(q63_t num, q31_t den)
{
    q31_t   result;
    uint64_t   absNum;
    int32_t   normalized;
    int32_t   norm;

    /*
     * if sum fits in 32bits
     * avoid costly 64-bit division
     */
    absNum = num > 0 ? num : -num;
    arm_norm_64_to_32u(absNum, &normalized, &norm);
    if (norm > 0)
        /*
         * 32-bit division
         */
        result = (q31_t) num / den;
    else
        /*
         * 64-bit division
         */
        result = (q31_t) (num / den);

    return result;
}


/*
 * @brief C custom defined intrinsic functions
 */
#if !defined (ARM_MATH_DSP)

  /*
   * @brief C custom defined QADD8
   */
  __STATIC_FORCEINLINE uint32_t __QADD8(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s, t, u;

    r = __SSAT(((((q31_t)x << 24) >> 24) + (((q31_t)y << 24) >> 24)), 8) & (int32_t)0x000000FF;
    s = __SSAT(((((q31_t)x << 16) >> 24) + (((q31_t)y << 16) >> 24)), 8) & (int32_t)0x000000FF;
    t = __SSAT(((((q31_t)x <<  8) >> 24) + (((q31_t)y <<  8) >> 24)), 8) & (int32_t)0x000000FF;
    u = __SSAT(((((q31_t)x      ) >> 24) + (((q31_t)y      ) >> 24)), 8) & (int32_t)0x000000FF;

    return ((uint32_t)((u << 24) | (t << 16) | (s <<  8) | (r      )));
  }


  /*
   * @brief C custom defined QSUB8
   */
  __STATIC_FORCEINLINE uint32_t __QSUB8(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s, t, u;

    r = __SSAT(((((q31_t)x << 24) >> 24) - (((q31_t)y << 24) >> 24)), 8) & (int32_t)0x000000FF;
    s = __SSAT(((((q31_t)x << 16) >> 24) - (((q31_t)y << 16) >> 24)), 8) & (int32_t)0x000000FF;
    t = __SSAT(((((q31_t)x <<  8) >> 24) - (((q31_t)y <<  8) >> 24)), 8) & (int32_t)0x000000FF;
    u = __SSAT(((((q31_t)x      ) >> 24) - (((q31_t)y      ) >> 24)), 8) & (int32_t)0x000000FF;

    return ((uint32_t)((u << 24) | (t << 16) | (s <<  8) | (r      )));
  }


  /*
   * @brief C custom defined QADD16
   */
  __STATIC_FORCEINLINE uint32_t __QADD16(
  uint32_t x,
  uint32_t y)
  {
/*  q31_t r,     s;  without initialisation 'arm_offset_q15 test' fails  but 'intrinsic' tests pass! for armCC */
    q31_t r = 0, s = 0;

    r = __SSAT(((((q31_t)x << 16) >> 16) + (((q31_t)y << 16) >> 16)), 16) & (int32_t)0x0000FFFF;
    s = __SSAT(((((q31_t)x      ) >> 16) + (((q31_t)y      ) >> 16)), 16) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r      )));
  }


  /*
   * @brief C custom defined SHADD16
   */
  __STATIC_FORCEINLINE uint32_t __SHADD16(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s;

    r = (((((q31_t)x << 16) >> 16) + (((q31_t)y << 16) >> 16)) >> 1) & (int32_t)0x0000FFFF;
    s = (((((q31_t)x      ) >> 16) + (((q31_t)y      ) >> 16)) >> 1) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r      )));
  }


  /*
   * @brief C custom defined QSUB16
   */
  __STATIC_FORCEINLINE uint32_t __QSUB16(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s;

    r = __SSAT(((((q31_t)x << 16) >> 16) - (((q31_t)y << 16) >> 16)), 16) & (int32_t)0x0000FFFF;
    s = __SSAT(((((q31_t)x      ) >> 16) - (((q31_t)y      ) >> 16)), 16) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r      )));
  }


  /*
   * @brief C custom defined SHSUB16
   */
  __STATIC_FORCEINLINE uint32_t __SHSUB16(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s;

    r = (((((q31_t)x << 16) >> 16) - (((q31_t)y << 16) >> 16)) >> 1) & (int32_t)0x0000FFFF;
    s = (((((q31_t)x      ) >> 16) - (((q31_t)y      ) >> 16)) >> 1) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r      )));
  }


  /*
   * @brief C custom defined QASX
   */
  __STATIC_FORCEINLINE uint32_t __QASX(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s;

    r = __SSAT(((((q31_t)x << 16) >> 16) - (((q31_t)y      ) >> 16)), 16) & (int32_t)0x0000FFFF;
    s = __SSAT(((((q31_t)x      ) >> 16) + (((q31_t)y << 16) >> 16)), 16) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r      )));
  }


  /*
   * @brief C custom defined SHASX
   */
  __STATIC_FORCEINLINE uint32_t __SHASX(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s;

    r = (((((q31_t)x << 16) >> 16) - (((q31_t)y      ) >> 16)) >> 1) & (int32_t)0x0000FFFF;
    s = (((((q31_t)x      ) >> 16) + (((q31_t)y << 16) >> 16)) >> 1) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r      )));
  }


  /*
   * @brief C custom defined QSAX
   */
  __STATIC_FORCEINLINE uint32_t __QSAX(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s;

    r = __SSAT(((((q31_t)x << 16) >> 16) + (((q31_t)y      ) >> 16)), 16) & (int32_t)0x0000FFFF;
    s = __SSAT(((((q31_t)x      ) >> 16) - (((q31_t)y << 16) >> 16)), 16) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r      )));
  }


  /*
   * @brief C custom defined SHSAX
   */
  __STATIC_FORCEINLINE uint32_t __SHSAX(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s;

    r = (((((q31_t)x << 16) >> 16) + (((q31_t)y      ) >> 16)) >> 1) & (int32_t)0x0000FFFF;
    s = (((((q31_t)x      ) >> 16) - (((q31_t)y << 16) >> 16)) >> 1) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r      )));
  }


  /*
   * @brief C custom defined SMUSDX
   */
  __STATIC_FORCEINLINE uint32_t __SMUSDX(
  uint32_t x,
  uint32_t y)
  {
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y      ) >> 16)) -
                       ((((q31_t)x      ) >> 16) * (((q31_t)y << 16) >> 16))   ));
  }

  /*
   * @brief C custom defined SMUADX
   */
  __STATIC_FORCEINLINE uint32_t __SMUADX(
  uint32_t x,
  uint32_t y)
  {
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y      ) >> 16)) +
                       ((((q31_t)x      ) >> 16) * (((q31_t)y << 16) >> 16))   ));
  }


  /*
   * @brief C custom defined QADD
   */
  __STATIC_FORCEINLINE int32_t __QADD(
  int32_t x,
  int32_t y)
  {
    return ((int32_t)(clip_q63_to_q31((q63_t)x + (q31_t)y)));
  }


  /*
   * @brief C custom defined QSUB
   */
  __STATIC_FORCEINLINE int32_t __QSUB(
  int32_t x,
  int32_t y)
  {
    return ((int32_t)(clip_q63_to_q31((q63_t)x - (q31_t)y)));
  }


  /*
   * @brief C custom defined SMLAD
   */
  __STATIC_FORCEINLINE uint32_t __SMLAD(
  uint32_t x,
  uint32_t y,
  uint32_t sum)
  {
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y << 16) >> 16)) +
                       ((((q31_t)x      ) >> 16) * (((q31_t)y      ) >> 16)) +
                       ( ((q31_t)sum    )                                  )   ));
  }


  /*
   * @brief C custom defined SMLADX
   */
  __STATIC_FORCEINLINE uint32_t __SMLADX(
  uint32_t x,
  uint32_t y,
  uint32_t sum)
  {
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y      ) >> 16)) +
                       ((((q31_t)x      ) >> 16) * (((q31_t)y << 16) >> 16)) +
                       ( ((q31_t)sum    )                                  )   ));
  }


  /*
   * @brief C custom defined SMLSDX
   */
  __STATIC_FORCEINLINE uint32_t __SMLSDX(
  uint32_t x,
  uint32_t y,
  uint32_t sum)
  {
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y      ) >> 16)) -
                       ((((q31_t)x      ) >> 16) * (((q31_t)y << 16) >> 16)) +
                       ( ((q31_t)sum    )                                  )   ));
  }


  /*
   * @brief C custom defined SMLALD
   */
  __STATIC_FORCEINLINE uint64_t __SMLALD(
  uint32_t x,
  uint32_t y,
  uint64_t sum)
  {
/*  return (sum + ((q15_t) (x >> 16) * (q15_t) (y >> 16)) + ((q15_t) x * (q15_t) y)); */
    return ((uint64_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y << 16) >> 16)) +
                       ((((q31_t)x      ) >> 16) * (((q31_t)y      ) >> 16)) +
                       ( ((q63_t)sum    )                                  )   ));
  }


  /*
   * @brief C custom defined SMLALDX
   */
  __STATIC_FORCEINLINE uint64_t __SMLALDX(
  uint32_t x,
  uint32_t y,
  uint64_t sum)
  {
/*  return (sum + ((q15_t) (x >> 16) * (q15_t) y)) + ((q15_t) x * (q15_t) (y >> 16)); */
    return ((uint64_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y      ) >> 16)) +
                       ((((q31_t)x      ) >> 16) * (((q31_t)y << 16) >> 16)) +
                       ( ((q63_t)sum    )                                  )   ));
  }


  /*
   * @brief C custom defined SMUAD
   */
  __STATIC_FORCEINLINE uint32_t __SMUAD(
  uint32_t x,
  uint32_t y)
  {
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y << 16) >> 16)) +
                       ((((q31_t)x      ) >> 16) * (((q31_t)y      ) >> 16))   ));
  }


  /*
   * @brief C custom defined SMUSD
   */
  __STATIC_FORCEINLINE uint32_t __SMUSD(
  uint32_t x,
  uint32_t y)
  {
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y << 16) >> 16)) -
                       ((((q31_t)x      ) >> 16) * (((q31_t)y      ) >> 16))   ));
  }


  /*
   * @brief C custom defined SXTB16
   */
  __STATIC_FORCEINLINE uint32_t __SXTB16(
  uint32_t x)
  {
    return ((uint32_t)(((((q31_t)x << 24) >> 24) & (q31_t)0x0000FFFF) |
                       ((((q31_t)x <<  8) >>  8) & (q31_t)0xFFFF0000)  ));
  }

  /*
   * @brief C custom defined SMMLA
   */
  __STATIC_FORCEINLINE int32_t __SMMLA(
  int32_t x,
  int32_t y,
  int32_t sum)
  {
    return (sum + (int32_t) (((int64_t) x * y) >> 32));
  }

#endif /* !defined (ARM_MATH_DSP) */


  /**
   * @brief Instance structure for the Q7 FIR filter.
   */
  typedef struct
  {
          uint16_t numTaps;        /**< number of filter coefficients in the filter. */
          q7_t *pState;            /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    const q7_t *pCoeffs;           /**< points to the coefficient array. The array is of length numTaps.*/
  } arm_fir_instance_q7;

  /**
   * @brief Instance structure for the Q15 FIR filter.
   */
  typedef struct
  {
          uint16_t numTaps;         /**< number of filter coefficients in the filter. */
          q15_t *pState;            /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    const q15_t *pCoeffs;           /**< points to the coefficient array. The array is of length numTaps.*/
  } arm_fir_instance_q15;

  /**
   * @brief Instance structure for the Q31 FIR filter.
   */
  typedef struct
  {
          uint16_t numTaps;         /**< number of filter coefficients in the filter. */
          q31_t *pState;            /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    const q31_t *pCoeffs;           /**< points to the coefficient array. The array is of length numTaps. */
  } arm_fir_instance_q31;

  /**
   * @brief Instance structure for the floating-point FIR filter.
   */
  typedef struct
  {
          uint16_t numTaps;     /**< number of filter coefficients in the filter. */
          float32_t *pState;    /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    const float32_t *pCoeffs;   /**< points to the coefficient array. The array is of length numTaps. */
  } arm_fir_instance_f32;

  /**
   * @brief Processing function for the Q7 FIR filter.
   * @param[in]  S          points to an instance of the Q7 FIR filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_q7(
  const arm_fir_instance_q7 * S,
  const q7_t * pSrc,
        q7_t * pDst,
        uint32_t blockSize);

  /**
   * @brief  Initialization function for the Q7 FIR filter.
   * @param[in,out] S          points to an instance of the Q7 FIR structure.
   * @param[in]     numTaps    Number of filter coefficients in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of samples that are processed.
   */
  void arm_fir_init_q7(
        arm_fir_instance_q7 * S,
        uint16_t numTaps,
  const q7_t * pCoeffs,
        q7_t * pState,
        uint32_t blockSize);

  /**
   * @brief Processing function for the Q15 FIR filter.
   * @param[in]  S          points to an instance of the Q15 FIR structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_q15(
  const arm_fir_instance_q15 * S,
  const q15_t * pSrc,
        q15_t * pDst,
        uint32_t blockSize);

  /**
   * @brief Processing function for the fast Q15 FIR filter (fast version).
   * @param[in]  S          points to an instance of the Q15 FIR filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_fast_q15(
  const arm_fir_instance_q15 * S,
  const q15_t * pSrc,
        q15_t * pDst,
        uint32_t blockSize);

  /**
   * @brief  Initialization function for the Q15 FIR filter.
   * @param[in,out] S          points to an instance of the Q15 FIR filter structure.
   * @param[in]     numTaps    Number of filter coefficients in the filter. Must be even and greater than or equal to 4.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of samples that are processed at a time.
   * @return     The function returns either
   * <code>ARM_MATH_SUCCESS</code> if initialization was successful or
   * <code>ARM_MATH_ARGUMENT_ERROR</code> if <code>numTaps</code> is not a supported value.
   */
  arm_status arm_fir_init_q15(
        arm_fir_instance_q15 * S,
        uint16_t numTaps,
  const q15_t * pCoeffs,
        q15_t * pState,
        uint32_t blockSize);

  /**
   * @brief Processing function for the Q31 FIR filter.
   * @param[in]  S          points to an instance of the Q31 FIR filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_q31(
  const arm_fir_instance_q31 * S,
  const q31_t * pSrc,
        q31_t * pDst,
        uint32_t blockSize);

  /**
   * @brief Processing function for the fast Q31 FIR filter (fast version).
   * @param[in]  S          points to an instance of the Q31 FIR filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_fast_q31(
  const arm_fir_instance_q31 * S,
  const q31_t * pSrc,
        q31_t * pDst,
        uint32_t blockSize);

  /**
   * @brief  Initialization function for the Q31 FIR filter.
   * @param[in,out] S          points to an instance of the Q31 FIR structure.
   * @param[in]     numTaps    Number of filter coefficients in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of samples that are processed at a time.
   */
  void arm_fir_init_q31(
        arm_fir_instance_q31 * S,
        uint16_t numTaps,
  const q31_t * pCoeffs,
        q31_t * pState,
        uint32_t blockSize);

  /**
   * @brief Processing function for the floating-point FIR filter.
   * @param[in]  S          points to an instance of the floating-point FIR structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_f32(
  const arm_fir_instance_f32 * S,
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize);

  /**
   * @brief  Initialization function for the floating-point FIR filter.
   * @param[in,out] S          points to an instance of the floating-point FIR filter structure.
   * @param[in]     numTaps    Number of filter coefficients in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of samples that are processed at a time.
   */
  void arm_fir_init_f32(
        arm_fir_instance_f32 * S,
        uint16_t numTaps,
  const float32_t * pCoeffs,
        float32_t * pState,
        uint32_t blockSize);

  /**
   * @brief Instance structure for the Q15 Biquad cascade filter.
   */
  typedef struct
  {
          int8_t numStages;        /**< number of 2nd order stages in the filter.  Overall order is 2*numStages. */
          q15_t *pState;           /**< Points to the array of state coefficients.  The array is of length 4*numStages. */
    const q15_t *pCoeffs;          /**< Points to the array of coefficients.  The array is of length 5*numStages. */
          int8_t postShift;        /**< Additional shift, in bits, applied to each output sample. */
  } arm_biquad_casd_df1_inst_q15;

  /**
   * @brief Instance structure for the Q31 Biquad cascade filter.
   */
  typedef struct
  {
          uint32_t numStages;      /**< number of 2nd order stages in the filter.  Overall order is 2*numStages. */
          q31_t *pState;           /**< Points to the array of state coefficients.  The array is of length 4*numStages. */
    const q31_t *pCoeffs;          /**< Points to the array of coefficients.  The array is of length 5*numStages. */
          uint8_t postShift;       /**< Additional shift, in bits, applied to each output sample. */
  } arm_biquad_casd_df1_inst_q31;

  /**
   * @brief Instance structure for the floating-point Biquad cascade filter.
   */
  typedef struct
  {
          uint32_t numStages;      /**< number of 2nd order stages in the filter.  Overall order is 2*numStages. */
          float32_t *pState;       /**< Points to the array of state coefficients.  The array is of length 4*numStages. */
    const float32_t *pCoeffs;      /**< Points to the array of coefficients.  The array is of length 5*numStages. */
  } arm_biquad_casd_df1_inst_f32;

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
  /**
   * @brief Instance structure for the modified Biquad coefs required by vectorized code.
   */
  typedef struct
  {
      float32_t coeffs[8][4]; /**< Points to the array of modified coefficients.  The array is of length 32. There is one per stage */
  } arm_biquad_mod_coef_f32;
#endif 

  /**
   * @brief Processing function for the Q15 Biquad cascade filter.
   * @param[in]  S          points to an instance of the Q15 Biquad cascade structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cascade_df1_q15(
  const arm_biquad_casd_df1_inst_q15 * S,
  const q15_t * pSrc,
        q15_t * pDst,
        uint32_t blockSize);

  /**
   * @brief  Initialization function for the Q15 Biquad cascade filter.
   * @param[in,out] S          points to an instance of the Q15 Biquad cascade structure.
   * @param[in]     numStages  number of 2nd order stages in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     postShift  Shift to be applied to the output. Varies according to the coefficients format
   */
  void arm_biquad_cascade_df1_init_q15(
        arm_biquad_casd_df1_inst_q15 * S,
        uint8_t numStages,
  const q15_t * pCoeffs,
        q15_t * pState,
        int8_t postShift);

  /**
   * @brief Fast but less precise processing function for the Q15 Biquad cascade filter for Cortex-M3 and Cortex-M4.
   * @param[in]  S          points to an instance of the Q15 Biquad cascade structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cascade_df1_fast_q15(
  const arm_biquad_casd_df1_inst_q15 * S,
  const q15_t * pSrc,
        q15_t * pDst,
        uint32_t blockSize);

  /**
   * @brief Processing function for the Q31 Biquad cascade filter
   * @param[in]  S          points to an instance of the Q31 Biquad cascade structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cascade_df1_q31(
  const arm_biquad_casd_df1_inst_q31 * S,
  const q31_t * pSrc,
        q31_t * pDst,
        uint32_t blockSize);

  /**
   * @brief Fast but less precise processing function for the Q31 Biquad cascade filter for Cortex-M3 and Cortex-M4.
   * @param[in]  S          points to an instance of the Q31 Biquad cascade structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cascade_df1_fast_q31(
  const arm_biquad_casd_df1_inst_q31 * S,
  const q31_t * pSrc,
        q31_t * pDst,
        uint32_t blockSize);

  /**
   * @brief  Initialization function for the Q31 Biquad cascade filter.
   * @param[in,out] S          points to an instance of the Q31 Biquad cascade structure.
   * @param[in]     numStages  number of 2nd order stages in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     postShift  Shift to be applied to the output. Varies according to the coefficients format
   */
  void arm_biquad_cascade_df1_init_q31(
        arm_biquad_casd_df1_inst_q31 * S,
        uint8_t numStages,
  const q31_t * pCoeffs,
        q31_t * pState,
        int8_t postShift);

  /**
   * @brief Processing function for the floating-point Biquad cascade filter.
   * @param[in]  S          points to an instance of the floating-point Biquad cascade structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cascade_df1_f32(
  const arm_biquad_casd_df1_inst_f32 * S,
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize);

  /**
   * @brief  Initialization function for the floating-point Biquad cascade filter.
   * @param[in,out] S          points to an instance of the floating-point Biquad cascade structure.
   * @param[in]     numStages  number of 2nd order stages in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pCoeffsMod points to the modified filter coefficients (only MVE version).
   * @param[in]     pState     points to the state buffer.
   */
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
  void arm_biquad_cascade_df1_mve_init_f32(
      arm_biquad_casd_df1_inst_f32 * S,
      uint8_t numStages,
      const float32_t * pCoeffs, 
      arm_biquad_mod_coef_f32 * pCoeffsMod, 
      float32_t * pState);
#endif
  
  void arm_biquad_cascade_df1_init_f32(
        arm_biquad_casd_df1_inst_f32 * S,
        uint8_t numStages,
  const float32_t * pCoeffs,
        float32_t * pState);


  /**
   * @brief         Compute the logical bitwise AND of two fixed-point vectors.
   * @param[in]     pSrcA      points to input vector A
   * @param[in]     pSrcB      points to input vector B
   * @param[out]    pDst       points to output vector
   * @param[in]     blockSize  number of samples in each vector
   * @return        none
   */
  void arm_and_u16(
    const uint16_t * pSrcA,
    const uint16_t * pSrcB,
          uint16_t * pDst,
          uint32_t blockSize);

  /**
   * @brief         Compute the logical bitwise AND of two fixed-point vectors.
   * @param[in]     pSrcA      points to input vector A
   * @param[in]     pSrcB      points to input vector B
   * @param[out]    pDst       points to output vector
   * @param[in]     blockSize  number of samples in each vector
   * @return        none
   */
  void arm_and_u32(
    const uint32_t * pSrcA,
    const uint32_t * pSrcB,
          uint32_t * pDst,
          uint32_t blockSize);

  /**
   * @brief         Compute the logical bitwise AND of two fixed-point vectors.
   * @param[in]     pSrcA      points to input vector A
   * @param[in]     pSrcB      points to input vector B
   * @param[out]    pDst       points to output vector
   * @param[in]     blockSize  number of samples in each vector
   * @return        none
   */
  void arm_and_u8(
    const uint8_t * pSrcA,
    const uint8_t * pSrcB,
          uint8_t * pDst,
          uint32_t blockSize);

  /**
   * @brief         Compute the logical bitwise OR of two fixed-point vectors.
   * @param[in]     pSrcA      points to input vector A
   * @param[in]     pSrcB      points to input vector B
   * @param[out]    pDst       points to output vector
   * @param[in]     blockSize  number of samples in each vector
   * @return        none
   */
  void arm_or_u16(
    const uint16_t * pSrcA,
    const uint16_t * pSrcB,
          uint16_t * pDst,
          uint32_t blockSize);

  /**
   * @brief         Compute the logical bitwise OR of two fixed-point vectors.
   * @param[in]     pSrcA      points to input vector A
   * @param[in]     pSrcB      points to input vector B
   * @param[out]    pDst       points to output vector
   * @param[in]     blockSize  number of samples in each vector
   * @return        none
   */
  void arm_or_u32(
    const uint32_t * pSrcA,
    const uint32_t * pSrcB,
          uint32_t * pDst,
          uint32_t blockSize);

  /**
   * @brief         Compute the logical bitwise OR of two fixed-point vectors.
   * @param[in]     pSrcA      points to input vector A
   * @param[in]     pSrcB      points to input vector B
   * @param[out]    pDst       points to output vector
   * @param[in]     blockSize  number of samples in each vector
   * @return        none
   */
  void arm_or_u8(
    const uint8_t * pSrcA,
    const uint8_t * pSrcB,
          uint8_t * pDst,
          uint32_t blockSize);

  /**
   * @brief         Compute the logical bitwise NOT of a fixed-point vector.
   * @param[in]     pSrc       points to input vector 
   * @param[out]    pDst       points to output vector
   * @param[in]     blockSize  number of samples in each vector
   * @return        none
   */
  void arm_not_u16(
    const uint16_t * pSrc,
          uint16_t * pDst,
          uint32_t blockSize);

  /**
   * @brief         Compute the logical bitwise NOT of a fixed-point vector.
   * @param[in]     pSrc       points to input vector 
   * @param[out]    pDst       points to output vector
   * @param[in]     blockSize  number of samples in each vector
   * @return        none
   */
  void arm_not_u32(
    const uint32_t * pSrc,
          uint32_t * pDst,
          uint32_t blockSize);

  /**
   * @brief         Compute the logical bitwise NOT of a fixed-point vector.
   * @param[in]     pSrc       points to input vector 
   * @param[out]    pDst       points to output vector
   * @param[in]     blockSize  number of samples in each vector
   * @return        none
   */
  void arm_not_u8(
    const uint8_t * pSrc,
          uint8_t * pDst,
          uint32_t blockSize);

/**
   * @brief         Compute the logical bitwise XOR of two fixed-point vectors.
   * @param[in]     pSrcA      points to input vector A
   * @param[in]     pSrcB      points to input vector B
   * @param[out]    pDst       points to output vector
   * @param[in]     blockSize  number of samples in each vector
   * @return        none
   */
  void arm_xor_u16(
    const uint16_t * pSrcA,
    const uint16_t * pSrcB,
          uint16_t * pDst,
          uint32_t blockSize);

  /**
   * @brief         Compute the logical bitwise XOR of two fixed-point vectors.
   * @param[in]     pSrcA      points to input vector A
   * @param[in]     pSrcB      points to input vector B
   * @param[out]    pDst       points to output vector
   * @param[in]     blockSize  number of samples in each vector
   * @return        none
   */
  void arm_xor_u32(
    const uint32_t * pSrcA,
    const uint32_t * pSrcB,
          uint32_t * pDst,
          uint32_t blockSize);

  /**
   * @brief         Compute the logical bitwise XOR of two fixed-point vectors.
   * @param[in]     pSrcA      points to input vector A
   * @param[in]     pSrcB      points to input vector B
   * @param[out]    pDst       points to output vector
   * @param[in]     blockSize  number of samples in each vector
   * @return        none
   */
  void arm_xor_u8(
    const uint8_t * pSrcA,
    const uint8_t * pSrcB,
          uint8_t * pDst,
    uint32_t blockSize);

  /**
   * @brief Struct for specifying sorting algorithm
   */
  typedef enum
  {
    ARM_SORT_BITONIC   = 0,
             /**< Bitonic sort   */
    ARM_SORT_BUBBLE    = 1,
             /**< Bubble sort    */
    ARM_SORT_HEAP      = 2,
             /**< Heap sort      */
    ARM_SORT_INSERTION = 3,
             /**< Insertion sort */
    ARM_SORT_QUICK     = 4,
             /**< Quick sort     */
    ARM_SORT_SELECTION = 5
             /**< Selection sort */
  } arm_sort_alg;

  /**
   * @brief Struct for specifying sorting algorithm
   */
  typedef enum
  {
    ARM_SORT_DESCENDING = 0,
             /**< Descending order (9 to 0) */
    ARM_SORT_ASCENDING = 1
             /**< Ascending order (0 to 9) */
  } arm_sort_dir;

  /**
   * @brief Instance structure for the sorting algorithms.
   */
  typedef struct            
  {
    arm_sort_alg alg;        /**< Sorting algorithm selected */
    arm_sort_dir dir;        /**< Sorting order (direction)  */
  } arm_sort_instance_f32;  

  /**
   * @param[in]  S          points to an instance of the sorting structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_sort_f32(
    const arm_sort_instance_f32 * S, 
          float32_t * pSrc, 
          float32_t * pDst, 
          uint32_t blockSize);

  /**
   * @param[in,out]  S            points to an instance of the sorting structure.
   * @param[in]      alg          Selected algorithm.
   * @param[in]      dir          Sorting order.
   */
  void arm_sort_init_f32(
    arm_sort_instance_f32 * S, 
    arm_sort_alg alg, 
    arm_sort_dir dir); 

  /**
   * @brief Instance structure for the sorting algorithms.
   */
  typedef struct            
  {
    arm_sort_dir dir;        /**< Sorting order (direction)  */
    float32_t * buffer;      /**< Working buffer */
  } arm_merge_sort_instance_f32;  

  /**
   * @param[in]      S          points to an instance of the sorting structure.
   * @param[in,out]  pSrc       points to the block of input data.
   * @param[out]     pDst       points to the block of output data
   * @param[in]      blockSize  number of samples to process.
   */
  void arm_merge_sort_f32(
    const arm_merge_sort_instance_f32 * S,
          float32_t *pSrc,
          float32_t *pDst,
          uint32_t blockSize);

  /**
   * @param[in,out]  S            points to an instance of the sorting structure.
   * @param[in]      dir          Sorting order.
   * @param[in]      buffer       Working buffer.
   */
  void arm_merge_sort_init_f32(
    arm_merge_sort_instance_f32 * S,
    arm_sort_dir dir,
    float32_t * buffer);

  /**
   * @brief Struct for specifying cubic spline type
   */
  typedef enum
  {
    ARM_SPLINE_NATURAL = 0,           /**< Natural spline */
    ARM_SPLINE_PARABOLIC_RUNOUT = 1   /**< Parabolic runout spline */
  } arm_spline_type;

  /**
   * @brief Instance structure for the floating-point cubic spline interpolation.
   */
  typedef struct
  {
    arm_spline_type type;      /**< Type (boundary conditions) */
    const float32_t * x;       /**< x values */
    const float32_t * y;       /**< y values */
    uint32_t n_x;              /**< Number of known data points */
    float32_t * coeffs;        /**< Coefficients buffer (b,c, and d) */
  } arm_spline_instance_f32;

  /**
   * @brief Processing function for the floating-point cubic spline interpolation.
   * @param[in]  S          points to an instance of the floating-point spline structure.
   * @param[in]  xq         points to the x values ot the interpolated data points.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples of output data.
   */
  void arm_spline_f32(
        arm_spline_instance_f32 * S, 
  const float32_t * xq,
        float32_t * pDst,
        uint32_t blockSize);

  /**
   * @brief Initialization function for the floating-point cubic spline interpolation.
   * @param[in,out] S        points to an instance of the floating-point spline structure.
   * @param[in]     type     type of cubic spline interpolation (boundary conditions)
   * @param[in]     x        points to the x values of the known data points.
   * @param[in]     y        points to the y values of the known data points.
   * @param[in]     n        number of known data points.
   * @param[in]     coeffs   coefficients array for b, c, and d
   * @param[in]     tempBuffer   buffer array for internal computations
   */
  void arm_spline_init_f32(
          arm_spline_instance_f32 * S,
          arm_spline_type type,
    const float32_t * x,
    const float32_t * y,
          uint32_t n, 
          float32_t * coeffs,
          float32_t * tempBuffer);

  /**
   * @brief Instance structure for the floating-point matrix structure.
   */
  typedef struct
  {
    uint16_t numRows;     /**< number of rows of the matrix.     */
    uint16_t numCols;     /**< number of columns of the matrix.  */
    float32_t *pData;     /**< points to the data of the matrix. */
  } arm_matrix_instance_f32;
 
 /**
   * @brief Instance structure for the floating-point matrix structure.
   */
  typedef struct
  {
    uint16_t numRows;     /**< number of rows of the matrix.     */
    uint16_t numCols;     /**< number of columns of the matrix.  */
    float64_t *pData;     /**< points to the data of the matrix. */
  } arm_matrix_instance_f64;

  /**
    * @brief Instance structure for the Q7 matrix structure.
    */
   typedef struct
   {
     uint16_t numRows;     /**< number of rows of the matrix.     */
     uint16_t numCols;     /**< number of columns of the matrix.  */
     q7_t *pData;         /**< points to the data of the matrix. */
   } arm_matrix_instance_q7;

  /**
   * @brief Instance structure for the Q15 matrix structure.
   */
  typedef struct
  {
    uint16_t numRows;     /**< number of rows of the matrix.     */
    uint16_t numCols;     /**< number of columns of the matrix.  */
    q15_t *pData;         /**< points to the data of the matrix. */
  } arm_matrix_instance_q15;

  /**
   * @brief Instance structure for the Q31 matrix structure.
   */
  typedef struct
  {
    uint16_t numRows;     /**< number of rows of the matrix.     */
    uint16_t numCols;     /**< number of columns of the matrix.  */
    q31_t *pData;         /**< points to the data of the matrix. */
  } arm_matrix_instance_q31;

  /**
   * @brief Floating-point matrix addition.
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_add_f32(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

  /**
   * @brief Q15 matrix addition.
   * @param[in]   pSrcA  points to the first input matrix structure
   * @param[in]   pSrcB  points to the second input matrix structure
   * @param[out]  pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_add_q15(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
        arm_matrix_instance_q15 * pDst);

  /**
   * @brief Q31 matrix addition.
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_add_q31(
  const arm_matrix_instance_q31 * pSrcA,
  const arm_matrix_instance_q31 * pSrcB,
        arm_matrix_instance_q31 * pDst);

  /**
   * @brief Floating-point, complex, matrix multiplication.
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_cmplx_mult_f32(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

  /**
   * @brief Q15, complex,  matrix multiplication.
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_cmplx_mult_q15(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
        arm_matrix_instance_q15 * pDst,
        q15_t * pScratch);

  /**
   * @brief Q31, complex, matrix multiplication.
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_cmplx_mult_q31(
  const arm_matrix_instance_q31 * pSrcA,
  const arm_matrix_instance_q31 * pSrcB,
        arm_matrix_instance_q31 * pDst);

  /**
   * @brief Floating-point matrix transpose.
   * @param[in]  pSrc  points to the input matrix
   * @param[out] pDst  points to the output matrix
   * @return    The function returns either  <code>ARM_MATH_SIZE_MISMATCH</code>
   * or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_trans_f32(
  const arm_matrix_instance_f32 * pSrc,
        arm_matrix_instance_f32 * pDst);

  /**
   * @brief Q15 matrix transpose.
   * @param[in]  pSrc  points to the input matrix
   * @param[out] pDst  points to the output matrix
   * @return    The function returns either  <code>ARM_MATH_SIZE_MISMATCH</code>
   * or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_trans_q15(
  const arm_matrix_instance_q15 * pSrc,
        arm_matrix_instance_q15 * pDst);

  /**
   * @brief Q31 matrix transpose.
   * @param[in]  pSrc  points to the input matrix
   * @param[out] pDst  points to the output matrix
   * @return    The function returns either  <code>ARM_MATH_SIZE_MISMATCH</code>
   * or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_trans_q31(
  const arm_matrix_instance_q31 * pSrc,
        arm_matrix_instance_q31 * pDst);

  /**
   * @brief Floating-point matrix multiplication
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_mult_f32(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

  /**
   * @brief Q15 matrix multiplication
   * @param[in]  pSrcA   points to the first input matrix structure
   * @param[in]  pSrcB   points to the second input matrix structure
   * @param[out] pDst    points to output matrix structure
   * @param[in]  pState  points to the array for storing intermediate results
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_mult_q15(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
        arm_matrix_instance_q15 * pDst,
        q15_t * pState);

  /**
   * @brief Q15 matrix multiplication (fast variant) for Cortex-M3 and Cortex-M4
   * @param[in]  pSrcA   points to the first input matrix structure
   * @param[in]  pSrcB   points to the second input matrix structure
   * @param[out] pDst    points to output matrix structure
   * @param[in]  pState  points to the array for storing intermediate results
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_mult_fast_q15(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
        arm_matrix_instance_q15 * pDst,
        q15_t * pState);

  /**
   * @brief Q31 matrix multiplication
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_mult_q31(
  const arm_matrix_instance_q31 * pSrcA,
  const arm_matrix_instance_q31 * pSrcB,
        arm_matrix_instance_q31 * pDst);

  /**
   * @brief Q31 matrix multiplication (fast variant) for Cortex-M3 and Cortex-M4
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_mult_fast_q31(
  const arm_matrix_instance_q31 * pSrcA,
  const arm_matrix_instance_q31 * pSrcB,
        arm_matrix_instance_q31 * pDst);

  /**
   * @brief Floating-point matrix subtraction
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_sub_f32(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

  /**
   * @brief Q15 matrix subtraction
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_sub_q15(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
        arm_matrix_instance_q15 * pDst);

  /**
   * @brief Q31 matrix subtraction
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_sub_q31(
  const arm_matrix_instance_q31 * pSrcA,
  const arm_matrix_instance_q31 * pSrcB,
        arm_matrix_instance_q31 * pDst);

  /**
   * @brief Floating-point matrix scaling.
   * @param[in]  pSrc   points to the input matrix
   * @param[in]  scale  scale factor
   * @param[out] pDst   points to the output matrix
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_scale_f32(
  const arm_matrix_instance_f32 * pSrc,
        float32_t scale,
        arm_matrix_instance_f32 * pDst);

  /**
   * @brief Q15 matrix scaling.
   * @param[in]  pSrc        points to input matrix
   * @param[in]  scaleFract  fractional portion of the scale factor
   * @param[in]  shift       number of bits to shift the result by
   * @param[out] pDst        points to output matrix
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_scale_q15(
  const arm_matrix_instance_q15 * pSrc,
        q15_t scaleFract,
        int32_t shift,
        arm_matrix_instance_q15 * pDst);

  /**
   * @brief Q31 matrix scaling.
   * @param[in]  pSrc        points to input matrix
   * @param[in]  scaleFract  fractional portion of the scale factor
   * @param[in]  shift       number of bits to shift the result by
   * @param[out] pDst        points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status arm_mat_scale_q31(
  const arm_matrix_instance_q31 * pSrc,
        q31_t scaleFract,
        int32_t shift,
        arm_matrix_instance_q31 * pDst);

  /**
   * @brief  Q31 matrix initialization.
   * @param[in,out] S         points to an instance of the floating-point matrix structure.
   * @param[in]     nRows     number of rows in the matrix.
   * @param[in]     nColumns  number of columns in the matrix.
   * @param[in]     pData     points to the matrix data array.
   */
void arm_mat_init_q31(
        arm_matrix_instance_q31 * S,
        uint16_t nRows,
        uint16_t nColumns,
        q31_t * pData);

  /**
   * @brief  Q15 matrix initialization.
   * @param[in,out] S         points to an instance of the floating-point matrix structure.
   * @param[in]     nRows     number of rows in the matrix.
   * @param[in]     nColumns  number of columns in the matrix.
   * @param[in]     pData     points to the matrix data array.
   */
void arm_mat_init_q15(
        arm_matrix_instance_q15 * S,
        uint16_t nRows,
        uint16_t nColumns,
        q15_t * pData);

  /**
   * @brief  Floating-point matrix initialization.
   * @param[in,out] S         points to an instance of the floating-point matrix structure.
   * @param[in]     nRows     number of rows in the matrix.
   * @param[in]     nColumns  number of columns in the matrix.
   * @param[in]     pData     points to the matrix data array.
   */
void arm_mat_init_f32(
        arm_matrix_instance_f32 * S,
        uint16_t nRows,
        uint16_t nColumns,
        float32_t * pData);


  /**
   * @brief Instance structure for the Q15 PID Control.
   */
  typedef struct
  {
          q15_t A0;           /**< The derived gain, A0 = Kp + Ki + Kd . */
#if !defined (ARM_MATH_DSP)
          q15_t A1;
          q15_t A2;
#else
          q31_t A1;           /**< The derived gain A1 = -Kp - 2Kd | Kd.*/
#endif
          q15_t state[3];     /**< The state array of length 3. */
          q15_t Kp;           /**< The proportional gain. */
          q15_t Ki;           /**< The integral gain. */
          q15_t Kd;           /**< The derivative gain. */
  } arm_pid_instance_q15;

  /**
   * @brief Instance structure for the Q31 PID Control.
   */
  typedef struct
  {
          q31_t A0;            /**< The derived gain, A0 = Kp + Ki + Kd . */
          q31_t A1;            /**< The derived gain, A1 = -Kp - 2Kd. */
          q31_t A2;            /**< The derived gain, A2 = Kd . */
          q31_t state[3];      /**< The state array of length 3. */
          q31_t Kp;            /**< The proportional gain. */
          q31_t Ki;            /**< The integral gain. */
          q31_t Kd;            /**< The derivative gain. */
  } arm_pid_instance_q31;

  /**
   * @brief Instance structure for the floating-point PID Control.
   */
  typedef struct
  {
          float32_t A0;          /**< The derived gain, A0 = Kp + Ki + Kd . */
          float32_t A1;          /**< The derived gain, A1 = -Kp - 2Kd. */
          float32_t A2;          /**< The derived gain, A2 = Kd . */
          float32_t state[3];    /**< The state array of length 3. */
          float32_t Kp;          /**< The proportional gain. */
          float32_t Ki;          /**< The integral gain. */
          float32_t Kd;          /**< The derivative gain. */
  } arm_pid_instance_f32;



  /**
   * @brief  Initialization function for the floating-point PID Control.
   * @param[in,out] S               points to an instance of the PID structure.
   * @param[in]     resetStateFlag  flag to reset the state. 0 = no change in state 1 = reset the state.
   */
  void arm_pid_init_f32(
        arm_pid_instance_f32 * S,
        int32_t resetStateFlag);


  /**
   * @brief  Reset function for the floating-point PID Control.
   * @param[in,out] S  is an instance of the floating-point PID Control structure
   */
  void arm_pid_reset_f32(
        arm_pid_instance_f32 * S);


  /**
   * @brief  Initialization function for the Q31 PID Control.
   * @param[in,out] S               points to an instance of the Q15 PID structure.
   * @param[in]     resetStateFlag  flag to reset the state. 0 = no change in state 1 = reset the state.
   */
  void arm_pid_init_q31(
        arm_pid_instance_q31 * S,
        int32_t resetStateFlag);


  /**
   * @brief  Reset function for the Q31 PID Control.
   * @param[in,out] S   points to an instance of the Q31 PID Control structure
   */

  void arm_pid_reset_q31(
        arm_pid_instance_q31 * S);


  /**
   * @brief  Initialization function for the Q15 PID Control.
   * @param[in,out] S               points to an instance of the Q15 PID structure.
   * @param[in]     resetStateFlag  flag to reset the state. 0 = no change in state 1 = reset the state.
   */
  void arm_pid_init_q15(
        arm_pid_instance_q15 * S,
        int32_t resetStateFlag);


  /**
   * @brief  Reset function for the Q15 PID Control.
   * @param[in,out] S  points to an instance of the q15 PID Control structure
   */
  void arm_pid_reset_q15(
        arm_pid_instance_q15 * S);


  /**
   * @brief Instance structure for the floating-point Linear Interpolate function.
   */
  typedef struct
  {
          uint32_t nValues;           /**< nValues */
          float32_t x1;               /**< x1 */
          float32_t xSpacing;         /**< xSpacing */
          float32_t *pYData;          /**< pointer to the table of Y values */
  } arm_linear_interp_instance_f32;

  /**
   * @brief Instance structure for the floating-point bilinear interpolation function.
   */
  typedef struct
  {
          uint16_t numRows;   /**< number of rows in the data table. */
          uint16_t numCols;   /**< number of columns in the data table. */
          float32_t *pData;   /**< points to the data table. */
  } arm_bilinear_interp_instance_f32;

   /**
   * @brief Instance structure for the Q31 bilinear interpolation function.
   */
  typedef struct
  {
          uint16_t numRows;   /**< number of rows in the data table. */
          uint16_t numCols;   /**< number of columns in the data table. */
          q31_t *pData;       /**< points to the data table. */
  } arm_bilinear_interp_instance_q31;

   /**
   * @brief Instance structure for the Q15 bilinear interpolation function.
   */
  typedef struct
  {
          uint16_t numRows;   /**< number of rows in the data table. */
          uint16_t numCols;   /**< number of columns in the data table. */
          q15_t *pData;       /**< points to the data table. */
  } arm_bilinear_interp_instance_q15;

   /**
   * @brief Instance structure for the Q15 bilinear interpolation function.
   */
  typedef struct
  {
          uint16_t numRows;   /**< number of rows in the data table. */
          uint16_t numCols;   /**< number of columns in the data table. */
          q7_t *pData;        /**< points to the data table. */
  } arm_bilinear_interp_instance_q7;


  /**
   * @brief Q7 vector multiplication.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_mult_q7(
  const q7_t * pSrcA,
  const q7_t * pSrcB,
        q7_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Q15 vector multiplication.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_mult_q15(
  const q15_t * pSrcA,
  const q15_t * pSrcB,
        q15_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Q31 vector multiplication.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_mult_q31(
  const q31_t * pSrcA,
  const q31_t * pSrcB,
        q31_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Floating-point vector multiplication.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_mult_f32(
  const float32_t * pSrcA,
  const float32_t * pSrcB,
        float32_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Instance structure for the Q15 CFFT/CIFFT function.
   */
  typedef struct
  {
          uint16_t fftLen;                 /**< length of the FFT. */
          uint8_t ifftFlag;                /**< flag that selects forward (ifftFlag=0) or inverse (ifftFlag=1) transform. */
          uint8_t bitReverseFlag;          /**< flag that enables (bitReverseFlag=1) or disables (bitReverseFlag=0) bit reversal of output. */
    const q15_t *pTwiddle;                 /**< points to the Sin twiddle factor table. */
    const uint16_t *pBitRevTable;          /**< points to the bit reversal table. */
          uint16_t twidCoefModifier;       /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
          uint16_t bitRevFactor;           /**< bit reversal modifier that supports different size FFTs with the same bit reversal table. */
  } arm_cfft_radix2_instance_q15;

/* Deprecated */
  arm_status arm_cfft_radix2_init_q15(
        arm_cfft_radix2_instance_q15 * S,
        uint16_t fftLen,
        uint8_t ifftFlag,
        uint8_t bitReverseFlag);

/* Deprecated */
  void arm_cfft_radix2_q15(
  const arm_cfft_radix2_instance_q15 * S,
        q15_t * pSrc);


  /**
   * @brief Instance structure for the Q15 CFFT/CIFFT function.
   */
  typedef struct
  {
          uint16_t fftLen;                 /**< length of the FFT. */
          uint8_t ifftFlag;                /**< flag that selects forward (ifftFlag=0) or inverse (ifftFlag=1) transform. */
          uint8_t bitReverseFlag;          /**< flag that enables (bitReverseFlag=1) or disables (bitReverseFlag=0) bit reversal of output. */
    const q15_t *pTwiddle;                 /**< points to the twiddle factor table. */
    const uint16_t *pBitRevTable;          /**< points to the bit reversal table. */
          uint16_t twidCoefModifier;       /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
          uint16_t bitRevFactor;           /**< bit reversal modifier that supports different size FFTs with the same bit reversal table. */
  } arm_cfft_radix4_instance_q15;

/* Deprecated */
  arm_status arm_cfft_radix4_init_q15(
        arm_cfft_radix4_instance_q15 * S,
        uint16_t fftLen,
        uint8_t ifftFlag,
        uint8_t bitReverseFlag);

/* Deprecated */
  void arm_cfft_radix4_q15(
  const arm_cfft_radix4_instance_q15 * S,
        q15_t * pSrc);

  /**
   * @brief Instance structure for the Radix-2 Q31 CFFT/CIFFT function.
   */
  typedef struct
  {
          uint16_t fftLen;                 /**< length of the FFT. */
          uint8_t ifftFlag;                /**< flag that selects forward (ifftFlag=0) or inverse (ifftFlag=1) transform. */
          uint8_t bitReverseFlag;          /**< flag that enables (bitReverseFlag=1) or disables (bitReverseFlag=0) bit reversal of output. */
    const q31_t *pTwiddle;                 /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable;          /**< points to the bit reversal table. */
          uint16_t twidCoefModifier;       /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
          uint16_t bitRevFactor;           /**< bit reversal modifier that supports different size FFTs with the same bit reversal table. */
  } arm_cfft_radix2_instance_q31;

/* Deprecated */
  arm_status arm_cfft_radix2_init_q31(
        arm_cfft_radix2_instance_q31 * S,
        uint16_t fftLen,
        uint8_t ifftFlag,
        uint8_t bitReverseFlag);

/* Deprecated */
  void arm_cfft_radix2_q31(
  const arm_cfft_radix2_instance_q31 * S,
        q31_t * pSrc);

  /**
   * @brief Instance structure for the Q31 CFFT/CIFFT function.
   */
  typedef struct
  {
          uint16_t fftLen;                 /**< length of the FFT. */
          uint8_t ifftFlag;                /**< flag that selects forward (ifftFlag=0) or inverse (ifftFlag=1) transform. */
          uint8_t bitReverseFlag;          /**< flag that enables (bitReverseFlag=1) or disables (bitReverseFlag=0) bit reversal of output. */
    const q31_t *pTwiddle;                 /**< points to the twiddle factor table. */
    const uint16_t *pBitRevTable;          /**< points to the bit reversal table. */
          uint16_t twidCoefModifier;       /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
          uint16_t bitRevFactor;           /**< bit reversal modifier that supports different size FFTs with the same bit reversal table. */
  } arm_cfft_radix4_instance_q31;

/* Deprecated */
  void arm_cfft_radix4_q31(
  const arm_cfft_radix4_instance_q31 * S,
        q31_t * pSrc);

/* Deprecated */
  arm_status arm_cfft_radix4_init_q31(
        arm_cfft_radix4_instance_q31 * S,
        uint16_t fftLen,
        uint8_t ifftFlag,
        uint8_t bitReverseFlag);

  /**
   * @brief Instance structure for the floating-point CFFT/CIFFT function.
   */
  typedef struct
  {
          uint16_t fftLen;                   /**< length of the FFT. */
          uint8_t ifftFlag;                  /**< flag that selects forward (ifftFlag=0) or inverse (ifftFlag=1) transform. */
          uint8_t bitReverseFlag;            /**< flag that enables (bitReverseFlag=1) or disables (bitReverseFlag=0) bit reversal of output. */
    const float32_t *pTwiddle;               /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable;            /**< points to the bit reversal table. */
          uint16_t twidCoefModifier;         /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
          uint16_t bitRevFactor;             /**< bit reversal modifier that supports different size FFTs with the same bit reversal table. */
          float32_t onebyfftLen;             /**< value of 1/fftLen. */
  } arm_cfft_radix2_instance_f32;

/* Deprecated */
  arm_status arm_cfft_radix2_init_f32(
        arm_cfft_radix2_instance_f32 * S,
        uint16_t fftLen,
        uint8_t ifftFlag,
        uint8_t bitReverseFlag);

/* Deprecated */
  void arm_cfft_radix2_f32(
  const arm_cfft_radix2_instance_f32 * S,
        float32_t * pSrc);

  /**
   * @brief Instance structure for the floating-point CFFT/CIFFT function.
   */
  typedef struct
  {
          uint16_t fftLen;                   /**< length of the FFT. */
          uint8_t ifftFlag;                  /**< flag that selects forward (ifftFlag=0) or inverse (ifftFlag=1) transform. */
          uint8_t bitReverseFlag;            /**< flag that enables (bitReverseFlag=1) or disables (bitReverseFlag=0) bit reversal of output. */
    const float32_t *pTwiddle;               /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable;            /**< points to the bit reversal table. */
          uint16_t twidCoefModifier;         /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
          uint16_t bitRevFactor;             /**< bit reversal modifier that supports different size FFTs with the same bit reversal table. */
          float32_t onebyfftLen;             /**< value of 1/fftLen. */
  } arm_cfft_radix4_instance_f32;

/* Deprecated */
  arm_status arm_cfft_radix4_init_f32(
        arm_cfft_radix4_instance_f32 * S,
        uint16_t fftLen,
        uint8_t ifftFlag,
        uint8_t bitReverseFlag);

/* Deprecated */
  void arm_cfft_radix4_f32(
  const arm_cfft_radix4_instance_f32 * S,
        float32_t * pSrc);

  /**
   * @brief Instance structure for the fixed-point CFFT/CIFFT function.
   */
  typedef struct
  {
          uint16_t fftLen;                   /**< length of the FFT. */
    const q15_t *pTwiddle;             /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable;      /**< points to the bit reversal table. */
          uint16_t bitRevLength;             /**< bit reversal table length. */
#if defined(ARM_MATH_MVEI)
   const uint32_t *rearranged_twiddle_tab_stride1_arr;        /**< Per stage reordered twiddle pointer (offset 1) */                                                       \
   const uint32_t *rearranged_twiddle_tab_stride2_arr;        /**< Per stage reordered twiddle pointer (offset 2) */                                                       \
   const uint32_t *rearranged_twiddle_tab_stride3_arr;        /**< Per stage reordered twiddle pointer (offset 3) */                                                       \
   const q15_t *rearranged_twiddle_stride1; /**< reordered twiddle offset 1 storage */                                                                   \
   const q15_t *rearranged_twiddle_stride2; /**< reordered twiddle offset 2 storage */                                                                   \
   const q15_t *rearranged_twiddle_stride3;
#endif
  } arm_cfft_instance_q15;

arm_status arm_cfft_init_q15(
  arm_cfft_instance_q15 * S,
  uint16_t fftLen);

void arm_cfft_q15(
    const arm_cfft_instance_q15 * S,
          q15_t * p1,
          uint8_t ifftFlag,
          uint8_t bitReverseFlag);

  /**
   * @brief Instance structure for the fixed-point CFFT/CIFFT function.
   */
  typedef struct
  {
          uint16_t fftLen;                   /**< length of the FFT. */
    const q31_t *pTwiddle;             /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable;      /**< points to the bit reversal table. */
          uint16_t bitRevLength;             /**< bit reversal table length. */
#if defined(ARM_MATH_MVEI)
   const uint32_t *rearranged_twiddle_tab_stride1_arr;        /**< Per stage reordered twiddle pointer (offset 1) */                                                       \
   const uint32_t *rearranged_twiddle_tab_stride2_arr;        /**< Per stage reordered twiddle pointer (offset 2) */                                                       \
   const uint32_t *rearranged_twiddle_tab_stride3_arr;        /**< Per stage reordered twiddle pointer (offset 3) */                                                       \
   const q31_t *rearranged_twiddle_stride1; /**< reordered twiddle offset 1 storage */                                                                   \
   const q31_t *rearranged_twiddle_stride2; /**< reordered twiddle offset 2 storage */                                                                   \
   const q31_t *rearranged_twiddle_stride3;
#endif
  } arm_cfft_instance_q31;

arm_status arm_cfft_init_q31(
  arm_cfft_instance_q31 * S,
  uint16_t fftLen);

void arm_cfft_q31(
    const arm_cfft_instance_q31 * S,
          q31_t * p1,
          uint8_t ifftFlag,
          uint8_t bitReverseFlag);

  /**
   * @brief Instance structure for the floating-point CFFT/CIFFT function.
   */
  typedef struct
  {
          uint16_t fftLen;                   /**< length of the FFT. */
    const float32_t *pTwiddle;         /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable;      /**< points to the bit reversal table. */
          uint16_t bitRevLength;             /**< bit reversal table length. */
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
   const uint32_t *rearranged_twiddle_tab_stride1_arr;        /**< Per stage reordered twiddle pointer (offset 1) */                                                       \
   const uint32_t *rearranged_twiddle_tab_stride2_arr;        /**< Per stage reordered twiddle pointer (offset 2) */                                                       \
   const uint32_t *rearranged_twiddle_tab_stride3_arr;        /**< Per stage reordered twiddle pointer (offset 3) */                                                       \
   const float32_t *rearranged_twiddle_stride1; /**< reordered twiddle offset 1 storage */                                                                   \
   const float32_t *rearranged_twiddle_stride2; /**< reordered twiddle offset 2 storage */                                                                   \
   const float32_t *rearranged_twiddle_stride3;
#endif
  } arm_cfft_instance_f32;


  arm_status arm_cfft_init_f32(
  arm_cfft_instance_f32 * S,
  uint16_t fftLen);

  void arm_cfft_f32(
  const arm_cfft_instance_f32 * S,
        float32_t * p1,
        uint8_t ifftFlag,
        uint8_t bitReverseFlag);


  /**
   * @brief Instance structure for the Double Precision Floating-point CFFT/CIFFT function.
   */
  typedef struct
  {
          uint16_t fftLen;                   /**< length of the FFT. */
    const float64_t *pTwiddle;         /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable;      /**< points to the bit reversal table. */
          uint16_t bitRevLength;             /**< bit reversal table length. */
  } arm_cfft_instance_f64;

  void arm_cfft_f64(
  const arm_cfft_instance_f64 * S,
        float64_t * p1,
        uint8_t ifftFlag,
        uint8_t bitReverseFlag);

  /**
   * @brief Instance structure for the Q15 RFFT/RIFFT function.
   */
  typedef struct
  {
          uint32_t fftLenReal;                      /**< length of the real FFT. */
          uint8_t ifftFlagR;                        /**< flag that selects forward (ifftFlagR=0) or inverse (ifftFlagR=1) transform. */
          uint8_t bitReverseFlagR;                  /**< flag that enables (bitReverseFlagR=1) or disables (bitReverseFlagR=0) bit reversal of output. */
          uint32_t twidCoefRModifier;               /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
    const q15_t *pTwiddleAReal;                     /**< points to the real twiddle factor table. */
    const q15_t *pTwiddleBReal;                     /**< points to the imag twiddle factor table. */
#if defined(ARM_MATH_MVEI)
    arm_cfft_instance_q15 cfftInst;
#else
    const arm_cfft_instance_q15 *pCfft;       /**< points to the complex FFT instance. */
#endif
  } arm_rfft_instance_q15;

  arm_status arm_rfft_init_q15(
        arm_rfft_instance_q15 * S,
        uint32_t fftLenReal,
        uint32_t ifftFlagR,
        uint32_t bitReverseFlag);

  void arm_rfft_q15(
  const arm_rfft_instance_q15 * S,
        q15_t * pSrc,
        q15_t * pDst);

  /**
   * @brief Instance structure for the Q31 RFFT/RIFFT function.
   */
  typedef struct
  {
          uint32_t fftLenReal;                        /**< length of the real FFT. */
          uint8_t ifftFlagR;                          /**< flag that selects forward (ifftFlagR=0) or inverse (ifftFlagR=1) transform. */
          uint8_t bitReverseFlagR;                    /**< flag that enables (bitReverseFlagR=1) or disables (bitReverseFlagR=0) bit reversal of output. */
          uint32_t twidCoefRModifier;                 /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
    const q31_t *pTwiddleAReal;                       /**< points to the real twiddle factor table. */
    const q31_t *pTwiddleBReal;                       /**< points to the imag twiddle factor table. */
#if defined(ARM_MATH_MVEI)
    arm_cfft_instance_q31 cfftInst;
#else
    const arm_cfft_instance_q31 *pCfft;         /**< points to the complex FFT instance. */
#endif
  } arm_rfft_instance_q31;

  arm_status arm_rfft_init_q31(
        arm_rfft_instance_q31 * S,
        uint32_t fftLenReal,
        uint32_t ifftFlagR,
        uint32_t bitReverseFlag);

  void arm_rfft_q31(
  const arm_rfft_instance_q31 * S,
        q31_t * pSrc,
        q31_t * pDst);

  /**
   * @brief Instance structure for the floating-point RFFT/RIFFT function.
   */
  typedef struct
  {
          uint32_t fftLenReal;                        /**< length of the real FFT. */
          uint16_t fftLenBy2;                         /**< length of the complex FFT. */
          uint8_t ifftFlagR;                          /**< flag that selects forward (ifftFlagR=0) or inverse (ifftFlagR=1) transform. */
          uint8_t bitReverseFlagR;                    /**< flag that enables (bitReverseFlagR=1) or disables (bitReverseFlagR=0) bit reversal of output. */
          uint32_t twidCoefRModifier;                     /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
    const float32_t *pTwiddleAReal;                   /**< points to the real twiddle factor table. */
    const float32_t *pTwiddleBReal;                   /**< points to the imag twiddle factor table. */
          arm_cfft_radix4_instance_f32 *pCfft;        /**< points to the complex FFT instance. */
  } arm_rfft_instance_f32;

  arm_status arm_rfft_init_f32(
        arm_rfft_instance_f32 * S,
        arm_cfft_radix4_instance_f32 * S_CFFT,
        uint32_t fftLenReal,
        uint32_t ifftFlagR,
        uint32_t bitReverseFlag);

  void arm_rfft_f32(
  const arm_rfft_instance_f32 * S,
        float32_t * pSrc,
        float32_t * pDst);

  /**
   * @brief Instance structure for the Double Precision Floating-point RFFT/RIFFT function.
   */
typedef struct
  {
          arm_cfft_instance_f64 Sint;      /**< Internal CFFT structure. */
          uint16_t fftLenRFFT;             /**< length of the real sequence */
    const float64_t * pTwiddleRFFT;        /**< Twiddle factors real stage  */
  } arm_rfft_fast_instance_f64 ;

arm_status arm_rfft_fast_init_f64 (
         arm_rfft_fast_instance_f64 * S,
         uint16_t fftLen);


void arm_rfft_fast_f64(
    arm_rfft_fast_instance_f64 * S,
    float64_t * p, float64_t * pOut,
    uint8_t ifftFlag);


  /**
   * @brief Instance structure for the floating-point RFFT/RIFFT function.
   */
typedef struct
  {
          arm_cfft_instance_f32 Sint;      /**< Internal CFFT structure. */
          uint16_t fftLenRFFT;             /**< length of the real sequence */
    const float32_t * pTwiddleRFFT;        /**< Twiddle factors real stage  */
  } arm_rfft_fast_instance_f32 ;

arm_status arm_rfft_fast_init_f32 (
         arm_rfft_fast_instance_f32 * S,
         uint16_t fftLen);


  void arm_rfft_fast_f32(
        const arm_rfft_fast_instance_f32 * S,
        float32_t * p, float32_t * pOut,
        uint8_t ifftFlag);

  /**
   * @brief Instance structure for the floating-point DCT4/IDCT4 function.
   */
  typedef struct
  {
          uint16_t N;                          /**< length of the DCT4. */
          uint16_t Nby2;                       /**< half of the length of the DCT4. */
          float32_t normalize;                 /**< normalizing factor. */
    const float32_t *pTwiddle;                 /**< points to the twiddle factor table. */
    const float32_t *pCosFactor;               /**< points to the cosFactor table. */
          arm_rfft_instance_f32 *pRfft;        /**< points to the real FFT instance. */
          arm_cfft_radix4_instance_f32 *pCfft; /**< points to the complex FFT instance. */
  } arm_dct4_instance_f32;


  /**
   * @brief  Initialization function for the floating-point DCT4/IDCT4.
   * @param[in,out] S          points to an instance of floating-point DCT4/IDCT4 structure.
   * @param[in]     S_RFFT     points to an instance of floating-point RFFT/RIFFT structure.
   * @param[in]     S_CFFT     points to an instance of floating-point CFFT/CIFFT structure.
   * @param[in]     N          length of the DCT4.
   * @param[in]     Nby2       half of the length of the DCT4.
   * @param[in]     normalize  normalizing factor.
   * @return      arm_status function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_ARGUMENT_ERROR if <code>fftLenReal</code> is not a supported transform length.
   */
  arm_status arm_dct4_init_f32(
        arm_dct4_instance_f32 * S,
        arm_rfft_instance_f32 * S_RFFT,
        arm_cfft_radix4_instance_f32 * S_CFFT,
        uint16_t N,
        uint16_t Nby2,
        float32_t normalize);


  /**
   * @brief Processing function for the floating-point DCT4/IDCT4.
   * @param[in]     S              points to an instance of the floating-point DCT4/IDCT4 structure.
   * @param[in]     pState         points to state buffer.
   * @param[in,out] pInlineBuffer  points to the in-place input and output buffer.
   */
  void arm_dct4_f32(
  const arm_dct4_instance_f32 * S,
        float32_t * pState,
        float32_t * pInlineBuffer);


  /**
   * @brief Instance structure for the Q31 DCT4/IDCT4 function.
   */
  typedef struct
  {
          uint16_t N;                          /**< length of the DCT4. */
          uint16_t Nby2;                       /**< half of the length of the DCT4. */
          q31_t normalize;                     /**< normalizing factor. */
    const q31_t *pTwiddle;                     /**< points to the twiddle factor table. */
    const q31_t *pCosFactor;                   /**< points to the cosFactor table. */
          arm_rfft_instance_q31 *pRfft;        /**< points to the real FFT instance. */
          arm_cfft_radix4_instance_q31 *pCfft; /**< points to the complex FFT instance. */
  } arm_dct4_instance_q31;


  /**
   * @brief  Initialization function for the Q31 DCT4/IDCT4.
   * @param[in,out] S          points to an instance of Q31 DCT4/IDCT4 structure.
   * @param[in]     S_RFFT     points to an instance of Q31 RFFT/RIFFT structure
   * @param[in]     S_CFFT     points to an instance of Q31 CFFT/CIFFT structure
   * @param[in]     N          length of the DCT4.
   * @param[in]     Nby2       half of the length of the DCT4.
   * @param[in]     normalize  normalizing factor.
   * @return      arm_status function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_ARGUMENT_ERROR if <code>N</code> is not a supported transform length.
   */
  arm_status arm_dct4_init_q31(
        arm_dct4_instance_q31 * S,
        arm_rfft_instance_q31 * S_RFFT,
        arm_cfft_radix4_instance_q31 * S_CFFT,
        uint16_t N,
        uint16_t Nby2,
        q31_t normalize);


  /**
   * @brief Processing function for the Q31 DCT4/IDCT4.
   * @param[in]     S              points to an instance of the Q31 DCT4 structure.
   * @param[in]     pState         points to state buffer.
   * @param[in,out] pInlineBuffer  points to the in-place input and output buffer.
   */
  void arm_dct4_q31(
  const arm_dct4_instance_q31 * S,
        q31_t * pState,
        q31_t * pInlineBuffer);


  /**
   * @brief Instance structure for the Q15 DCT4/IDCT4 function.
   */
  typedef struct
  {
          uint16_t N;                          /**< length of the DCT4. */
          uint16_t Nby2;                       /**< half of the length of the DCT4. */
          q15_t normalize;                     /**< normalizing factor. */
    const q15_t *pTwiddle;                     /**< points to the twiddle factor table. */
    const q15_t *pCosFactor;                   /**< points to the cosFactor table. */
          arm_rfft_instance_q15 *pRfft;        /**< points to the real FFT instance. */
          arm_cfft_radix4_instance_q15 *pCfft; /**< points to the complex FFT instance. */
  } arm_dct4_instance_q15;


  /**
   * @brief  Initialization function for the Q15 DCT4/IDCT4.
   * @param[in,out] S          points to an instance of Q15 DCT4/IDCT4 structure.
   * @param[in]     S_RFFT     points to an instance of Q15 RFFT/RIFFT structure.
   * @param[in]     S_CFFT     points to an instance of Q15 CFFT/CIFFT structure.
   * @param[in]     N          length of the DCT4.
   * @param[in]     Nby2       half of the length of the DCT4.
   * @param[in]     normalize  normalizing factor.
   * @return      arm_status function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_ARGUMENT_ERROR if <code>N</code> is not a supported transform length.
   */
  arm_status arm_dct4_init_q15(
        arm_dct4_instance_q15 * S,
        arm_rfft_instance_q15 * S_RFFT,
        arm_cfft_radix4_instance_q15 * S_CFFT,
        uint16_t N,
        uint16_t Nby2,
        q15_t normalize);


  /**
   * @brief Processing function for the Q15 DCT4/IDCT4.
   * @param[in]     S              points to an instance of the Q15 DCT4 structure.
   * @param[in]     pState         points to state buffer.
   * @param[in,out] pInlineBuffer  points to the in-place input and output buffer.
   */
  void arm_dct4_q15(
  const arm_dct4_instance_q15 * S,
        q15_t * pState,
        q15_t * pInlineBuffer);


  /**
   * @brief Floating-point vector addition.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_add_f32(
  const float32_t * pSrcA,
  const float32_t * pSrcB,
        float32_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Q7 vector addition.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_add_q7(
  const q7_t * pSrcA,
  const q7_t * pSrcB,
        q7_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Q15 vector addition.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_add_q15(
  const q15_t * pSrcA,
  const q15_t * pSrcB,
        q15_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Q31 vector addition.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_add_q31(
  const q31_t * pSrcA,
  const q31_t * pSrcB,
        q31_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Floating-point vector subtraction.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_sub_f32(
  const float32_t * pSrcA,
  const float32_t * pSrcB,
        float32_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Q7 vector subtraction.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_sub_q7(
  const q7_t * pSrcA,
  const q7_t * pSrcB,
        q7_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Q15 vector subtraction.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_sub_q15(
  const q15_t * pSrcA,
  const q15_t * pSrcB,
        q15_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Q31 vector subtraction.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_sub_q31(
  const q31_t * pSrcA,
  const q31_t * pSrcB,
        q31_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Multiplies a floating-point vector by a scalar.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  scale      scale factor to be applied
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_scale_f32(
  const float32_t * pSrc,
        float32_t scale,
        float32_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Multiplies a Q7 vector by a scalar.
   * @param[in]  pSrc        points to the input vector
   * @param[in]  scaleFract  fractional portion of the scale value
   * @param[in]  shift       number of bits to shift the result by
   * @param[out] pDst        points to the output vector
   * @param[in]  blockSize   number of samples in the vector
   */
  void arm_scale_q7(
  const q7_t * pSrc,
        q7_t scaleFract,
        int8_t shift,
        q7_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Multiplies a Q15 vector by a scalar.
   * @param[in]  pSrc        points to the input vector
   * @param[in]  scaleFract  fractional portion of the scale value
   * @param[in]  shift       number of bits to shift the result by
   * @param[out] pDst        points to the output vector
   * @param[in]  blockSize   number of samples in the vector
   */
  void arm_scale_q15(
  const q15_t * pSrc,
        q15_t scaleFract,
        int8_t shift,
        q15_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Multiplies a Q31 vector by a scalar.
   * @param[in]  pSrc        points to the input vector
   * @param[in]  scaleFract  fractional portion of the scale value
   * @param[in]  shift       number of bits to shift the result by
   * @param[out] pDst        points to the output vector
   * @param[in]  blockSize   number of samples in the vector
   */
  void arm_scale_q31(
  const q31_t * pSrc,
        q31_t scaleFract,
        int8_t shift,
        q31_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Q7 vector absolute value.
   * @param[in]  pSrc       points to the input buffer
   * @param[out] pDst       points to the output buffer
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_abs_q7(
  const q7_t * pSrc,
        q7_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Floating-point vector absolute value.
   * @param[in]  pSrc       points to the input buffer
   * @param[out] pDst       points to the output buffer
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_abs_f32(
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Q15 vector absolute value.
   * @param[in]  pSrc       points to the input buffer
   * @param[out] pDst       points to the output buffer
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_abs_q15(
  const q15_t * pSrc,
        q15_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Q31 vector absolute value.
   * @param[in]  pSrc       points to the input buffer
   * @param[out] pDst       points to the output buffer
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_abs_q31(
  const q31_t * pSrc,
        q31_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Dot product of floating-point vectors.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[in]  blockSize  number of samples in each vector
   * @param[out] result     output result returned here
   */
  void arm_dot_prod_f32(
  const float32_t * pSrcA,
  const float32_t * pSrcB,
        uint32_t blockSize,
        float32_t * result);


  /**
   * @brief Dot product of Q7 vectors.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[in]  blockSize  number of samples in each vector
   * @param[out] result     output result returned here
   */
  void arm_dot_prod_q7(
  const q7_t * pSrcA,
  const q7_t * pSrcB,
        uint32_t blockSize,
        q31_t * result);


  /**
   * @brief Dot product of Q15 vectors.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[in]  blockSize  number of samples in each vector
   * @param[out] result     output result returned here
   */
  void arm_dot_prod_q15(
  const q15_t * pSrcA,
  const q15_t * pSrcB,
        uint32_t blockSize,
        q63_t * result);


  /**
   * @brief Dot product of Q31 vectors.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[in]  blockSize  number of samples in each vector
   * @param[out] result     output result returned here
   */
  void arm_dot_prod_q31(
  const q31_t * pSrcA,
  const q31_t * pSrcB,
        uint32_t blockSize,
        q63_t * result);


  /**
   * @brief  Shifts the elements of a Q7 vector a specified number of bits.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  shiftBits  number of bits to shift.  A positive value shifts left; a negative value shifts right.
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_shift_q7(
  const q7_t * pSrc,
        int8_t shiftBits,
        q7_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Shifts the elements of a Q15 vector a specified number of bits.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  shiftBits  number of bits to shift.  A positive value shifts left; a negative value shifts right.
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_shift_q15(
  const q15_t * pSrc,
        int8_t shiftBits,
        q15_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Shifts the elements of a Q31 vector a specified number of bits.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  shiftBits  number of bits to shift.  A positive value shifts left; a negative value shifts right.
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_shift_q31(
  const q31_t * pSrc,
        int8_t shiftBits,
        q31_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Adds a constant offset to a floating-point vector.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  offset     is the offset to be added
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_offset_f32(
  const float32_t * pSrc,
        float32_t offset,
        float32_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Adds a constant offset to a Q7 vector.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  offset     is the offset to be added
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_offset_q7(
  const q7_t * pSrc,
        q7_t offset,
        q7_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Adds a constant offset to a Q15 vector.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  offset     is the offset to be added
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_offset_q15(
  const q15_t * pSrc,
        q15_t offset,
        q15_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Adds a constant offset to a Q31 vector.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  offset     is the offset to be added
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_offset_q31(
  const q31_t * pSrc,
        q31_t offset,
        q31_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Negates the elements of a floating-point vector.
   * @param[in]  pSrc       points to the input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_negate_f32(
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Negates the elements of a Q7 vector.
   * @param[in]  pSrc       points to the input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_negate_q7(
  const q7_t * pSrc,
        q7_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Negates the elements of a Q15 vector.
   * @param[in]  pSrc       points to the input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_negate_q15(
  const q15_t * pSrc,
        q15_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Negates the elements of a Q31 vector.
   * @param[in]  pSrc       points to the input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_negate_q31(
  const q31_t * pSrc,
        q31_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Copies the elements of a floating-point vector.
   * @param[in]  pSrc       input pointer
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_copy_f32(
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Copies the elements of a Q7 vector.
   * @param[in]  pSrc       input pointer
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_copy_q7(
  const q7_t * pSrc,
        q7_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Copies the elements of a Q15 vector.
   * @param[in]  pSrc       input pointer
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_copy_q15(
  const q15_t * pSrc,
        q15_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Copies the elements of a Q31 vector.
   * @param[in]  pSrc       input pointer
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_copy_q31(
  const q31_t * pSrc,
        q31_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Fills a constant value into a floating-point vector.
   * @param[in]  value      input value to be filled
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_fill_f32(
        float32_t value,
        float32_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Fills a constant value into a Q7 vector.
   * @param[in]  value      input value to be filled
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_fill_q7(
        q7_t value,
        q7_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Fills a constant value into a Q15 vector.
   * @param[in]  value      input value to be filled
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_fill_q15(
        q15_t value,
        q15_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Fills a constant value into a Q31 vector.
   * @param[in]  value      input value to be filled
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_fill_q31(
        q31_t value,
        q31_t * pDst,
        uint32_t blockSize);


/**
 * @brief Convolution of floating-point sequences.
 * @param[in]  pSrcA    points to the first input sequence.
 * @param[in]  srcALen  length of the first input sequence.
 * @param[in]  pSrcB    points to the second input sequence.
 * @param[in]  srcBLen  length of the second input sequence.
 * @param[out] pDst     points to the location where the output result is written.  Length srcALen+srcBLen-1.
 */
  void arm_conv_f32(
  const float32_t * pSrcA,
        uint32_t srcALen,
  const float32_t * pSrcB,
        uint32_t srcBLen,
        float32_t * pDst);


  /**
   * @brief Convolution of Q15 sequences.
   * @param[in]  pSrcA      points to the first input sequence.
   * @param[in]  srcALen    length of the first input sequence.
   * @param[in]  pSrcB      points to the second input sequence.
   * @param[in]  srcBLen    length of the second input sequence.
   * @param[out] pDst       points to the block of output data  Length srcALen+srcBLen-1.
   * @param[in]  pScratch1  points to scratch buffer of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
   * @param[in]  pScratch2  points to scratch buffer of size min(srcALen, srcBLen).
   */
  void arm_conv_opt_q15(
  const q15_t * pSrcA,
        uint32_t srcALen,
  const q15_t * pSrcB,
        uint32_t srcBLen,
        q15_t * pDst,
        q15_t * pScratch1,
        q15_t * pScratch2);


/**
 * @brief Convolution of Q15 sequences.
 * @param[in]  pSrcA    points to the first input sequence.
 * @param[in]  srcALen  length of the first input sequence.
 * @param[in]  pSrcB    points to the second input sequence.
 * @param[in]  srcBLen  length of the second input sequence.
 * @param[out] pDst     points to the location where the output result is written.  Length srcALen+srcBLen-1.
 */
  void arm_conv_q15(
  const q15_t * pSrcA,
        uint32_t srcALen,
  const q15_t * pSrcB,
        uint32_t srcBLen,
        q15_t * pDst);


  /**
   * @brief Convolution of Q15 sequences (fast version) for Cortex-M3 and Cortex-M4
   * @param[in]  pSrcA    points to the first input sequence.
   * @param[in]  srcALen  length of the first input sequence.
   * @param[in]  pSrcB    points to the second input sequence.
   * @param[in]  srcBLen  length of the second input sequence.
   * @param[out] pDst     points to the block of output data  Length srcALen+srcBLen-1.
   */
  void arm_conv_fast_q15(
  const q15_t * pSrcA,
        uint32_t srcALen,
  const q15_t * pSrcB,
        uint32_t srcBLen,
        q15_t * pDst);


  /**
   * @brief Convolution of Q15 sequences (fast version) for Cortex-M3 and Cortex-M4
   * @param[in]  pSrcA      points to the first input sequence.
   * @param[in]  srcALen    length of the first input sequence.
   * @param[in]  pSrcB      points to the second input sequence.
   * @param[in]  srcBLen    length of the second input sequence.
   * @param[out] pDst       points to the block of output data  Length srcALen+srcBLen-1.
   * @param[in]  pScratch1  points to scratch buffer of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
   * @param[in]  pScratch2  points to scratch buffer of size min(srcALen, srcBLen).
   */
  void arm_conv_fast_opt_q15(
  const q15_t * pSrcA,
        uint32_t srcALen,
  const q15_t * pSrcB,
        uint32_t srcBLen,
        q15_t * pDst,
        q15_t * pScratch1,
        q15_t * pScratch2);


  /**
   * @brief Convolution of Q31 sequences.
   * @param[in]  pSrcA    points to the first input sequence.
   * @param[in]  srcALen  length of the first input sequence.
   * @param[in]  pSrcB    points to the second input sequence.
   * @param[in]  srcBLen  length of the second input sequence.
   * @param[out] pDst     points to the block of output data  Length srcALen+srcBLen-1.
   */
  void arm_conv_q31(
  const q31_t * pSrcA,
        uint32_t srcALen,
  const q31_t * pSrcB,
        uint32_t srcBLen,
        q31_t * pDst);


  /**
   * @brief Convolution of Q31 sequences (fast version) for Cortex-M3 and Cortex-M4
   * @param[in]  pSrcA    points to the first input sequence.
   * @param[in]  srcALen  length of the first input sequence.
   * @param[in]  pSrcB    points to the second input sequence.
   * @param[in]  srcBLen  length of the second input sequence.
   * @param[out] pDst     points to the block of output data  Length srcALen+srcBLen-1.
   */
  void arm_conv_fast_q31(
  const q31_t * pSrcA,
        uint32_t srcALen,
  const q31_t * pSrcB,
        uint32_t srcBLen,
        q31_t * pDst);


    /**
   * @brief Convolution of Q7 sequences.
   * @param[in]  pSrcA      points to the first input sequence.
   * @param[in]  srcALen    length of the first input sequence.
   * @param[in]  pSrcB      points to the second input sequence.
   * @param[in]  srcBLen    length of the second input sequence.
   * @param[out] pDst       points to the block of output data  Length srcALen+srcBLen-1.
   * @param[in]  pScratch1  points to scratch buffer(of type q15_t) of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
   * @param[in]  pScratch2  points to scratch buffer (of type q15_t) of size min(srcALen, srcBLen).
   */
  void arm_conv_opt_q7(
  const q7_t * pSrcA,
        uint32_t srcALen,
  const q7_t * pSrcB,
        uint32_t srcBLen,
        q7_t * pDst,
        q15_t * pScratch1,
        q15_t * pScratch2);


  /**
   * @brief Convolution of Q7 sequences.
   * @param[in]  pSrcA    points to the first input sequence.
   * @param[in]  srcALen  length of the first input sequence.
   * @param[in]  pSrcB    points to the second input sequence.
   * @param[in]  srcBLen  length of the second input sequence.
   * @param[out] pDst     points to the block of output data  Length srcALen+srcBLen-1.
   */
  void arm_conv_q7(
  const q7_t * pSrcA,
        uint32_t srcALen,
  const q7_t * pSrcB,
        uint32_t srcBLen,
        q7_t * pDst);


  /**
   * @brief Partial convolution of floating-point sequences.
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_f32(
  const float32_t * pSrcA,
        uint32_t srcALen,
  const float32_t * pSrcB,
        uint32_t srcBLen,
        float32_t * pDst,
        uint32_t firstIndex,
        uint32_t numPoints);


  /**
   * @brief Partial convolution of Q15 sequences.
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @param[in]  pScratch1   points to scratch buffer of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
   * @param[in]  pScratch2   points to scratch buffer of size min(srcALen, srcBLen).
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_opt_q15(
  const q15_t * pSrcA,
        uint32_t srcALen,
  const q15_t * pSrcB,
        uint32_t srcBLen,
        q15_t * pDst,
        uint32_t firstIndex,
        uint32_t numPoints,
        q15_t * pScratch1,
        q15_t * pScratch2);


  /**
   * @brief Partial convolution of Q15 sequences.
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_q15(
  const q15_t * pSrcA,
        uint32_t srcALen,
  const q15_t * pSrcB,
        uint32_t srcBLen,
        q15_t * pDst,
        uint32_t firstIndex,
        uint32_t numPoints);


  /**
   * @brief Partial convolution of Q15 sequences (fast version) for Cortex-M3 and Cortex-M4
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_fast_q15(
  const q15_t * pSrcA,
        uint32_t srcALen,
  const q15_t * pSrcB,
        uint32_t srcBLen,
        q15_t * pDst,
        uint32_t firstIndex,
        uint32_t numPoints);


  /**
   * @brief Partial convolution of Q15 sequences (fast version) for Cortex-M3 and Cortex-M4
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @param[in]  pScratch1   points to scratch buffer of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
   * @param[in]  pScratch2   points to scratch buffer of size min(srcALen, srcBLen).
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_fast_opt_q15(
  const q15_t * pSrcA,
        uint32_t srcALen,
  const q15_t * pSrcB,
        uint32_t srcBLen,
        q15_t * pDst,
        uint32_t firstIndex,
        uint32_t numPoints,
        q15_t * pScratch1,
        q15_t * pScratch2);


  /**
   * @brief Partial convolution of Q31 sequences.
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_q31(
  const q31_t * pSrcA,
        uint32_t srcALen,
  const q31_t * pSrcB,
        uint32_t srcBLen,
        q31_t * pDst,
        uint32_t firstIndex,
        uint32_t numPoints);


  /**
   * @brief Partial convolution of Q31 sequences (fast version) for Cortex-M3 and Cortex-M4
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_fast_q31(
  const q31_t * pSrcA,
        uint32_t srcALen,
  const q31_t * pSrcB,
        uint32_t srcBLen,
        q31_t * pDst,
        uint32_t firstIndex,
        uint32_t numPoints);


  /**
   * @brief Partial convolution of Q7 sequences
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @param[in]  pScratch1   points to scratch buffer(of type q15_t) of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
   * @param[in]  pScratch2   points to scratch buffer (of type q15_t) of size min(srcALen, srcBLen).
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_opt_q7(
  const q7_t * pSrcA,
        uint32_t srcALen,
  const q7_t * pSrcB,
        uint32_t srcBLen,
        q7_t * pDst,
        uint32_t firstIndex,
        uint32_t numPoints,
        q15_t * pScratch1,
        q15_t * pScratch2);


/**
   * @brief Partial convolution of Q7 sequences.
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_q7(
  const q7_t * pSrcA,
        uint32_t srcALen,
  const q7_t * pSrcB,
        uint32_t srcBLen,
        q7_t * pDst,
        uint32_t firstIndex,
        uint32_t numPoints);


  /**
   * @brief Instance structure for the Q15 FIR decimator.
   */
  typedef struct
  {
          uint8_t M;                  /**< decimation factor. */
          uint16_t numTaps;           /**< number of coefficients in the filter. */
    const q15_t *pCoeffs;             /**< points to the coefficient array. The array is of length numTaps.*/
          q15_t *pState;              /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
  } arm_fir_decimate_instance_q15;

  /**
   * @brief Instance structure for the Q31 FIR decimator.
   */
  typedef struct
  {
          uint8_t M;                  /**< decimation factor. */
          uint16_t numTaps;           /**< number of coefficients in the filter. */
    const q31_t *pCoeffs;             /**< points to the coefficient array. The array is of length numTaps.*/
          q31_t *pState;              /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
  } arm_fir_decimate_instance_q31;

/**
  @brief Instance structure for floating-point FIR decimator.
 */
typedef struct
  {
          uint8_t M;                  /**< decimation factor. */
          uint16_t numTaps;           /**< number of coefficients in the filter. */
    const float32_t *pCoeffs;         /**< points to the coefficient array. The array is of length numTaps.*/
          float32_t *pState;          /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
  } arm_fir_decimate_instance_f32;


/**
  @brief         Processing function for floating-point FIR decimator.
  @param[in]     S         points to an instance of the floating-point FIR decimator structure
  @param[in]     pSrc      points to the block of input data
  @param[out]    pDst      points to the block of output data
  @param[in]     blockSize number of samples to process
 */
void arm_fir_decimate_f32(
  const arm_fir_decimate_instance_f32 * S,
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize);


/**
  @brief         Initialization function for the floating-point FIR decimator.
  @param[in,out] S          points to an instance of the floating-point FIR decimator structure
  @param[in]     numTaps    number of coefficients in the filter
  @param[in]     M          decimation factor
  @param[in]     pCoeffs    points to the filter coefficients
  @param[in]     pState     points to the state buffer
  @param[in]     blockSize  number of input samples to process per call
  @return        execution status
                   - \ref ARM_MATH_SUCCESS      : Operation successful
                   - \ref ARM_MATH_LENGTH_ERROR : <code>blockSize</code> is not a multiple of <code>M</code>
 */
arm_status arm_fir_decimate_init_f32(
        arm_fir_decimate_instance_f32 * S,
        uint16_t numTaps,
        uint8_t M,
  const float32_t * pCoeffs,
        float32_t * pState,
        uint32_t blockSize);


  /**
   * @brief Processing function for the Q15 FIR decimator.
   * @param[in]  S          points to an instance of the Q15 FIR decimator structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of input samples to process per call.
   */
  void arm_fir_decimate_q15(
  const arm_fir_decimate_instance_q15 * S,
  const q15_t * pSrc,
        q15_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Processing function for the Q15 FIR decimator (fast variant) for Cortex-M3 and Cortex-M4.
   * @param[in]  S          points to an instance of the Q15 FIR decimator structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of input samples to process per call.
   */
  void arm_fir_decimate_fast_q15(
  const arm_fir_decimate_instance_q15 * S,
  const q15_t * pSrc,
        q15_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q15 FIR decimator.
   * @param[in,out] S          points to an instance of the Q15 FIR decimator structure.
   * @param[in]     numTaps    number of coefficients in the filter.
   * @param[in]     M          decimation factor.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of input samples to process per call.
   * @return    The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_LENGTH_ERROR if
   * <code>blockSize</code> is not a multiple of <code>M</code>.
   */
  arm_status arm_fir_decimate_init_q15(
        arm_fir_decimate_instance_q15 * S,
        uint16_t numTaps,
        uint8_t M,
  const q15_t * pCoeffs,
        q15_t * pState,
        uint32_t blockSize);


  /**
   * @brief Processing function for the Q31 FIR decimator.
   * @param[in]  S     points to an instance of the Q31 FIR decimator structure.
   * @param[in]  pSrc  points to the block of input data.
   * @param[out] pDst  points to the block of output data
   * @param[in] blockSize number of input samples to process per call.
   */
  void arm_fir_decimate_q31(
  const arm_fir_decimate_instance_q31 * S,
  const q31_t * pSrc,
        q31_t * pDst,
        uint32_t blockSize);

  /**
   * @brief Processing function for the Q31 FIR decimator (fast variant) for Cortex-M3 and Cortex-M4.
   * @param[in]  S          points to an instance of the Q31 FIR decimator structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of input samples to process per call.
   */
  void arm_fir_decimate_fast_q31(
  const arm_fir_decimate_instance_q31 * S,
  const q31_t * pSrc,
        q31_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q31 FIR decimator.
   * @param[in,out] S          points to an instance of the Q31 FIR decimator structure.
   * @param[in]     numTaps    number of coefficients in the filter.
   * @param[in]     M          decimation factor.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of input samples to process per call.
   * @return    The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_LENGTH_ERROR if
   * <code>blockSize</code> is not a multiple of <code>M</code>.
   */
  arm_status arm_fir_decimate_init_q31(
        arm_fir_decimate_instance_q31 * S,
        uint16_t numTaps,
        uint8_t M,
  const q31_t * pCoeffs,
        q31_t * pState,
        uint32_t blockSize);


  /**
   * @brief Instance structure for the Q15 FIR interpolator.
   */
  typedef struct
  {
        uint8_t L;                      /**< upsample factor. */
        uint16_t phaseLength;           /**< length of each polyphase filter component. */
  const q15_t *pCoeffs;                 /**< points to the coefficient array. The array is of length L*phaseLength. */
        q15_t *pState;                  /**< points to the state variable array. The array is of length blockSize+phaseLength-1. */
  } arm_fir_interpolate_instance_q15;

  /**
   * @brief Instance structure for the Q31 FIR interpolator.
   */
  typedef struct
  {
        uint8_t L;                      /**< upsample factor. */
        uint16_t phaseLength;           /**< length of each polyphase filter component. */
  const q31_t *pCoeffs;                 /**< points to the coefficient array. The array is of length L*phaseLength. */
        q31_t *pState;                  /**< points to the state variable array. The array is of length blockSize+phaseLength-1. */
  } arm_fir_interpolate_instance_q31;

  /**
   * @brief Instance structure for the floating-point FIR interpolator.
   */
  typedef struct
  {
        uint8_t L;                     /**< upsample factor. */
        uint16_t phaseLength;          /**< length of each polyphase filter component. */
  const float32_t *pCoeffs;            /**< points to the coefficient array. The array is of length L*phaseLength. */
        float32_t *pState;             /**< points to the state variable array. The array is of length phaseLength+numTaps-1. */
  } arm_fir_interpolate_instance_f32;


  /**
   * @brief Processing function for the Q15 FIR interpolator.
   * @param[in]  S          points to an instance of the Q15 FIR interpolator structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of input samples to process per call.
   */
  void arm_fir_interpolate_q15(
  const arm_fir_interpolate_instance_q15 * S,
  const q15_t * pSrc,
        q15_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q15 FIR interpolator.
   * @param[in,out] S          points to an instance of the Q15 FIR interpolator structure.
   * @param[in]     L          upsample factor.
   * @param[in]     numTaps    number of filter coefficients in the filter.
   * @param[in]     pCoeffs    points to the filter coefficient buffer.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of input samples to process per call.
   * @return        The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_LENGTH_ERROR if
   * the filter length <code>numTaps</code> is not a multiple of the interpolation factor <code>L</code>.
   */
  arm_status arm_fir_interpolate_init_q15(
        arm_fir_interpolate_instance_q15 * S,
        uint8_t L,
        uint16_t numTaps,
  const q15_t * pCoeffs,
        q15_t * pState,
        uint32_t blockSize);


  /**
   * @brief Processing function for the Q31 FIR interpolator.
   * @param[in]  S          points to an instance of the Q15 FIR interpolator structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of input samples to process per call.
   */
  void arm_fir_interpolate_q31(
  const arm_fir_interpolate_instance_q31 * S,
  const q31_t * pSrc,
        q31_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q31 FIR interpolator.
   * @param[in,out] S          points to an instance of the Q31 FIR interpolator structure.
   * @param[in]     L          upsample factor.
   * @param[in]     numTaps    number of filter coefficients in the filter.
   * @param[in]     pCoeffs    points to the filter coefficient buffer.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of input samples to process per call.
   * @return        The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_LENGTH_ERROR if
   * the filter length <code>numTaps</code> is not a multiple of the interpolation factor <code>L</code>.
   */
  arm_status arm_fir_interpolate_init_q31(
        arm_fir_interpolate_instance_q31 * S,
        uint8_t L,
        uint16_t numTaps,
  const q31_t * pCoeffs,
        q31_t * pState,
        uint32_t blockSize);


  /**
   * @brief Processing function for the floating-point FIR interpolator.
   * @param[in]  S          points to an instance of the floating-point FIR interpolator structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of input samples to process per call.
   */
  void arm_fir_interpolate_f32(
  const arm_fir_interpolate_instance_f32 * S,
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Initialization function for the floating-point FIR interpolator.
   * @param[in,out] S          points to an instance of the floating-point FIR interpolator structure.
   * @param[in]     L          upsample factor.
   * @param[in]     numTaps    number of filter coefficients in the filter.
   * @param[in]     pCoeffs    points to the filter coefficient buffer.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of input samples to process per call.
   * @return        The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_LENGTH_ERROR if
   * the filter length <code>numTaps</code> is not a multiple of the interpolation factor <code>L</code>.
   */
  arm_status arm_fir_interpolate_init_f32(
        arm_fir_interpolate_instance_f32 * S,
        uint8_t L,
        uint16_t numTaps,
  const float32_t * pCoeffs,
        float32_t * pState,
        uint32_t blockSize);


  /**
   * @brief Instance structure for the high precision Q31 Biquad cascade filter.
   */
  typedef struct
  {
          uint8_t numStages;       /**< number of 2nd order stages in the filter.  Overall order is 2*numStages. */
          q63_t *pState;           /**< points to the array of state coefficients.  The array is of length 4*numStages. */
    const q31_t *pCoeffs;          /**< points to the array of coefficients.  The array is of length 5*numStages. */
          uint8_t postShift;       /**< additional shift, in bits, applied to each output sample. */
  } arm_biquad_cas_df1_32x64_ins_q31;


  /**
   * @param[in]  S          points to an instance of the high precision Q31 Biquad cascade filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cas_df1_32x64_q31(
  const arm_biquad_cas_df1_32x64_ins_q31 * S,
  const q31_t * pSrc,
        q31_t * pDst,
        uint32_t blockSize);


  /**
   * @param[in,out] S          points to an instance of the high precision Q31 Biquad cascade filter structure.
   * @param[in]     numStages  number of 2nd order stages in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     postShift  shift to be applied to the output. Varies according to the coefficients format
   */
  void arm_biquad_cas_df1_32x64_init_q31(
        arm_biquad_cas_df1_32x64_ins_q31 * S,
        uint8_t numStages,
  const q31_t * pCoeffs,
        q63_t * pState,
        uint8_t postShift);


  /**
   * @brief Instance structure for the floating-point transposed direct form II Biquad cascade filter.
   */
  typedef struct
  {
          uint8_t numStages;         /**< number of 2nd order stages in the filter.  Overall order is 2*numStages. */
          float32_t *pState;         /**< points to the array of state coefficients.  The array is of length 2*numStages. */
    const float32_t *pCoeffs;        /**< points to the array of coefficients.  The array is of length 5*numStages. */
  } arm_biquad_cascade_df2T_instance_f32;

  /**
   * @brief Instance structure for the floating-point transposed direct form II Biquad cascade filter.
   */
  typedef struct
  {
          uint8_t numStages;         /**< number of 2nd order stages in the filter.  Overall order is 2*numStages. */
          float32_t *pState;         /**< points to the array of state coefficients.  The array is of length 4*numStages. */
    const float32_t *pCoeffs;        /**< points to the array of coefficients.  The array is of length 5*numStages. */
  } arm_biquad_cascade_stereo_df2T_instance_f32;

  /**
   * @brief Instance structure for the floating-point transposed direct form II Biquad cascade filter.
   */
  typedef struct
  {
          uint8_t numStages;         /**< number of 2nd order stages in the filter.  Overall order is 2*numStages. */
          float64_t *pState;         /**< points to the array of state coefficients.  The array is of length 2*numStages. */
    const float64_t *pCoeffs;        /**< points to the array of coefficients.  The array is of length 5*numStages. */
  } arm_biquad_cascade_df2T_instance_f64;


  /**
   * @brief Processing function for the floating-point transposed direct form II Biquad cascade filter.
   * @param[in]  S          points to an instance of the filter data structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cascade_df2T_f32(
  const arm_biquad_cascade_df2T_instance_f32 * S,
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Processing function for the floating-point transposed direct form II Biquad cascade filter. 2 channels
   * @param[in]  S          points to an instance of the filter data structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cascade_stereo_df2T_f32(
  const arm_biquad_cascade_stereo_df2T_instance_f32 * S,
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Processing function for the floating-point transposed direct form II Biquad cascade filter.
   * @param[in]  S          points to an instance of the filter data structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cascade_df2T_f64(
  const arm_biquad_cascade_df2T_instance_f64 * S,
  const float64_t * pSrc,
        float64_t * pDst,
        uint32_t blockSize);


#if defined(ARM_MATH_NEON) 
void arm_biquad_cascade_df2T_compute_coefs_f32(
  arm_biquad_cascade_df2T_instance_f32 * S,
  uint8_t numStages,
  float32_t * pCoeffs);
#endif
  /**
   * @brief  Initialization function for the floating-point transposed direct form II Biquad cascade filter.
   * @param[in,out] S          points to an instance of the filter data structure.
   * @param[in]     numStages  number of 2nd order stages in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   */
  void arm_biquad_cascade_df2T_init_f32(
        arm_biquad_cascade_df2T_instance_f32 * S,
        uint8_t numStages,
  const float32_t * pCoeffs,
        float32_t * pState);


  /**
   * @brief  Initialization function for the floating-point transposed direct form II Biquad cascade filter.
   * @param[in,out] S          points to an instance of the filter data structure.
   * @param[in]     numStages  number of 2nd order stages in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   */
  void arm_biquad_cascade_stereo_df2T_init_f32(
        arm_biquad_cascade_stereo_df2T_instance_f32 * S,
        uint8_t numStages,
  const float32_t * pCoeffs,
        float32_t * pState);


  /**
   * @brief  Initialization function for the floating-point transposed direct form II Biquad cascade filter.
   * @param[in,out] S          points to an instance of the filter data structure.
   * @param[in]     numStages  number of 2nd order stages in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   */
  void arm_biquad_cascade_df2T_init_f64(
        arm_biquad_cascade_df2T_instance_f64 * S,
        uint8_t numStages,
        const float64_t * pCoeffs,
        float64_t * pState);


  /**
   * @brief Instance structure for the Q15 FIR lattice filter.
   */
  typedef struct
  {
          uint16_t numStages;                  /**< number of filter stages. */
          q15_t *pState;                       /**< points to the state variable array. The array is of length numStages. */
    const q15_t *pCoeffs;                      /**< points to the coefficient array. The array is of length numStages. */
  } arm_fir_lattice_instance_q15;

  /**
   * @brief Instance structure for the Q31 FIR lattice filter.
   */
  typedef struct
  {
          uint16_t numStages;                  /**< number of filter stages. */
          q31_t *pState;                       /**< points to the state variable array. The array is of length numStages. */
    const q31_t *pCoeffs;                      /**< points to the coefficient array. The array is of length numStages. */
  } arm_fir_lattice_instance_q31;

  /**
   * @brief Instance structure for the floating-point FIR lattice filter.
   */
  typedef struct
  {
          uint16_t numStages;                  /**< number of filter stages. */
          float32_t *pState;                   /**< points to the state variable array. The array is of length numStages. */
    const float32_t *pCoeffs;                  /**< points to the coefficient array. The array is of length numStages. */
  } arm_fir_lattice_instance_f32;


  /**
   * @brief Initialization function for the Q15 FIR lattice filter.
   * @param[in] S          points to an instance of the Q15 FIR lattice structure.
   * @param[in] numStages  number of filter stages.
   * @param[in] pCoeffs    points to the coefficient buffer.  The array is of length numStages.
   * @param[in] pState     points to the state buffer.  The array is of length numStages.
   */
  void arm_fir_lattice_init_q15(
        arm_fir_lattice_instance_q15 * S,
        uint16_t numStages,
  const q15_t * pCoeffs,
        q15_t * pState);


  /**
   * @brief Processing function for the Q15 FIR lattice filter.
   * @param[in]  S          points to an instance of the Q15 FIR lattice structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_lattice_q15(
  const arm_fir_lattice_instance_q15 * S,
  const q15_t * pSrc,
        q15_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Initialization function for the Q31 FIR lattice filter.
   * @param[in] S          points to an instance of the Q31 FIR lattice structure.
   * @param[in] numStages  number of filter stages.
   * @param[in] pCoeffs    points to the coefficient buffer.  The array is of length numStages.
   * @param[in] pState     points to the state buffer.   The array is of length numStages.
   */
  void arm_fir_lattice_init_q31(
        arm_fir_lattice_instance_q31 * S,
        uint16_t numStages,
  const q31_t * pCoeffs,
        q31_t * pState);


  /**
   * @brief Processing function for the Q31 FIR lattice filter.
   * @param[in]  S          points to an instance of the Q31 FIR lattice structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_lattice_q31(
  const arm_fir_lattice_instance_q31 * S,
  const q31_t * pSrc,
        q31_t * pDst,
        uint32_t blockSize);


/**
 * @brief Initialization function for the floating-point FIR lattice filter.
 * @param[in] S          points to an instance of the floating-point FIR lattice structure.
 * @param[in] numStages  number of filter stages.
 * @param[in] pCoeffs    points to the coefficient buffer.  The array is of length numStages.
 * @param[in] pState     points to the state buffer.  The array is of length numStages.
 */
  void arm_fir_lattice_init_f32(
        arm_fir_lattice_instance_f32 * S,
        uint16_t numStages,
  const float32_t * pCoeffs,
        float32_t * pState);


  /**
   * @brief Processing function for the floating-point FIR lattice filter.
   * @param[in]  S          points to an instance of the floating-point FIR lattice structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_lattice_f32(
  const arm_fir_lattice_instance_f32 * S,
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Instance structure for the Q15 IIR lattice filter.
   */
  typedef struct
  {
          uint16_t numStages;                  /**< number of stages in the filter. */
          q15_t *pState;                       /**< points to the state variable array. The array is of length numStages+blockSize. */
          q15_t *pkCoeffs;                     /**< points to the reflection coefficient array. The array is of length numStages. */
          q15_t *pvCoeffs;                     /**< points to the ladder coefficient array. The array is of length numStages+1. */
  } arm_iir_lattice_instance_q15;

  /**
   * @brief Instance structure for the Q31 IIR lattice filter.
   */
  typedef struct
  {
          uint16_t numStages;                  /**< number of stages in the filter. */
          q31_t *pState;                       /**< points to the state variable array. The array is of length numStages+blockSize. */
          q31_t *pkCoeffs;                     /**< points to the reflection coefficient array. The array is of length numStages. */
          q31_t *pvCoeffs;                     /**< points to the ladder coefficient array. The array is of length numStages+1. */
  } arm_iir_lattice_instance_q31;

  /**
   * @brief Instance structure for the floating-point IIR lattice filter.
   */
  typedef struct
  {
          uint16_t numStages;                  /**< number of stages in the filter. */
          float32_t *pState;                   /**< points to the state variable array. The array is of length numStages+blockSize. */
          float32_t *pkCoeffs;                 /**< points to the reflection coefficient array. The array is of length numStages. */
          float32_t *pvCoeffs;                 /**< points to the ladder coefficient array. The array is of length numStages+1. */
  } arm_iir_lattice_instance_f32;


  /**
   * @brief Processing function for the floating-point IIR lattice filter.
   * @param[in]  S          points to an instance of the floating-point IIR lattice structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_iir_lattice_f32(
  const arm_iir_lattice_instance_f32 * S,
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Initialization function for the floating-point IIR lattice filter.
   * @param[in] S          points to an instance of the floating-point IIR lattice structure.
   * @param[in] numStages  number of stages in the filter.
   * @param[in] pkCoeffs   points to the reflection coefficient buffer.  The array is of length numStages.
   * @param[in] pvCoeffs   points to the ladder coefficient buffer.  The array is of length numStages+1.
   * @param[in] pState     points to the state buffer.  The array is of length numStages+blockSize-1.
   * @param[in] blockSize  number of samples to process.
   */
  void arm_iir_lattice_init_f32(
        arm_iir_lattice_instance_f32 * S,
        uint16_t numStages,
        float32_t * pkCoeffs,
        float32_t * pvCoeffs,
        float32_t * pState,
        uint32_t blockSize);


  /**
   * @brief Processing function for the Q31 IIR lattice filter.
   * @param[in]  S          points to an instance of the Q31 IIR lattice structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_iir_lattice_q31(
  const arm_iir_lattice_instance_q31 * S,
  const q31_t * pSrc,
        q31_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Initialization function for the Q31 IIR lattice filter.
   * @param[in] S          points to an instance of the Q31 IIR lattice structure.
   * @param[in] numStages  number of stages in the filter.
   * @param[in] pkCoeffs   points to the reflection coefficient buffer.  The array is of length numStages.
   * @param[in] pvCoeffs   points to the ladder coefficient buffer.  The array is of length numStages+1.
   * @param[in] pState     points to the state buffer.  The array is of length numStages+blockSize.
   * @param[in] blockSize  number of samples to process.
   */
  void arm_iir_lattice_init_q31(
        arm_iir_lattice_instance_q31 * S,
        uint16_t numStages,
        q31_t * pkCoeffs,
        q31_t * pvCoeffs,
        q31_t * pState,
        uint32_t blockSize);


  /**
   * @brief Processing function for the Q15 IIR lattice filter.
   * @param[in]  S          points to an instance of the Q15 IIR lattice structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_iir_lattice_q15(
  const arm_iir_lattice_instance_q15 * S,
  const q15_t * pSrc,
        q15_t * pDst,
        uint32_t blockSize);


/**
 * @brief Initialization function for the Q15 IIR lattice filter.
 * @param[in] S          points to an instance of the fixed-point Q15 IIR lattice structure.
 * @param[in] numStages  number of stages in the filter.
 * @param[in] pkCoeffs   points to reflection coefficient buffer.  The array is of length numStages.
 * @param[in] pvCoeffs   points to ladder coefficient buffer.  The array is of length numStages+1.
 * @param[in] pState     points to state buffer.  The array is of length numStages+blockSize.
 * @param[in] blockSize  number of samples to process per call.
 */
  void arm_iir_lattice_init_q15(
        arm_iir_lattice_instance_q15 * S,
        uint16_t numStages,
        q15_t * pkCoeffs,
        q15_t * pvCoeffs,
        q15_t * pState,
        uint32_t blockSize);


  /**
   * @brief Instance structure for the floating-point LMS filter.
   */
  typedef struct
  {
          uint16_t numTaps;    /**< number of coefficients in the filter. */
          float32_t *pState;   /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
          float32_t *pCoeffs;  /**< points to the coefficient array. The array is of length numTaps. */
          float32_t mu;        /**< step size that controls filter coefficient updates. */
  } arm_lms_instance_f32;


  /**
   * @brief Processing function for floating-point LMS filter.
   * @param[in]  S          points to an instance of the floating-point LMS filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[in]  pRef       points to the block of reference data.
   * @param[out] pOut       points to the block of output data.
   * @param[out] pErr       points to the block of error data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_lms_f32(
  const arm_lms_instance_f32 * S,
  const float32_t * pSrc,
        float32_t * pRef,
        float32_t * pOut,
        float32_t * pErr,
        uint32_t blockSize);


  /**
   * @brief Initialization function for floating-point LMS filter.
   * @param[in] S          points to an instance of the floating-point LMS filter structure.
   * @param[in] numTaps    number of filter coefficients.
   * @param[in] pCoeffs    points to the coefficient buffer.
   * @param[in] pState     points to state buffer.
   * @param[in] mu         step size that controls filter coefficient updates.
   * @param[in] blockSize  number of samples to process.
   */
  void arm_lms_init_f32(
        arm_lms_instance_f32 * S,
        uint16_t numTaps,
        float32_t * pCoeffs,
        float32_t * pState,
        float32_t mu,
        uint32_t blockSize);


  /**
   * @brief Instance structure for the Q15 LMS filter.
   */
  typedef struct
  {
          uint16_t numTaps;    /**< number of coefficients in the filter. */
          q15_t *pState;       /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
          q15_t *pCoeffs;      /**< points to the coefficient array. The array is of length numTaps. */
          q15_t mu;            /**< step size that controls filter coefficient updates. */
          uint32_t postShift;  /**< bit shift applied to coefficients. */
  } arm_lms_instance_q15;


  /**
   * @brief Initialization function for the Q15 LMS filter.
   * @param[in] S          points to an instance of the Q15 LMS filter structure.
   * @param[in] numTaps    number of filter coefficients.
   * @param[in] pCoeffs    points to the coefficient buffer.
   * @param[in] pState     points to the state buffer.
   * @param[in] mu         step size that controls filter coefficient updates.
   * @param[in] blockSize  number of samples to process.
   * @param[in] postShift  bit shift applied to coefficients.
   */
  void arm_lms_init_q15(
        arm_lms_instance_q15 * S,
        uint16_t numTaps,
        q15_t * pCoeffs,
        q15_t * pState,
        q15_t mu,
        uint32_t blockSize,
        uint32_t postShift);


  /**
   * @brief Processing function for Q15 LMS filter.
   * @param[in]  S          points to an instance of the Q15 LMS filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[in]  pRef       points to the block of reference data.
   * @param[out] pOut       points to the block of output data.
   * @param[out] pErr       points to the block of error data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_lms_q15(
  const arm_lms_instance_q15 * S,
  const q15_t * pSrc,
        q15_t * pRef,
        q15_t * pOut,
        q15_t * pErr,
        uint32_t blockSize);


  /**
   * @brief Instance structure for the Q31 LMS filter.
   */
  typedef struct
  {
          uint16_t numTaps;    /**< number of coefficients in the filter. */
          q31_t *pState;       /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
          q31_t *pCoeffs;      /**< points to the coefficient array. The array is of length numTaps. */
          q31_t mu;            /**< step size that controls filter coefficient updates. */
          uint32_t postShift;  /**< bit shift applied to coefficients. */
  } arm_lms_instance_q31;


  /**
   * @brief Processing function for Q31 LMS filter.
   * @param[in]  S          points to an instance of the Q15 LMS filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[in]  pRef       points to the block of reference data.
   * @param[out] pOut       points to the block of output data.
   * @param[out] pErr       points to the block of error data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_lms_q31(
  const arm_lms_instance_q31 * S,
  const q31_t * pSrc,
        q31_t * pRef,
        q31_t * pOut,
        q31_t * pErr,
        uint32_t blockSize);


  /**
   * @brief Initialization function for Q31 LMS filter.
   * @param[in] S          points to an instance of the Q31 LMS filter structure.
   * @param[in] numTaps    number of filter coefficients.
   * @param[in] pCoeffs    points to coefficient buffer.
   * @param[in] pState     points to state buffer.
   * @param[in] mu         step size that controls filter coefficient updates.
   * @param[in] blockSize  number of samples to process.
   * @param[in] postShift  bit shift applied to coefficients.
   */
  void arm_lms_init_q31(
        arm_lms_instance_q31 * S,
        uint16_t numTaps,
        q31_t * pCoeffs,
        q31_t * pState,
        q31_t mu,
        uint32_t blockSize,
        uint32_t postShift);


  /**
   * @brief Instance structure for the floating-point normalized LMS filter.
   */
  typedef struct
  {
          uint16_t numTaps;     /**< number of coefficients in the filter. */
          float32_t *pState;    /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
          float32_t *pCoeffs;   /**< points to the coefficient array. The array is of length numTaps. */
          float32_t mu;         /**< step size that control filter coefficient updates. */
          float32_t energy;     /**< saves previous frame energy. */
          float32_t x0;         /**< saves previous input sample. */
  } arm_lms_norm_instance_f32;


  /**
   * @brief Processing function for floating-point normalized LMS filter.
   * @param[in]  S          points to an instance of the floating-point normalized LMS filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[in]  pRef       points to the block of reference data.
   * @param[out] pOut       points to the block of output data.
   * @param[out] pErr       points to the block of error data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_lms_norm_f32(
        arm_lms_norm_instance_f32 * S,
  const float32_t * pSrc,
        float32_t * pRef,
        float32_t * pOut,
        float32_t * pErr,
        uint32_t blockSize);


  /**
   * @brief Initialization function for floating-point normalized LMS filter.
   * @param[in] S          points to an instance of the floating-point LMS filter structure.
   * @param[in] numTaps    number of filter coefficients.
   * @param[in] pCoeffs    points to coefficient buffer.
   * @param[in] pState     points to state buffer.
   * @param[in] mu         step size that controls filter coefficient updates.
   * @param[in] blockSize  number of samples to process.
   */
  void arm_lms_norm_init_f32(
        arm_lms_norm_instance_f32 * S,
        uint16_t numTaps,
        float32_t * pCoeffs,
        float32_t * pState,
        float32_t mu,
        uint32_t blockSize);


  /**
   * @brief Instance structure for the Q31 normalized LMS filter.
   */
  typedef struct
  {
          uint16_t numTaps;     /**< number of coefficients in the filter. */
          q31_t *pState;        /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
          q31_t *pCoeffs;       /**< points to the coefficient array. The array is of length numTaps. */
          q31_t mu;             /**< step size that controls filter coefficient updates. */
          uint8_t postShift;    /**< bit shift applied to coefficients. */
    const q31_t *recipTable;    /**< points to the reciprocal initial value table. */
          q31_t energy;         /**< saves previous frame energy. */
          q31_t x0;             /**< saves previous input sample. */
  } arm_lms_norm_instance_q31;


  /**
   * @brief Processing function for Q31 normalized LMS filter.
   * @param[in]  S          points to an instance of the Q31 normalized LMS filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[in]  pRef       points to the block of reference data.
   * @param[out] pOut       points to the block of output data.
   * @param[out] pErr       points to the block of error data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_lms_norm_q31(
        arm_lms_norm_instance_q31 * S,
  const q31_t * pSrc,
        q31_t * pRef,
        q31_t * pOut,
        q31_t * pErr,
        uint32_t blockSize);


  /**
   * @brief Initialization function for Q31 normalized LMS filter.
   * @param[in] S          points to an instance of the Q31 normalized LMS filter structure.
   * @param[in] numTaps    number of filter coefficients.
   * @param[in] pCoeffs    points to coefficient buffer.
   * @param[in] pState     points to state buffer.
   * @param[in] mu         step size that controls filter coefficient updates.
   * @param[in] blockSize  number of samples to process.
   * @param[in] postShift  bit shift applied to coefficients.
   */
  void arm_lms_norm_init_q31(
        arm_lms_norm_instance_q31 * S,
        uint16_t numTaps,
        q31_t * pCoeffs,
        q31_t * pState,
        q31_t mu,
        uint32_t blockSize,
        uint8_t postShift);


  /**
   * @brief Instance structure for the Q15 normalized LMS filter.
   */
  typedef struct
  {
          uint16_t numTaps;     /**< Number of coefficients in the filter. */
          q15_t *pState;        /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
          q15_t *pCoeffs;       /**< points to the coefficient array. The array is of length numTaps. */
          q15_t mu;             /**< step size that controls filter coefficient updates. */
          uint8_t postShift;    /**< bit shift applied to coefficients. */
    const q15_t *recipTable;    /**< Points to the reciprocal initial value table. */
          q15_t energy;         /**< saves previous frame energy. */
          q15_t x0;             /**< saves previous input sample. */
  } arm_lms_norm_instance_q15;


  /**
   * @brief Processing function for Q15 normalized LMS filter.
   * @param[in]  S          points to an instance of the Q15 normalized LMS filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[in]  pRef       points to the block of reference data.
   * @param[out] pOut       points to the block of output data.
   * @param[out] pErr       points to the block of error data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_lms_norm_q15(
        arm_lms_norm_instance_q15 * S,
  const q15_t * pSrc,
        q15_t * pRef,
        q15_t * pOut,
        q15_t * pErr,
        uint32_t blockSize);


  /**
   * @brief Initialization function for Q15 normalized LMS filter.
   * @param[in] S          points to an instance of the Q15 normalized LMS filter structure.
   * @param[in] numTaps    number of filter coefficients.
   * @param[in] pCoeffs    points to coefficient buffer.
   * @param[in] pState     points to state buffer.
   * @param[in] mu         step size that controls filter coefficient updates.
   * @param[in] blockSize  number of samples to process.
   * @param[in] postShift  bit shift applied to coefficients.
   */
  void arm_lms_norm_init_q15(
        arm_lms_norm_instance_q15 * S,
        uint16_t numTaps,
        q15_t * pCoeffs,
        q15_t * pState,
        q15_t mu,
        uint32_t blockSize,
        uint8_t postShift);


  /**
   * @brief Correlation of floating-point sequences.
   * @param[in]  pSrcA    points to the first input sequence.
   * @param[in]  srcALen  length of the first input sequence.
   * @param[in]  pSrcB    points to the second input sequence.
   * @param[in]  srcBLen  length of the second input sequence.
   * @param[out] pDst     points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
   */
  void arm_correlate_f32(
  const float32_t * pSrcA,
        uint32_t srcALen,
  const float32_t * pSrcB,
        uint32_t srcBLen,
        float32_t * pDst);


/**
 @brief Correlation of Q15 sequences
 @param[in]  pSrcA     points to the first input sequence
 @param[in]  srcALen   length of the first input sequence
 @param[in]  pSrcB     points to the second input sequence
 @param[in]  srcBLen   length of the second input sequence
 @param[out] pDst      points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
 @param[in]  pScratch  points to scratch buffer of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
*/
void arm_correlate_opt_q15(
  const q15_t * pSrcA,
        uint32_t srcALen,
  const q15_t * pSrcB,
        uint32_t srcBLen,
        q15_t * pDst,
        q15_t * pScratch);


/**
  @brief Correlation of Q15 sequences.
  @param[in]  pSrcA    points to the first input sequence
  @param[in]  srcALen  length of the first input sequence
  @param[in]  pSrcB    points to the second input sequence
  @param[in]  srcBLen  length of the second input sequence
  @param[out] pDst     points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
 */
  void arm_correlate_q15(
  const q15_t * pSrcA,
        uint32_t srcALen,
  const q15_t * pSrcB,
        uint32_t srcBLen,
        q15_t * pDst);


/**
  @brief         Correlation of Q15 sequences (fast version).
  @param[in]     pSrcA      points to the first input sequence
  @param[in]     srcALen    length of the first input sequence
  @param[in]     pSrcB      points to the second input sequence
  @param[in]     srcBLen    length of the second input sequence
  @param[out]    pDst       points to the location where the output result is written.  Length 2 * max(srcALen, srcBLen) - 1.
  @return        none
 */
void arm_correlate_fast_q15(
  const q15_t * pSrcA,
        uint32_t srcALen,
  const q15_t * pSrcB,
        uint32_t srcBLen,
        q15_t * pDst);


/**
  @brief Correlation of Q15 sequences (fast version).
  @param[in]  pSrcA     points to the first input sequence.
  @param[in]  srcALen   length of the first input sequence.
  @param[in]  pSrcB     points to the second input sequence.
  @param[in]  srcBLen   length of the second input sequence.
  @param[out] pDst      points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
  @param[in]  pScratch  points to scratch buffer of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
 */
void arm_correlate_fast_opt_q15(
  const q15_t * pSrcA,
        uint32_t srcALen,
  const q15_t * pSrcB,
        uint32_t srcBLen,
        q15_t * pDst,
        q15_t * pScratch);


  /**
   * @brief Correlation of Q31 sequences.
   * @param[in]  pSrcA    points to the first input sequence.
   * @param[in]  srcALen  length of the first input sequence.
   * @param[in]  pSrcB    points to the second input sequence.
   * @param[in]  srcBLen  length of the second input sequence.
   * @param[out] pDst     points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
   */
  void arm_correlate_q31(
  const q31_t * pSrcA,
        uint32_t srcALen,
  const q31_t * pSrcB,
        uint32_t srcBLen,
        q31_t * pDst);


/**
  @brief Correlation of Q31 sequences (fast version).
  @param[in]  pSrcA    points to the first input sequence
  @param[in]  srcALen  length of the first input sequence
  @param[in]  pSrcB    points to the second input sequence
  @param[in]  srcBLen  length of the second input sequence
  @param[out] pDst     points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
 */
void arm_correlate_fast_q31(
  const q31_t * pSrcA,
        uint32_t srcALen,
  const q31_t * pSrcB,
        uint32_t srcBLen,
        q31_t * pDst);


 /**
   * @brief Correlation of Q7 sequences.
   * @param[in]  pSrcA      points to the first input sequence.
   * @param[in]  srcALen    length of the first input sequence.
   * @param[in]  pSrcB      points to the second input sequence.
   * @param[in]  srcBLen    length of the second input sequence.
   * @param[out] pDst       points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
   * @param[in]  pScratch1  points to scratch buffer(of type q15_t) of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
   * @param[in]  pScratch2  points to scratch buffer (of type q15_t) of size min(srcALen, srcBLen).
   */
  void arm_correlate_opt_q7(
  const q7_t * pSrcA,
        uint32_t srcALen,
  const q7_t * pSrcB,
        uint32_t srcBLen,
        q7_t * pDst,
        q15_t * pScratch1,
        q15_t * pScratch2);


  /**
   * @brief Correlation of Q7 sequences.
   * @param[in]  pSrcA    points to the first input sequence.
   * @param[in]  srcALen  length of the first input sequence.
   * @param[in]  pSrcB    points to the second input sequence.
   * @param[in]  srcBLen  length of the second input sequence.
   * @param[out] pDst     points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
   */
  void arm_correlate_q7(
  const q7_t * pSrcA,
        uint32_t srcALen,
  const q7_t * pSrcB,
        uint32_t srcBLen,
        q7_t * pDst);


  /**
   * @brief Instance structure for the floating-point sparse FIR filter.
   */
  typedef struct
  {
          uint16_t numTaps;             /**< number of coefficients in the filter. */
          uint16_t stateIndex;          /**< state buffer index.  Points to the oldest sample in the state buffer. */
          float32_t *pState;            /**< points to the state buffer array. The array is of length maxDelay+blockSize-1. */
    const float32_t *pCoeffs;           /**< points to the coefficient array. The array is of length numTaps.*/
          uint16_t maxDelay;            /**< maximum offset specified by the pTapDelay array. */
          int32_t *pTapDelay;           /**< points to the array of delay values.  The array is of length numTaps. */
  } arm_fir_sparse_instance_f32;

  /**
   * @brief Instance structure for the Q31 sparse FIR filter.
   */
  typedef struct
  {
          uint16_t numTaps;             /**< number of coefficients in the filter. */
          uint16_t stateIndex;          /**< state buffer index.  Points to the oldest sample in the state buffer. */
          q31_t *pState;                /**< points to the state buffer array. The array is of length maxDelay+blockSize-1. */
    const q31_t *pCoeffs;               /**< points to the coefficient array. The array is of length numTaps.*/
          uint16_t maxDelay;            /**< maximum offset specified by the pTapDelay array. */
          int32_t *pTapDelay;           /**< points to the array of delay values.  The array is of length numTaps. */
  } arm_fir_sparse_instance_q31;

  /**
   * @brief Instance structure for the Q15 sparse FIR filter.
   */
  typedef struct
  {
          uint16_t numTaps;             /**< number of coefficients in the filter. */
          uint16_t stateIndex;          /**< state buffer index.  Points to the oldest sample in the state buffer. */
          q15_t *pState;                /**< points to the state buffer array. The array is of length maxDelay+blockSize-1. */
    const q15_t *pCoeffs;               /**< points to the coefficient array. The array is of length numTaps.*/
          uint16_t maxDelay;            /**< maximum offset specified by the pTapDelay array. */
          int32_t *pTapDelay;           /**< points to the array of delay values.  The array is of length numTaps. */
  } arm_fir_sparse_instance_q15;

  /**
   * @brief Instance structure for the Q7 sparse FIR filter.
   */
  typedef struct
  {
          uint16_t numTaps;             /**< number of coefficients in the filter. */
          uint16_t stateIndex;          /**< state buffer index.  Points to the oldest sample in the state buffer. */
          q7_t *pState;                 /**< points to the state buffer array. The array is of length maxDelay+blockSize-1. */
    const q7_t *pCoeffs;                /**< points to the coefficient array. The array is of length numTaps.*/
          uint16_t maxDelay;            /**< maximum offset specified by the pTapDelay array. */
          int32_t *pTapDelay;           /**< points to the array of delay values.  The array is of length numTaps. */
  } arm_fir_sparse_instance_q7;


  /**
   * @brief Processing function for the floating-point sparse FIR filter.
   * @param[in]  S           points to an instance of the floating-point sparse FIR structure.
   * @param[in]  pSrc        points to the block of input data.
   * @param[out] pDst        points to the block of output data
   * @param[in]  pScratchIn  points to a temporary buffer of size blockSize.
   * @param[in]  blockSize   number of input samples to process per call.
   */
  void arm_fir_sparse_f32(
        arm_fir_sparse_instance_f32 * S,
  const float32_t * pSrc,
        float32_t * pDst,
        float32_t * pScratchIn,
        uint32_t blockSize);


  /**
   * @brief  Initialization function for the floating-point sparse FIR filter.
   * @param[in,out] S          points to an instance of the floating-point sparse FIR structure.
   * @param[in]     numTaps    number of nonzero coefficients in the filter.
   * @param[in]     pCoeffs    points to the array of filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     pTapDelay  points to the array of offset times.
   * @param[in]     maxDelay   maximum offset time supported.
   * @param[in]     blockSize  number of samples that will be processed per block.
   */
  void arm_fir_sparse_init_f32(
        arm_fir_sparse_instance_f32 * S,
        uint16_t numTaps,
  const float32_t * pCoeffs,
        float32_t * pState,
        int32_t * pTapDelay,
        uint16_t maxDelay,
        uint32_t blockSize);


  /**
   * @brief Processing function for the Q31 sparse FIR filter.
   * @param[in]  S           points to an instance of the Q31 sparse FIR structure.
   * @param[in]  pSrc        points to the block of input data.
   * @param[out] pDst        points to the block of output data
   * @param[in]  pScratchIn  points to a temporary buffer of size blockSize.
   * @param[in]  blockSize   number of input samples to process per call.
   */
  void arm_fir_sparse_q31(
        arm_fir_sparse_instance_q31 * S,
  const q31_t * pSrc,
        q31_t * pDst,
        q31_t * pScratchIn,
        uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q31 sparse FIR filter.
   * @param[in,out] S          points to an instance of the Q31 sparse FIR structure.
   * @param[in]     numTaps    number of nonzero coefficients in the filter.
   * @param[in]     pCoeffs    points to the array of filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     pTapDelay  points to the array of offset times.
   * @param[in]     maxDelay   maximum offset time supported.
   * @param[in]     blockSize  number of samples that will be processed per block.
   */
  void arm_fir_sparse_init_q31(
        arm_fir_sparse_instance_q31 * S,
        uint16_t numTaps,
  const q31_t * pCoeffs,
        q31_t * pState,
        int32_t * pTapDelay,
        uint16_t maxDelay,
        uint32_t blockSize);


  /**
   * @brief Processing function for the Q15 sparse FIR filter.
   * @param[in]  S            points to an instance of the Q15 sparse FIR structure.
   * @param[in]  pSrc         points to the block of input data.
   * @param[out] pDst         points to the block of output data
   * @param[in]  pScratchIn   points to a temporary buffer of size blockSize.
   * @param[in]  pScratchOut  points to a temporary buffer of size blockSize.
   * @param[in]  blockSize    number of input samples to process per call.
   */
  void arm_fir_sparse_q15(
        arm_fir_sparse_instance_q15 * S,
  const q15_t * pSrc,
        q15_t * pDst,
        q15_t * pScratchIn,
        q31_t * pScratchOut,
        uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q15 sparse FIR filter.
   * @param[in,out] S          points to an instance of the Q15 sparse FIR structure.
   * @param[in]     numTaps    number of nonzero coefficients in the filter.
   * @param[in]     pCoeffs    points to the array of filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     pTapDelay  points to the array of offset times.
   * @param[in]     maxDelay   maximum offset time supported.
   * @param[in]     blockSize  number of samples that will be processed per block.
   */
  void arm_fir_sparse_init_q15(
        arm_fir_sparse_instance_q15 * S,
        uint16_t numTaps,
  const q15_t * pCoeffs,
        q15_t * pState,
        int32_t * pTapDelay,
        uint16_t maxDelay,
        uint32_t blockSize);


  /**
   * @brief Processing function for the Q7 sparse FIR filter.
   * @param[in]  S            points to an instance of the Q7 sparse FIR structure.
   * @param[in]  pSrc         points to the block of input data.
   * @param[out] pDst         points to the block of output data
   * @param[in]  pScratchIn   points to a temporary buffer of size blockSize.
   * @param[in]  pScratchOut  points to a temporary buffer of size blockSize.
   * @param[in]  blockSize    number of input samples to process per call.
   */
  void arm_fir_sparse_q7(
        arm_fir_sparse_instance_q7 * S,
  const q7_t * pSrc,
        q7_t * pDst,
        q7_t * pScratchIn,
        q31_t * pScratchOut,
        uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q7 sparse FIR filter.
   * @param[in,out] S          points to an instance of the Q7 sparse FIR structure.
   * @param[in]     numTaps    number of nonzero coefficients in the filter.
   * @param[in]     pCoeffs    points to the array of filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     pTapDelay  points to the array of offset times.
   * @param[in]     maxDelay   maximum offset time supported.
   * @param[in]     blockSize  number of samples that will be processed per block.
   */
  void arm_fir_sparse_init_q7(
        arm_fir_sparse_instance_q7 * S,
        uint16_t numTaps,
  const q7_t * pCoeffs,
        q7_t * pState,
        int32_t * pTapDelay,
        uint16_t maxDelay,
        uint32_t blockSize);


  /**
   * @brief  Floating-point sin_cos function.
   * @param[in]  theta   input value in degrees
   * @param[out] pSinVal  points to the processed sine output.
   * @param[out] pCosVal  points to the processed cos output.
   */
  void arm_sin_cos_f32(
        float32_t theta,
        float32_t * pSinVal,
        float32_t * pCosVal);


  /**
   * @brief  Q31 sin_cos function.
   * @param[in]  theta    scaled input value in degrees
   * @param[out] pSinVal  points to the processed sine output.
   * @param[out] pCosVal  points to the processed cosine output.
   */
  void arm_sin_cos_q31(
        q31_t theta,
        q31_t * pSinVal,
        q31_t * pCosVal);


  /**
   * @brief  Floating-point complex conjugate.
   * @param[in]  pSrc        points to the input vector
   * @param[out] pDst        points to the output vector
   * @param[in]  numSamples  number of complex samples in each vector
   */
  void arm_cmplx_conj_f32(
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t numSamples);

  /**
   * @brief  Q31 complex conjugate.
   * @param[in]  pSrc        points to the input vector
   * @param[out] pDst        points to the output vector
   * @param[in]  numSamples  number of complex samples in each vector
   */
  void arm_cmplx_conj_q31(
  const q31_t * pSrc,
        q31_t * pDst,
        uint32_t numSamples);


  /**
   * @brief  Q15 complex conjugate.
   * @param[in]  pSrc        points to the input vector
   * @param[out] pDst        points to the output vector
   * @param[in]  numSamples  number of complex samples in each vector
   */
  void arm_cmplx_conj_q15(
  const q15_t * pSrc,
        q15_t * pDst,
        uint32_t numSamples);


  /**
   * @brief  Floating-point complex magnitude squared
   * @param[in]  pSrc        points to the complex input vector
   * @param[out] pDst        points to the real output vector
   * @param[in]  numSamples  number of complex samples in the input vector
   */
  void arm_cmplx_mag_squared_f32(
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t numSamples);


  /**
   * @brief  Q31 complex magnitude squared
   * @param[in]  pSrc        points to the complex input vector
   * @param[out] pDst        points to the real output vector
   * @param[in]  numSamples  number of complex samples in the input vector
   */
  void arm_cmplx_mag_squared_q31(
  const q31_t * pSrc,
        q31_t * pDst,
        uint32_t numSamples);


  /**
   * @brief  Q15 complex magnitude squared
   * @param[in]  pSrc        points to the complex input vector
   * @param[out] pDst        points to the real output vector
   * @param[in]  numSamples  number of complex samples in the input vector
   */
  void arm_cmplx_mag_squared_q15(
  const q15_t * pSrc,
        q15_t * pDst,
        uint32_t numSamples);


 /**
   * @ingroup groupController
   */

  /**
   * @defgroup PID PID Motor Control
   *
   * A Proportional Integral Derivative (PID) controller is a generic feedback control
   * loop mechanism widely used in industrial control systems.
   * A PID controller is the most commonly used type of feedback controller.
   *
   * This set of functions implements (PID) controllers
   * for Q15, Q31, and floating-point data types.  The functions operate on a single sample
   * of data and each call to the function returns a single processed value.
   * <code>S</code> points to an instance of the PID control data structure.  <code>in</code>
   * is the input sample value. The functions return the output value.
   *
   * \par Algorithm:
   * <pre>
   *    y[n] = y[n-1] + A0 * x[n] + A1 * x[n-1] + A2 * x[n-2]
   *    A0 = Kp + Ki + Kd
   *    A1 = (-Kp ) - (2 * Kd )
   *    A2 = Kd
   * </pre>
   *
   * \par
   * where \c Kp is proportional constant, \c Ki is Integral constant and \c Kd is Derivative constant
   *
   * \par
   * \image html PID.gif "Proportional Integral Derivative Controller"
   *
   * \par
   * The PID controller calculates an "error" value as the difference between
   * the measured output and the reference input.
   * The controller attempts to minimize the error by adjusting the process control inputs.
   * The proportional value determines the reaction to the current error,
   * the integral value determines the reaction based on the sum of recent errors,
   * and the derivative value determines the reaction based on the rate at which the error has been changing.
   *
   * \par Instance Structure
   * The Gains A0, A1, A2 and state variables for a PID controller are stored together in an instance data structure.
   * A separate instance structure must be defined for each PID Controller.
   * There are separate instance structure declarations for each of the 3 supported data types.
   *
   * \par Reset Functions
   * There is also an associated reset function for each data type which clears the state array.
   *
   * \par Initialization Functions
   * There is also an associated initialization function for each data type.
   * The initialization function performs the following operations:
   * - Initializes the Gains A0, A1, A2 from Kp,Ki, Kd gains.
   * - Zeros out the values in the state buffer.
   *
   * \par
   * Instance structure cannot be placed into a const data section and it is recommended to use the initialization function.
   *
   * \par Fixed-Point Behavior
   * Care must be taken when using the fixed-point versions of the PID Controller functions.
   * In particular, the overflow and saturation behavior of the accumulator used in each function must be considered.
   * Refer to the function specific documentation below for usage guidelines.
   */

  /**
   * @addtogroup PID
   * @{
   */

  /**
   * @brief         Process function for the floating-point PID Control.
   * @param[in,out] S   is an instance of the floating-point PID Control structure
   * @param[in]     in  input sample to process
   * @return        processed output sample.
   */
  __STATIC_FORCEINLINE float32_t arm_pid_f32(
  arm_pid_instance_f32 * S,
  float32_t in)
  {
    float32_t out;

    /* y[n] = y[n-1] + A0 * x[n] + A1 * x[n-1] + A2 * x[n-2]  */
    out = (S->A0 * in) +
      (S->A1 * S->state[0]) + (S->A2 * S->state[1]) + (S->state[2]);

    /* Update state */
    S->state[1] = S->state[0];
    S->state[0] = in;
    S->state[2] = out;

    /* return to application */
    return (out);

  }

/**
  @brief         Process function for the Q31 PID Control.
  @param[in,out] S  points to an instance of the Q31 PID Control structure
  @param[in]     in  input sample to process
  @return        processed output sample.

  \par Scaling and Overflow Behavior
         The function is implemented using an internal 64-bit accumulator.
         The accumulator has a 2.62 format and maintains full precision of the intermediate multiplication results but provides only a single guard bit.
         Thus, if the accumulator result overflows it wraps around rather than clip.
         In order to avoid overflows completely the input signal must be scaled down by 2 bits as there are four additions.
         After all multiply-accumulates are performed, the 2.62 accumulator is truncated to 1.32 format and then saturated to 1.31 format.
 */
__STATIC_FORCEINLINE q31_t arm_pid_q31(
  arm_pid_instance_q31 * S,
  q31_t in)
  {
    q63_t acc;
    q31_t out;

    /* acc = A0 * x[n]  */
    acc = (q63_t) S->A0 * in;

    /* acc += A1 * x[n-1] */
    acc += (q63_t) S->A1 * S->state[0];

    /* acc += A2 * x[n-2]  */
    acc += (q63_t) S->A2 * S->state[1];

    /* convert output to 1.31 format to add y[n-1] */
    out = (q31_t) (acc >> 31U);

    /* out += y[n-1] */
    out += S->state[2];

    /* Update state */
    S->state[1] = S->state[0];
    S->state[0] = in;
    S->state[2] = out;

    /* return to application */
    return (out);
  }


/**
  @brief         Process function for the Q15 PID Control.
  @param[in,out] S   points to an instance of the Q15 PID Control structure
  @param[in]     in  input sample to process
  @return        processed output sample.

  \par Scaling and Overflow Behavior
         The function is implemented using a 64-bit internal accumulator.
         Both Gains and state variables are represented in 1.15 format and multiplications yield a 2.30 result.
         The 2.30 intermediate results are accumulated in a 64-bit accumulator in 34.30 format.
         There is no risk of internal overflow with this approach and the full precision of intermediate multiplications is preserved.
         After all additions have been performed, the accumulator is truncated to 34.15 format by discarding low 15 bits.
         Lastly, the accumulator is saturated to yield a result in 1.15 format.
 */
__STATIC_FORCEINLINE q15_t arm_pid_q15(
  arm_pid_instance_q15 * S,
  q15_t in)
  {
    q63_t acc;
    q15_t out;

#if defined (ARM_MATH_DSP)
    /* Implementation of PID controller */

    /* acc = A0 * x[n]  */
    acc = (q31_t) __SMUAD((uint32_t)S->A0, (uint32_t)in);

    /* acc += A1 * x[n-1] + A2 * x[n-2]  */
    acc = (q63_t)__SMLALD((uint32_t)S->A1, (uint32_t)read_q15x2 (S->state), (uint64_t)acc);
#else
    /* acc = A0 * x[n]  */
    acc = ((q31_t) S->A0) * in;

    /* acc += A1 * x[n-1] + A2 * x[n-2]  */
    acc += (q31_t) S->A1 * S->state[0];
    acc += (q31_t) S->A2 * S->state[1];
#endif

    /* acc += y[n-1] */
    acc += (q31_t) S->state[2] << 15;

    /* saturate the output */
    out = (q15_t) (__SSAT((q31_t)(acc >> 15), 16));

    /* Update state */
    S->state[1] = S->state[0];
    S->state[0] = in;
    S->state[2] = out;

    /* return to application */
    return (out);
  }

  /**
   * @} end of PID group
   */


  /**
   * @brief Floating-point matrix inverse.
   * @param[in]  src   points to the instance of the input floating-point matrix structure.
   * @param[out] dst   points to the instance of the output floating-point matrix structure.
   * @return The function returns ARM_MATH_SIZE_MISMATCH, if the dimensions do not match.
   * If the input matrix is singular (does not have an inverse), then the algorithm terminates and returns error status ARM_MATH_SINGULAR.
   */
  arm_status arm_mat_inverse_f32(
  const arm_matrix_instance_f32 * src,
  arm_matrix_instance_f32 * dst);


  /**
   * @brief Floating-point matrix inverse.
   * @param[in]  src   points to the instance of the input floating-point matrix structure.
   * @param[out] dst   points to the instance of the output floating-point matrix structure.
   * @return The function returns ARM_MATH_SIZE_MISMATCH, if the dimensions do not match.
   * If the input matrix is singular (does not have an inverse), then the algorithm terminates and returns error status ARM_MATH_SINGULAR.
   */
  arm_status arm_mat_inverse_f64(
  const arm_matrix_instance_f64 * src,
  arm_matrix_instance_f64 * dst);



  /**
   * @ingroup groupController
   */

  /**
   * @defgroup clarke Vector Clarke Transform
   * Forward Clarke transform converts the instantaneous stator phases into a two-coordinate time invariant vector.
   * Generally the Clarke transform uses three-phase currents <code>Ia, Ib and Ic</code> to calculate currents
   * in the two-phase orthogonal stator axis <code>Ialpha</code> and <code>Ibeta</code>.
   * When <code>Ialpha</code> is superposed with <code>Ia</code> as shown in the figure below
   * \image html clarke.gif Stator current space vector and its components in (a,b).
   * and <code>Ia + Ib + Ic = 0</code>, in this condition <code>Ialpha</code> and <code>Ibeta</code>
   * can be calculated using only <code>Ia</code> and <code>Ib</code>.
   *
   * The function operates on a single sample of data and each call to the function returns the processed output.
   * The library provides separate functions for Q31 and floating-point data types.
   * \par Algorithm
   * \image html clarkeFormula.gif
   * where <code>Ia</code> and <code>Ib</code> are the instantaneous stator phases and
   * <code>pIalpha</code> and <code>pIbeta</code> are the two coordinates of time invariant vector.
   * \par Fixed-Point Behavior
   * Care must be taken when using the Q31 version of the Clarke transform.
   * In particular, the overflow and saturation behavior of the accumulator used must be considered.
   * Refer to the function specific documentation below for usage guidelines.
   */

  /**
   * @addtogroup clarke
   * @{
   */

  /**
   *
   * @brief  Floating-point Clarke transform
   * @param[in]  Ia       input three-phase coordinate <code>a</code>
   * @param[in]  Ib       input three-phase coordinate <code>b</code>
   * @param[out] pIalpha  points to output two-phase orthogonal vector axis alpha
   * @param[out] pIbeta   points to output two-phase orthogonal vector axis beta
   * @return        none
   */
  __STATIC_FORCEINLINE void arm_clarke_f32(
  float32_t Ia,
  float32_t Ib,
  float32_t * pIalpha,
  float32_t * pIbeta)
  {
    /* Calculate pIalpha using the equation, pIalpha = Ia */
    *pIalpha = Ia;

    /* Calculate pIbeta using the equation, pIbeta = (1/sqrt(3)) * Ia + (2/sqrt(3)) * Ib */
    *pIbeta = ((float32_t) 0.57735026919 * Ia + (float32_t) 1.15470053838 * Ib);
  }


/**
  @brief  Clarke transform for Q31 version
  @param[in]  Ia       input three-phase coordinate <code>a</code>
  @param[in]  Ib       input three-phase coordinate <code>b</code>
  @param[out] pIalpha  points to output two-phase orthogonal vector axis alpha
  @param[out] pIbeta   points to output two-phase orthogonal vector axis beta
  @return     none

  \par Scaling and Overflow Behavior
         The function is implemented using an internal 32-bit accumulator.
         The accumulator maintains 1.31 format by truncating lower 31 bits of the intermediate multiplication in 2.62 format.
         There is saturation on the addition, hence there is no risk of overflow.
 */
__STATIC_FORCEINLINE void arm_clarke_q31(
  q31_t Ia,
  q31_t Ib,
  q31_t * pIalpha,
  q31_t * pIbeta)
  {
    q31_t product1, product2;                    /* Temporary variables used to store intermediate results */

    /* Calculating pIalpha from Ia by equation pIalpha = Ia */
    *pIalpha = Ia;

    /* Intermediate product is calculated by (1/(sqrt(3)) * Ia) */
    product1 = (q31_t) (((q63_t) Ia * 0x24F34E8B) >> 30);

    /* Intermediate product is calculated by (2/sqrt(3) * Ib) */
    product2 = (q31_t) (((q63_t) Ib * 0x49E69D16) >> 30);

    /* pIbeta is calculated by adding the intermediate products */
    *pIbeta = __QADD(product1, product2);
  }

  /**
   * @} end of clarke group
   */


  /**
   * @ingroup groupController
   */

  /**
   * @defgroup inv_clarke Vector Inverse Clarke Transform
   * Inverse Clarke transform converts the two-coordinate time invariant vector into instantaneous stator phases.
   *
   * The function operates on a single sample of data and each call to the function returns the processed output.
   * The library provides separate functions for Q31 and floating-point data types.
   * \par Algorithm
   * \image html clarkeInvFormula.gif
   * where <code>pIa</code> and <code>pIb</code> are the instantaneous stator phases and
   * <code>Ialpha</code> and <code>Ibeta</code> are the two coordinates of time invariant vector.
   * \par Fixed-Point Behavior
   * Care must be taken when using the Q31 version of the Clarke transform.
   * In particular, the overflow and saturation behavior of the accumulator used must be considered.
   * Refer to the function specific documentation below for usage guidelines.
   */

  /**
   * @addtogroup inv_clarke
   * @{
   */

   /**
   * @brief  Floating-point Inverse Clarke transform
   * @param[in]  Ialpha  input two-phase orthogonal vector axis alpha
   * @param[in]  Ibeta   input two-phase orthogonal vector axis beta
   * @param[out] pIa     points to output three-phase coordinate <code>a</code>
   * @param[out] pIb     points to output three-phase coordinate <code>b</code>
   * @return     none
   */
  __STATIC_FORCEINLINE void arm_inv_clarke_f32(
  float32_t Ialpha,
  float32_t Ibeta,
  float32_t * pIa,
  float32_t * pIb)
  {
    /* Calculating pIa from Ialpha by equation pIa = Ialpha */
    *pIa = Ialpha;

    /* Calculating pIb from Ialpha and Ibeta by equation pIb = -(1/2) * Ialpha + (sqrt(3)/2) * Ibeta */
    *pIb = -0.5f * Ialpha + 0.8660254039f * Ibeta;
  }


/**
  @brief  Inverse Clarke transform for Q31 version
  @param[in]  Ialpha  input two-phase orthogonal vector axis alpha
  @param[in]  Ibeta   input two-phase orthogonal vector axis beta
  @param[out] pIa     points to output three-phase coordinate <code>a</code>
  @param[out] pIb     points to output three-phase coordinate <code>b</code>
  @return     none

  \par Scaling and Overflow Behavior
         The function is implemented using an internal 32-bit accumulator.
         The accumulator maintains 1.31 format by truncating lower 31 bits of the intermediate multiplication in 2.62 format.
         There is saturation on the subtraction, hence there is no risk of overflow.
 */
__STATIC_FORCEINLINE void arm_inv_clarke_q31(
  q31_t Ialpha,
  q31_t Ibeta,
  q31_t * pIa,
  q31_t * pIb)
  {
    q31_t product1, product2;                    /* Temporary variables used to store intermediate results */

    /* Calculating pIa from Ialpha by equation pIa = Ialpha */
    *pIa = Ialpha;

    /* Intermediate product is calculated by (1/(2*sqrt(3)) * Ia) */
    product1 = (q31_t) (((q63_t) (Ialpha) * (0x40000000)) >> 31);

    /* Intermediate product is calculated by (1/sqrt(3) * pIb) */
    product2 = (q31_t) (((q63_t) (Ibeta) * (0x6ED9EBA1)) >> 31);

    /* pIb is calculated by subtracting the products */
    *pIb = __QSUB(product2, product1);
  }

  /**
   * @} end of inv_clarke group
   */



  /**
   * @ingroup groupController
   */

  /**
   * @defgroup park Vector Park Transform
   *
   * Forward Park transform converts the input two-coordinate vector to flux and torque components.
   * The Park transform can be used to realize the transformation of the <code>Ialpha</code> and the <code>Ibeta</code> currents
   * from the stationary to the moving reference frame and control the spatial relationship between
   * the stator vector current and rotor flux vector.
   * If we consider the d axis aligned with the rotor flux, the diagram below shows the
   * current vector and the relationship from the two reference frames:
   * \image html park.gif "Stator current space vector and its component in (a,b) and in the d,q rotating reference frame"
   *
   * The function operates on a single sample of data and each call to the function returns the processed output.
   * The library provides separate functions for Q31 and floating-point data types.
   * \par Algorithm
   * \image html parkFormula.gif
   * where <code>Ialpha</code> and <code>Ibeta</code> are the stator vector components,
   * <code>pId</code> and <code>pIq</code> are rotor vector components and <code>cosVal</code> and <code>sinVal</code> are the
   * cosine and sine values of theta (rotor flux position).
   * \par Fixed-Point Behavior
   * Care must be taken when using the Q31 version of the Park transform.
   * In particular, the overflow and saturation behavior of the accumulator used must be considered.
   * Refer to the function specific documentation below for usage guidelines.
   */

  /**
   * @addtogroup park
   * @{
   */

  /**
   * @brief Floating-point Park transform
   * @param[in]  Ialpha  input two-phase vector coordinate alpha
   * @param[in]  Ibeta   input two-phase vector coordinate beta
   * @param[out] pId     points to output   rotor reference frame d
   * @param[out] pIq     points to output   rotor reference frame q
   * @param[in]  sinVal  sine value of rotation angle theta
   * @param[in]  cosVal  cosine value of rotation angle theta
   * @return     none
   *
   * The function implements the forward Park transform.
   *
   */
  __STATIC_FORCEINLINE void arm_park_f32(
  float32_t Ialpha,
  float32_t Ibeta,
  float32_t * pId,
  float32_t * pIq,
  float32_t sinVal,
  float32_t cosVal)
  {
    /* Calculate pId using the equation, pId = Ialpha * cosVal + Ibeta * sinVal */
    *pId = Ialpha * cosVal + Ibeta * sinVal;

    /* Calculate pIq using the equation, pIq = - Ialpha * sinVal + Ibeta * cosVal */
    *pIq = -Ialpha * sinVal + Ibeta * cosVal;
  }


/**
  @brief  Park transform for Q31 version
  @param[in]  Ialpha  input two-phase vector coordinate alpha
  @param[in]  Ibeta   input two-phase vector coordinate beta
  @param[out] pId     points to output rotor reference frame d
  @param[out] pIq     points to output rotor reference frame q
  @param[in]  sinVal  sine value of rotation angle theta
  @param[in]  cosVal  cosine value of rotation angle theta
  @return     none

  \par Scaling and Overflow Behavior
         The function is implemented using an internal 32-bit accumulator.
         The accumulator maintains 1.31 format by truncating lower 31 bits of the intermediate multiplication in 2.62 format.
         There is saturation on the addition and subtraction, hence there is no risk of overflow.
 */
__STATIC_FORCEINLINE void arm_park_q31(
  q31_t Ialpha,
  q31_t Ibeta,
  q31_t * pId,
  q31_t * pIq,
  q31_t sinVal,
  q31_t cosVal)
  {
    q31_t product1, product2;                    /* Temporary variables used to store intermediate results */
    q31_t product3, product4;                    /* Temporary variables used to store intermediate results */

    /* Intermediate product is calculated by (Ialpha * cosVal) */
    product1 = (q31_t) (((q63_t) (Ialpha) * (cosVal)) >> 31);

    /* Intermediate product is calculated by (Ibeta * sinVal) */
    product2 = (q31_t) (((q63_t) (Ibeta) * (sinVal)) >> 31);


    /* Intermediate product is calculated by (Ialpha * sinVal) */
    product3 = (q31_t) (((q63_t) (Ialpha) * (sinVal)) >> 31);

    /* Intermediate product is calculated by (Ibeta * cosVal) */
    product4 = (q31_t) (((q63_t) (Ibeta) * (cosVal)) >> 31);

    /* Calculate pId by adding the two intermediate products 1 and 2 */
    *pId = __QADD(product1, product2);

    /* Calculate pIq by subtracting the two intermediate products 3 from 4 */
    *pIq = __QSUB(product4, product3);
  }

  /**
   * @} end of park group
   */


  /**
   * @ingroup groupController
   */

  /**
   * @defgroup inv_park Vector Inverse Park transform
   * Inverse Park transform converts the input flux and torque components to two-coordinate vector.
   *
   * The function operates on a single sample of data and each call to the function returns the processed output.
   * The library provides separate functions for Q31 and floating-point data types.
   * \par Algorithm
   * \image html parkInvFormula.gif
   * where <code>pIalpha</code> and <code>pIbeta</code> are the stator vector components,
   * <code>Id</code> and <code>Iq</code> are rotor vector components and <code>cosVal</code> and <code>sinVal</code> are the
   * cosine and sine values of theta (rotor flux position).
   * \par Fixed-Point Behavior
   * Care must be taken when using the Q31 version of the Park transform.
   * In particular, the overflow and saturation behavior of the accumulator used must be considered.
   * Refer to the function specific documentation below for usage guidelines.
   */

  /**
   * @addtogroup inv_park
   * @{
   */

   /**
   * @brief  Floating-point Inverse Park transform
   * @param[in]  Id       input coordinate of rotor reference frame d
   * @param[in]  Iq       input coordinate of rotor reference frame q
   * @param[out] pIalpha  points to output two-phase orthogonal vector axis alpha
   * @param[out] pIbeta   points to output two-phase orthogonal vector axis beta
   * @param[in]  sinVal   sine value of rotation angle theta
   * @param[in]  cosVal   cosine value of rotation angle theta
   * @return     none
   */
  __STATIC_FORCEINLINE void arm_inv_park_f32(
  float32_t Id,
  float32_t Iq,
  float32_t * pIalpha,
  float32_t * pIbeta,
  float32_t sinVal,
  float32_t cosVal)
  {
    /* Calculate pIalpha using the equation, pIalpha = Id * cosVal - Iq * sinVal */
    *pIalpha = Id * cosVal - Iq * sinVal;

    /* Calculate pIbeta using the equation, pIbeta = Id * sinVal + Iq * cosVal */
    *pIbeta = Id * sinVal + Iq * cosVal;
  }


/**
  @brief  Inverse Park transform for   Q31 version
  @param[in]  Id       input coordinate of rotor reference frame d
  @param[in]  Iq       input coordinate of rotor reference frame q
  @param[out] pIalpha  points to output two-phase orthogonal vector axis alpha
  @param[out] pIbeta   points to output two-phase orthogonal vector axis beta
  @param[in]  sinVal   sine value of rotation angle theta
  @param[in]  cosVal   cosine value of rotation angle theta
  @return     none

  @par Scaling and Overflow Behavior
         The function is implemented using an internal 32-bit accumulator.
         The accumulator maintains 1.31 format by truncating lower 31 bits of the intermediate multiplication in 2.62 format.
         There is saturation on the addition, hence there is no risk of overflow.
 */
__STATIC_FORCEINLINE void arm_inv_park_q31(
  q31_t Id,
  q31_t Iq,
  q31_t * pIalpha,
  q31_t * pIbeta,
  q31_t sinVal,
  q31_t cosVal)
  {
    q31_t product1, product2;                    /* Temporary variables used to store intermediate results */
    q31_t product3, product4;                    /* Temporary variables used to store intermediate results */

    /* Intermediate product is calculated by (Id * cosVal) */
    product1 = (q31_t) (((q63_t) (Id) * (cosVal)) >> 31);

    /* Intermediate product is calculated by (Iq * sinVal) */
    product2 = (q31_t) (((q63_t) (Iq) * (sinVal)) >> 31);


    /* Intermediate product is calculated by (Id * sinVal) */
    product3 = (q31_t) (((q63_t) (Id) * (sinVal)) >> 31);

    /* Intermediate product is calculated by (Iq * cosVal) */
    product4 = (q31_t) (((q63_t) (Iq) * (cosVal)) >> 31);

    /* Calculate pIalpha by using the two intermediate products 1 and 2 */
    *pIalpha = __QSUB(product1, product2);

    /* Calculate pIbeta by using the two intermediate products 3 and 4 */
    *pIbeta = __QADD(product4, product3);
  }

  /**
   * @} end of Inverse park group
   */


  /**
   * @ingroup groupInterpolation
   */

  /**
   * @defgroup LinearInterpolate Linear Interpolation
   *
   * Linear interpolation is a method of curve fitting using linear polynomials.
   * Linear interpolation works by effectively drawing a straight line between two neighboring samples and returning the appropriate point along that line
   *
   * \par
   * \image html LinearInterp.gif "Linear interpolation"
   *
   * \par
   * A  Linear Interpolate function calculates an output value(y), for the input(x)
   * using linear interpolation of the input values x0, x1( nearest input values) and the output values y0 and y1(nearest output values)
   *
   * \par Algorithm:
   * <pre>
   *       y = y0 + (x - x0) * ((y1 - y0)/(x1-x0))
   *       where x0, x1 are nearest values of input x
   *             y0, y1 are nearest values to output y
   * </pre>
   *
   * \par
   * This set of functions implements Linear interpolation process
   * for Q7, Q15, Q31, and floating-point data types.  The functions operate on a single
   * sample of data and each call to the function returns a single processed value.
   * <code>S</code> points to an instance of the Linear Interpolate function data structure.
   * <code>x</code> is the input sample value. The functions returns the output value.
   *
   * \par
   * if x is outside of the table boundary, Linear interpolation returns first value of the table
   * if x is below input range and returns last value of table if x is above range.
   */

  /**
   * @addtogroup LinearInterpolate
   * @{
   */

  /**
   * @brief  Process function for the floating-point Linear Interpolation Function.
   * @param[in,out] S  is an instance of the floating-point Linear Interpolation structure
   * @param[in]     x  input sample to process
   * @return y processed output sample.
   *
   */
  __STATIC_FORCEINLINE float32_t arm_linear_interp_f32(
  arm_linear_interp_instance_f32 * S,
  float32_t x)
  {
    float32_t y;
    float32_t x0, x1;                            /* Nearest input values */
    float32_t y0, y1;                            /* Nearest output values */
    float32_t xSpacing = S->xSpacing;            /* spacing between input values */
    int32_t i;                                   /* Index variable */
    float32_t *pYData = S->pYData;               /* pointer to output table */

    /* Calculation of index */
    i = (int32_t) ((x - S->x1) / xSpacing);

    if (i < 0)
    {
      /* Iniatilize output for below specified range as least output value of table */
      y = pYData[0];
    }
    else if ((uint32_t)i >= (S->nValues - 1))
    {
      /* Iniatilize output for above specified range as last output value of table */
      y = pYData[S->nValues - 1];
    }
    else
    {
      /* Calculation of nearest input values */
      x0 = S->x1 +  i      * xSpacing;
      x1 = S->x1 + (i + 1) * xSpacing;

      /* Read of nearest output values */
      y0 = pYData[i];
      y1 = pYData[i + 1];

      /* Calculation of output */
      y = y0 + (x - x0) * ((y1 - y0) / (x1 - x0));

    }

    /* returns output value */
    return (y);
  }


   /**
   *
   * @brief  Process function for the Q31 Linear Interpolation Function.
   * @param[in] pYData   pointer to Q31 Linear Interpolation table
   * @param[in] x        input sample to process
   * @param[in] nValues  number of table values
   * @return y processed output sample.
   *
   * \par
   * Input sample <code>x</code> is in 12.20 format which contains 12 bits for table index and 20 bits for fractional part.
   * This function can support maximum of table size 2^12.
   *
   */
  __STATIC_FORCEINLINE q31_t arm_linear_interp_q31(
  q31_t * pYData,
  q31_t x,
  uint32_t nValues)
  {
    q31_t y;                                     /* output */
    q31_t y0, y1;                                /* Nearest output values */
    q31_t fract;                                 /* fractional part */
    int32_t index;                               /* Index to read nearest output values */

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    index = ((x & (q31_t)0xFFF00000) >> 20);

    if (index >= (int32_t)(nValues - 1))
    {
      return (pYData[nValues - 1]);
    }
    else if (index < 0)
    {
      return (pYData[0]);
    }
    else
    {
      /* 20 bits for the fractional part */
      /* shift left by 11 to keep fract in 1.31 format */
      fract = (x & 0x000FFFFF) << 11;

      /* Read two nearest output values from the index in 1.31(q31) format */
      y0 = pYData[index];
      y1 = pYData[index + 1];

      /* Calculation of y0 * (1-fract) and y is in 2.30 format */
      y = ((q31_t) ((q63_t) y0 * (0x7FFFFFFF - fract) >> 32));

      /* Calculation of y0 * (1-fract) + y1 *fract and y is in 2.30 format */
      y += ((q31_t) (((q63_t) y1 * fract) >> 32));

      /* Convert y to 1.31 format */
      return (y << 1U);
    }
  }


  /**
   *
   * @brief  Process function for the Q15 Linear Interpolation Function.
   * @param[in] pYData   pointer to Q15 Linear Interpolation table
   * @param[in] x        input sample to process
   * @param[in] nValues  number of table values
   * @return y processed output sample.
   *
   * \par
   * Input sample <code>x</code> is in 12.20 format which contains 12 bits for table index and 20 bits for fractional part.
   * This function can support maximum of table size 2^12.
   *
   */
  __STATIC_FORCEINLINE q15_t arm_linear_interp_q15(
  q15_t * pYData,
  q31_t x,
  uint32_t nValues)
  {
    q63_t y;                                     /* output */
    q15_t y0, y1;                                /* Nearest output values */
    q31_t fract;                                 /* fractional part */
    int32_t index;                               /* Index to read nearest output values */

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    index = ((x & (int32_t)0xFFF00000) >> 20);

    if (index >= (int32_t)(nValues - 1))
    {
      return (pYData[nValues - 1]);
    }
    else if (index < 0)
    {
      return (pYData[0]);
    }
    else
    {
      /* 20 bits for the fractional part */
      /* fract is in 12.20 format */
      fract = (x & 0x000FFFFF);

      /* Read two nearest output values from the index */
      y0 = pYData[index];
      y1 = pYData[index + 1];

      /* Calculation of y0 * (1-fract) and y is in 13.35 format */
      y = ((q63_t) y0 * (0xFFFFF - fract));

      /* Calculation of (y0 * (1-fract) + y1 * fract) and y is in 13.35 format */
      y += ((q63_t) y1 * (fract));

      /* convert y to 1.15 format */
      return (q15_t) (y >> 20);
    }
  }


  /**
   *
   * @brief  Process function for the Q7 Linear Interpolation Function.
   * @param[in] pYData   pointer to Q7 Linear Interpolation table
   * @param[in] x        input sample to process
   * @param[in] nValues  number of table values
   * @return y processed output sample.
   *
   * \par
   * Input sample <code>x</code> is in 12.20 format which contains 12 bits for table index and 20 bits for fractional part.
   * This function can support maximum of table size 2^12.
   */
  __STATIC_FORCEINLINE q7_t arm_linear_interp_q7(
  q7_t * pYData,
  q31_t x,
  uint32_t nValues)
  {
    q31_t y;                                     /* output */
    q7_t y0, y1;                                 /* Nearest output values */
    q31_t fract;                                 /* fractional part */
    uint32_t index;                              /* Index to read nearest output values */

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    if (x < 0)
    {
      return (pYData[0]);
    }
    index = (x >> 20) & 0xfff;

    if (index >= (nValues - 1))
    {
      return (pYData[nValues - 1]);
    }
    else
    {
      /* 20 bits for the fractional part */
      /* fract is in 12.20 format */
      fract = (x & 0x000FFFFF);

      /* Read two nearest output values from the index and are in 1.7(q7) format */
      y0 = pYData[index];
      y1 = pYData[index + 1];

      /* Calculation of y0 * (1-fract ) and y is in 13.27(q27) format */
      y = ((y0 * (0xFFFFF - fract)));

      /* Calculation of y1 * fract + y0 * (1-fract) and y is in 13.27(q27) format */
      y += (y1 * fract);

      /* convert y to 1.7(q7) format */
      return (q7_t) (y >> 20);
     }
  }

  /**
   * @} end of LinearInterpolate group
   */

  /**
   * @brief  Fast approximation to the trigonometric sine function for floating-point data.
   * @param[in] x  input value in radians.
   * @return  sin(x).
   */
  float32_t arm_sin_f32(
  float32_t x);


  /**
   * @brief  Fast approximation to the trigonometric sine function for Q31 data.
   * @param[in] x  Scaled input value in radians.
   * @return  sin(x).
   */
  q31_t arm_sin_q31(
  q31_t x);


  /**
   * @brief  Fast approximation to the trigonometric sine function for Q15 data.
   * @param[in] x  Scaled input value in radians.
   * @return  sin(x).
   */
  q15_t arm_sin_q15(
  q15_t x);


  /**
   * @brief  Fast approximation to the trigonometric cosine function for floating-point data.
   * @param[in] x  input value in radians.
   * @return  cos(x).
   */
  float32_t arm_cos_f32(
  float32_t x);


  /**
   * @brief Fast approximation to the trigonometric cosine function for Q31 data.
   * @param[in] x  Scaled input value in radians.
   * @return  cos(x).
   */
  q31_t arm_cos_q31(
  q31_t x);


  /**
   * @brief  Fast approximation to the trigonometric cosine function for Q15 data.
   * @param[in] x  Scaled input value in radians.
   * @return  cos(x).
   */
  q15_t arm_cos_q15(
  q15_t x);


/**
  @brief         Floating-point vector of log values.
  @param[in]     pSrc       points to the input vector
  @param[out]    pDst       points to the output vector
  @param[in]     blockSize  number of samples in each vector
  @return        none
 */
  void arm_vlog_f32(
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize);

/**
  @brief         Floating-point vector of exp values.
  @param[in]     pSrc       points to the input vector
  @param[out]    pDst       points to the output vector
  @param[in]     blockSize  number of samples in each vector
  @return        none
 */
  void arm_vexp_f32(
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize);

  /**
   * @ingroup groupFastMath
   */


  /**
   * @defgroup SQRT Square Root
   *
   * Computes the square root of a number.
   * There are separate functions for Q15, Q31, and floating-point data types.
   * The square root function is computed using the Newton-Raphson algorithm.
   * This is an iterative algorithm of the form:
   * <pre>
   *      x1 = x0 - f(x0)/f'(x0)
   * </pre>
   * where <code>x1</code> is the current estimate,
   * <code>x0</code> is the previous estimate, and
   * <code>f'(x0)</code> is the derivative of <code>f()</code> evaluated at <code>x0</code>.
   * For the square root function, the algorithm reduces to:
   * <pre>
   *     x0 = in/2                         [initial guess]
   *     x1 = 1/2 * ( x0 + in / x0)        [each iteration]
   * </pre>
   */


  /**
   * @addtogroup SQRT
   * @{
   */

/**
  @brief         Floating-point square root function.
  @param[in]     in    input value
  @param[out]    pOut  square root of input value
  @return        execution status
                   - \ref ARM_MATH_SUCCESS        : input value is positive
                   - \ref ARM_MATH_ARGUMENT_ERROR : input value is negative; *pOut is set to 0
 */
__STATIC_FORCEINLINE arm_status arm_sqrt_f32(
  float32_t in,
  float32_t * pOut)
  {
    if (in >= 0.0f)
    {
#if defined ( __CC_ARM )
  #if defined __TARGET_FPU_VFP
      *pOut = __sqrtf(in);
  #else
      *pOut = sqrtf(in);
  #endif

#elif defined ( __ICCARM__ )
  #if defined __ARMVFP__
      __ASM("VSQRT.F32 %0,%1" : "=t"(*pOut) : "t"(in));
  #else
      *pOut = sqrtf(in);
  #endif

#else
      *pOut = sqrtf(in);
#endif

      return (ARM_MATH_SUCCESS);
    }
    else
    {
      *pOut = 0.0f;
      return (ARM_MATH_ARGUMENT_ERROR);
    }
  }


/**
  @brief         Q31 square root function.
  @param[in]     in    input value.  The range of the input value is [0 +1) or 0x00000000 to 0x7FFFFFFF
  @param[out]    pOut  points to square root of input value
  @return        execution status
                   - \ref ARM_MATH_SUCCESS        : input value is positive
                   - \ref ARM_MATH_ARGUMENT_ERROR : input value is negative; *pOut is set to 0
 */
arm_status arm_sqrt_q31(
  q31_t in,
  q31_t * pOut);


/**
  @brief         Q15 square root function.
  @param[in]     in    input value.  The range of the input value is [0 +1) or 0x0000 to 0x7FFF
  @param[out]    pOut  points to square root of input value
  @return        execution status
                   - \ref ARM_MATH_SUCCESS        : input value is positive
                   - \ref ARM_MATH_ARGUMENT_ERROR : input value is negative; *pOut is set to 0
 */
arm_status arm_sqrt_q15(
  q15_t in,
  q15_t * pOut);

  /**
   * @brief  Vector Floating-point square root function.
   * @param[in]  pIn   input vector.
   * @param[out] pOut  vector of square roots of input elements.
   * @param[in]  len   length of input vector.
   * @return The function returns ARM_MATH_SUCCESS if input value is positive value or ARM_MATH_ARGUMENT_ERROR if
   * <code>in</code> is negative value and returns zero output for negative values.
   */
  void arm_vsqrt_f32(
  float32_t * pIn,
  float32_t * pOut,
  uint16_t len);

  void arm_vsqrt_q31(
  q31_t * pIn,
  q31_t * pOut,
  uint16_t len);

  void arm_vsqrt_q15(
  q15_t * pIn,
  q15_t * pOut,
  uint16_t len);

  /**
   * @} end of SQRT group
   */


  /**
   * @brief floating-point Circular write function.
   */
  __STATIC_FORCEINLINE void arm_circularWrite_f32(
  int32_t * circBuffer,
  int32_t L,
  uint16_t * writeOffset,
  int32_t bufferInc,
  const int32_t * src,
  int32_t srcInc,
  uint32_t blockSize)
  {
    uint32_t i = 0U;
    int32_t wOffset;

    /* Copy the value of Index pointer that points
     * to the current location where the input samples to be copied */
    wOffset = *writeOffset;

    /* Loop over the blockSize */
    i = blockSize;

    while (i > 0U)
    {
      /* copy the input sample to the circular buffer */
      circBuffer[wOffset] = *src;

      /* Update the input pointer */
      src += srcInc;

      /* Circularly update wOffset.  Watch out for positive and negative value */
      wOffset += bufferInc;
      if (wOffset >= L)
        wOffset -= L;

      /* Decrement the loop counter */
      i--;
    }

    /* Update the index pointer */
    *writeOffset = (uint16_t)wOffset;
  }



  /**
   * @brief floating-point Circular Read function.
   */
  __STATIC_FORCEINLINE void arm_circularRead_f32(
  int32_t * circBuffer,
  int32_t L,
  int32_t * readOffset,
  int32_t bufferInc,
  int32_t * dst,
  int32_t * dst_base,
  int32_t dst_length,
  int32_t dstInc,
  uint32_t blockSize)
  {
    uint32_t i = 0U;
    int32_t rOffset;
    int32_t* dst_end;

    /* Copy the value of Index pointer that points
     * to the current location from where the input samples to be read */
    rOffset = *readOffset;
    dst_end = dst_base + dst_length;

    /* Loop over the blockSize */
    i = blockSize;

    while (i > 0U)
    {
      /* copy the sample from the circular buffer to the destination buffer */
      *dst = circBuffer[rOffset];

      /* Update the input pointer */
      dst += dstInc;

      if (dst == dst_end)
      {
        dst = dst_base;
      }

      /* Circularly update rOffset.  Watch out for positive and negative value  */
      rOffset += bufferInc;

      if (rOffset >= L)
      {
        rOffset -= L;
      }

      /* Decrement the loop counter */
      i--;
    }

    /* Update the index pointer */
    *readOffset = rOffset;
  }


  /**
   * @brief Q15 Circular write function.
   */
  __STATIC_FORCEINLINE void arm_circularWrite_q15(
  q15_t * circBuffer,
  int32_t L,
  uint16_t * writeOffset,
  int32_t bufferInc,
  const q15_t * src,
  int32_t srcInc,
  uint32_t blockSize)
  {
    uint32_t i = 0U;
    int32_t wOffset;

    /* Copy the value of Index pointer that points
     * to the current location where the input samples to be copied */
    wOffset = *writeOffset;

    /* Loop over the blockSize */
    i = blockSize;

    while (i > 0U)
    {
      /* copy the input sample to the circular buffer */
      circBuffer[wOffset] = *src;

      /* Update the input pointer */
      src += srcInc;

      /* Circularly update wOffset.  Watch out for positive and negative value */
      wOffset += bufferInc;
      if (wOffset >= L)
        wOffset -= L;

      /* Decrement the loop counter */
      i--;
    }

    /* Update the index pointer */
    *writeOffset = (uint16_t)wOffset;
  }


  /**
   * @brief Q15 Circular Read function.
   */
  __STATIC_FORCEINLINE void arm_circularRead_q15(
  q15_t * circBuffer,
  int32_t L,
  int32_t * readOffset,
  int32_t bufferInc,
  q15_t * dst,
  q15_t * dst_base,
  int32_t dst_length,
  int32_t dstInc,
  uint32_t blockSize)
  {
    uint32_t i = 0;
    int32_t rOffset;
    q15_t* dst_end;

    /* Copy the value of Index pointer that points
     * to the current location from where the input samples to be read */
    rOffset = *readOffset;

    dst_end = dst_base + dst_length;

    /* Loop over the blockSize */
    i = blockSize;

    while (i > 0U)
    {
      /* copy the sample from the circular buffer to the destination buffer */
      *dst = circBuffer[rOffset];

      /* Update the input pointer */
      dst += dstInc;

      if (dst == dst_end)
      {
        dst = dst_base;
      }

      /* Circularly update wOffset.  Watch out for positive and negative value */
      rOffset += bufferInc;

      if (rOffset >= L)
      {
        rOffset -= L;
      }

      /* Decrement the loop counter */
      i--;
    }

    /* Update the index pointer */
    *readOffset = rOffset;
  }


  /**
   * @brief Q7 Circular write function.
   */
  __STATIC_FORCEINLINE void arm_circularWrite_q7(
  q7_t * circBuffer,
  int32_t L,
  uint16_t * writeOffset,
  int32_t bufferInc,
  const q7_t * src,
  int32_t srcInc,
  uint32_t blockSize)
  {
    uint32_t i = 0U;
    int32_t wOffset;

    /* Copy the value of Index pointer that points
     * to the current location where the input samples to be copied */
    wOffset = *writeOffset;

    /* Loop over the blockSize */
    i = blockSize;

    while (i > 0U)
    {
      /* copy the input sample to the circular buffer */
      circBuffer[wOffset] = *src;

      /* Update the input pointer */
      src += srcInc;

      /* Circularly update wOffset.  Watch out for positive and negative value */
      wOffset += bufferInc;
      if (wOffset >= L)
        wOffset -= L;

      /* Decrement the loop counter */
      i--;
    }

    /* Update the index pointer */
    *writeOffset = (uint16_t)wOffset;
  }


  /**
   * @brief Q7 Circular Read function.
   */
  __STATIC_FORCEINLINE void arm_circularRead_q7(
  q7_t * circBuffer,
  int32_t L,
  int32_t * readOffset,
  int32_t bufferInc,
  q7_t * dst,
  q7_t * dst_base,
  int32_t dst_length,
  int32_t dstInc,
  uint32_t blockSize)
  {
    uint32_t i = 0;
    int32_t rOffset;
    q7_t* dst_end;

    /* Copy the value of Index pointer that points
     * to the current location from where the input samples to be read */
    rOffset = *readOffset;

    dst_end = dst_base + dst_length;

    /* Loop over the blockSize */
    i = blockSize;

    while (i > 0U)
    {
      /* copy the sample from the circular buffer to the destination buffer */
      *dst = circBuffer[rOffset];

      /* Update the input pointer */
      dst += dstInc;

      if (dst == dst_end)
      {
        dst = dst_base;
      }

      /* Circularly update rOffset.  Watch out for positive and negative value */
      rOffset += bufferInc;

      if (rOffset >= L)
      {
        rOffset -= L;
      }

      /* Decrement the loop counter */
      i--;
    }

    /* Update the index pointer */
    *readOffset = rOffset;
  }


  /**
   * @brief  Sum of the squares of the elements of a Q31 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_power_q31(
  const q31_t * pSrc,
        uint32_t blockSize,
        q63_t * pResult);


  /**
   * @brief  Sum of the squares of the elements of a floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_power_f32(
  const float32_t * pSrc,
        uint32_t blockSize,
        float32_t * pResult);


  /**
   * @brief  Sum of the squares of the elements of a Q15 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_power_q15(
  const q15_t * pSrc,
        uint32_t blockSize,
        q63_t * pResult);


  /**
   * @brief  Sum of the squares of the elements of a Q7 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_power_q7(
  const q7_t * pSrc,
        uint32_t blockSize,
        q31_t * pResult);


  /**
   * @brief  Mean value of a Q7 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_mean_q7(
  const q7_t * pSrc,
        uint32_t blockSize,
        q7_t * pResult);


  /**
   * @brief  Mean value of a Q15 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_mean_q15(
  const q15_t * pSrc,
        uint32_t blockSize,
        q15_t * pResult);


  /**
   * @brief  Mean value of a Q31 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_mean_q31(
  const q31_t * pSrc,
        uint32_t blockSize,
        q31_t * pResult);


  /**
   * @brief  Mean value of a floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_mean_f32(
  const float32_t * pSrc,
        uint32_t blockSize,
        float32_t * pResult);


  /**
   * @brief  Variance of the elements of a floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_var_f32(
  const float32_t * pSrc,
        uint32_t blockSize,
        float32_t * pResult);


  /**
   * @brief  Variance of the elements of a Q31 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_var_q31(
  const q31_t * pSrc,
        uint32_t blockSize,
        q31_t * pResult);


  /**
   * @brief  Variance of the elements of a Q15 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_var_q15(
  const q15_t * pSrc,
        uint32_t blockSize,
        q15_t * pResult);


  /**
   * @brief  Root Mean Square of the elements of a floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_rms_f32(
  const float32_t * pSrc,
        uint32_t blockSize,
        float32_t * pResult);


  /**
   * @brief  Root Mean Square of the elements of a Q31 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_rms_q31(
  const q31_t * pSrc,
        uint32_t blockSize,
        q31_t * pResult);


  /**
   * @brief  Root Mean Square of the elements of a Q15 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_rms_q15(
  const q15_t * pSrc,
        uint32_t blockSize,
        q15_t * pResult);


  /**
   * @brief  Standard deviation of the elements of a floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_std_f32(
  const float32_t * pSrc,
        uint32_t blockSize,
        float32_t * pResult);


  /**
   * @brief  Standard deviation of the elements of a Q31 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_std_q31(
  const q31_t * pSrc,
        uint32_t blockSize,
        q31_t * pResult);


  /**
   * @brief  Standard deviation of the elements of a Q15 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_std_q15(
  const q15_t * pSrc,
        uint32_t blockSize,
        q15_t * pResult);


  /**
   * @brief  Floating-point complex magnitude
   * @param[in]  pSrc        points to the complex input vector
   * @param[out] pDst        points to the real output vector
   * @param[in]  numSamples  number of complex samples in the input vector
   */
  void arm_cmplx_mag_f32(
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t numSamples);


  /**
   * @brief  Q31 complex magnitude
   * @param[in]  pSrc        points to the complex input vector
   * @param[out] pDst        points to the real output vector
   * @param[in]  numSamples  number of complex samples in the input vector
   */
  void arm_cmplx_mag_q31(
  const q31_t * pSrc,
        q31_t * pDst,
        uint32_t numSamples);


  /**
   * @brief  Q15 complex magnitude
   * @param[in]  pSrc        points to the complex input vector
   * @param[out] pDst        points to the real output vector
   * @param[in]  numSamples  number of complex samples in the input vector
   */
  void arm_cmplx_mag_q15(
  const q15_t * pSrc,
        q15_t * pDst,
        uint32_t numSamples);


  /**
   * @brief  Q15 complex dot product
   * @param[in]  pSrcA       points to the first input vector
   * @param[in]  pSrcB       points to the second input vector
   * @param[in]  numSamples  number of complex samples in each vector
   * @param[out] realResult  real part of the result returned here
   * @param[out] imagResult  imaginary part of the result returned here
   */
  void arm_cmplx_dot_prod_q15(
  const q15_t * pSrcA,
  const q15_t * pSrcB,
        uint32_t numSamples,
        q31_t * realResult,
        q31_t * imagResult);


  /**
   * @brief  Q31 complex dot product
   * @param[in]  pSrcA       points to the first input vector
   * @param[in]  pSrcB       points to the second input vector
   * @param[in]  numSamples  number of complex samples in each vector
   * @param[out] realResult  real part of the result returned here
   * @param[out] imagResult  imaginary part of the result returned here
   */
  void arm_cmplx_dot_prod_q31(
  const q31_t * pSrcA,
  const q31_t * pSrcB,
        uint32_t numSamples,
        q63_t * realResult,
        q63_t * imagResult);


  /**
   * @brief  Floating-point complex dot product
   * @param[in]  pSrcA       points to the first input vector
   * @param[in]  pSrcB       points to the second input vector
   * @param[in]  numSamples  number of complex samples in each vector
   * @param[out] realResult  real part of the result returned here
   * @param[out] imagResult  imaginary part of the result returned here
   */
  void arm_cmplx_dot_prod_f32(
  const float32_t * pSrcA,
  const float32_t * pSrcB,
        uint32_t numSamples,
        float32_t * realResult,
        float32_t * imagResult);


  /**
   * @brief  Q15 complex-by-real multiplication
   * @param[in]  pSrcCmplx   points to the complex input vector
   * @param[in]  pSrcReal    points to the real input vector
   * @param[out] pCmplxDst   points to the complex output vector
   * @param[in]  numSamples  number of samples in each vector
   */
  void arm_cmplx_mult_real_q15(
  const q15_t * pSrcCmplx,
  const q15_t * pSrcReal,
        q15_t * pCmplxDst,
        uint32_t numSamples);


  /**
   * @brief  Q31 complex-by-real multiplication
   * @param[in]  pSrcCmplx   points to the complex input vector
   * @param[in]  pSrcReal    points to the real input vector
   * @param[out] pCmplxDst   points to the complex output vector
   * @param[in]  numSamples  number of samples in each vector
   */
  void arm_cmplx_mult_real_q31(
  const q31_t * pSrcCmplx,
  const q31_t * pSrcReal,
        q31_t * pCmplxDst,
        uint32_t numSamples);


  /**
   * @brief  Floating-point complex-by-real multiplication
   * @param[in]  pSrcCmplx   points to the complex input vector
   * @param[in]  pSrcReal    points to the real input vector
   * @param[out] pCmplxDst   points to the complex output vector
   * @param[in]  numSamples  number of samples in each vector
   */
  void arm_cmplx_mult_real_f32(
  const float32_t * pSrcCmplx,
  const float32_t * pSrcReal,
        float32_t * pCmplxDst,
        uint32_t numSamples);


  /**
   * @brief  Minimum value of a Q7 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] result     is output pointer
   * @param[in]  index      is the array index of the minimum value in the input buffer.
   */
  void arm_min_q7(
  const q7_t * pSrc,
        uint32_t blockSize,
        q7_t * result,
        uint32_t * index);


  /**
   * @brief  Minimum value of a Q15 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output pointer
   * @param[in]  pIndex     is the array index of the minimum value in the input buffer.
   */
  void arm_min_q15(
  const q15_t * pSrc,
        uint32_t blockSize,
        q15_t * pResult,
        uint32_t * pIndex);


  /**
   * @brief  Minimum value of a Q31 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output pointer
   * @param[out] pIndex     is the array index of the minimum value in the input buffer.
   */
  void arm_min_q31(
  const q31_t * pSrc,
        uint32_t blockSize,
        q31_t * pResult,
        uint32_t * pIndex);


  /**
   * @brief  Minimum value of a floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output pointer
   * @param[out] pIndex     is the array index of the minimum value in the input buffer.
   */
  void arm_min_f32(
  const float32_t * pSrc,
        uint32_t blockSize,
        float32_t * pResult,
        uint32_t * pIndex);


/**
 * @brief Maximum value of a Q7 vector.
 * @param[in]  pSrc       points to the input buffer
 * @param[in]  blockSize  length of the input vector
 * @param[out] pResult    maximum value returned here
 * @param[out] pIndex     index of maximum value returned here
 */
  void arm_max_q7(
  const q7_t * pSrc,
        uint32_t blockSize,
        q7_t * pResult,
        uint32_t * pIndex);


/**
 * @brief Maximum value of a Q15 vector.
 * @param[in]  pSrc       points to the input buffer
 * @param[in]  blockSize  length of the input vector
 * @param[out] pResult    maximum value returned here
 * @param[out] pIndex     index of maximum value returned here
 */
  void arm_max_q15(
  const q15_t * pSrc,
        uint32_t blockSize,
        q15_t * pResult,
        uint32_t * pIndex);


/**
 * @brief Maximum value of a Q31 vector.
 * @param[in]  pSrc       points to the input buffer
 * @param[in]  blockSize  length of the input vector
 * @param[out] pResult    maximum value returned here
 * @param[out] pIndex     index of maximum value returned here
 */
  void arm_max_q31(
  const q31_t * pSrc,
        uint32_t blockSize,
        q31_t * pResult,
        uint32_t * pIndex);


/**
 * @brief Maximum value of a floating-point vector.
 * @param[in]  pSrc       points to the input buffer
 * @param[in]  blockSize  length of the input vector
 * @param[out] pResult    maximum value returned here
 * @param[out] pIndex     index of maximum value returned here
 */
  void arm_max_f32(
  const float32_t * pSrc,
        uint32_t blockSize,
        float32_t * pResult,
        uint32_t * pIndex);

  /**
    @brief         Maximum value of a floating-point vector.
    @param[in]     pSrc       points to the input vector
    @param[in]     blockSize  number of samples in input vector
    @param[out]    pResult    maximum value returned here
    @return        none
   */
  void arm_max_no_idx_f32(
      const float32_t *pSrc,
      uint32_t   blockSize,
      float32_t *pResult);

  /**
   * @brief  Q15 complex-by-complex multiplication
   * @param[in]  pSrcA       points to the first input vector
   * @param[in]  pSrcB       points to the second input vector
   * @param[out] pDst        points to the output vector
   * @param[in]  numSamples  number of complex samples in each vector
   */
  void arm_cmplx_mult_cmplx_q15(
  const q15_t * pSrcA,
  const q15_t * pSrcB,
        q15_t * pDst,
        uint32_t numSamples);


  /**
   * @brief  Q31 complex-by-complex multiplication
   * @param[in]  pSrcA       points to the first input vector
   * @param[in]  pSrcB       points to the second input vector
   * @param[out] pDst        points to the output vector
   * @param[in]  numSamples  number of complex samples in each vector
   */
  void arm_cmplx_mult_cmplx_q31(
  const q31_t * pSrcA,
  const q31_t * pSrcB,
        q31_t * pDst,
        uint32_t numSamples);


  /**
   * @brief  Floating-point complex-by-complex multiplication
   * @param[in]  pSrcA       points to the first input vector
   * @param[in]  pSrcB       points to the second input vector
   * @param[out] pDst        points to the output vector
   * @param[in]  numSamples  number of complex samples in each vector
   */
  void arm_cmplx_mult_cmplx_f32(
  const float32_t * pSrcA,
  const float32_t * pSrcB,
        float32_t * pDst,
        uint32_t numSamples);


  /**
   * @brief Converts the elements of the floating-point vector to Q31 vector.
   * @param[in]  pSrc       points to the floating-point input vector
   * @param[out] pDst       points to the Q31 output vector
   * @param[in]  blockSize  length of the input vector
   */
  void arm_float_to_q31(
  const float32_t * pSrc,
        q31_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Converts the elements of the floating-point vector to Q15 vector.
   * @param[in]  pSrc       points to the floating-point input vector
   * @param[out] pDst       points to the Q15 output vector
   * @param[in]  blockSize  length of the input vector
   */
  void arm_float_to_q15(
  const float32_t * pSrc,
        q15_t * pDst,
        uint32_t blockSize);


  /**
   * @brief Converts the elements of the floating-point vector to Q7 vector.
   * @param[in]  pSrc       points to the floating-point input vector
   * @param[out] pDst       points to the Q7 output vector
   * @param[in]  blockSize  length of the input vector
   */
  void arm_float_to_q7(
  const float32_t * pSrc,
        q7_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Converts the elements of the Q31 vector to floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[out] pDst       is output pointer
   * @param[in]  blockSize  is the number of samples to process
   */
  void arm_q31_to_float(
  const q31_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Converts the elements of the Q31 vector to Q15 vector.
   * @param[in]  pSrc       is input pointer
   * @param[out] pDst       is output pointer
   * @param[in]  blockSize  is the number of samples to process
   */
  void arm_q31_to_q15(
  const q31_t * pSrc,
        q15_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Converts the elements of the Q31 vector to Q7 vector.
   * @param[in]  pSrc       is input pointer
   * @param[out] pDst       is output pointer
   * @param[in]  blockSize  is the number of samples to process
   */
  void arm_q31_to_q7(
  const q31_t * pSrc,
        q7_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Converts the elements of the Q15 vector to floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[out] pDst       is output pointer
   * @param[in]  blockSize  is the number of samples to process
   */
  void arm_q15_to_float(
  const q15_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Converts the elements of the Q15 vector to Q31 vector.
   * @param[in]  pSrc       is input pointer
   * @param[out] pDst       is output pointer
   * @param[in]  blockSize  is the number of samples to process
   */
  void arm_q15_to_q31(
  const q15_t * pSrc,
        q31_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Converts the elements of the Q15 vector to Q7 vector.
   * @param[in]  pSrc       is input pointer
   * @param[out] pDst       is output pointer
   * @param[in]  blockSize  is the number of samples to process
   */
  void arm_q15_to_q7(
  const q15_t * pSrc,
        q7_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Converts the elements of the Q7 vector to floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[out] pDst       is output pointer
   * @param[in]  blockSize  is the number of samples to process
   */
  void arm_q7_to_float(
  const q7_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Converts the elements of the Q7 vector to Q31 vector.
   * @param[in]  pSrc       input pointer
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_q7_to_q31(
  const q7_t * pSrc,
        q31_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Converts the elements of the Q7 vector to Q15 vector.
   * @param[in]  pSrc       input pointer
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_q7_to_q15(
  const q7_t * pSrc,
        q15_t * pDst,
        uint32_t blockSize);

/**
 * @brief Struct for specifying SVM Kernel
 */
typedef enum
{
    ARM_ML_KERNEL_LINEAR = 0,
             /**< Linear kernel */
    ARM_ML_KERNEL_POLYNOMIAL = 1,
             /**< Polynomial kernel */
    ARM_ML_KERNEL_RBF = 2,
             /**< Radial Basis Function kernel */
    ARM_ML_KERNEL_SIGMOID = 3
             /**< Sigmoid kernel */
} arm_ml_kernel_type;


/**
 * @brief Instance structure for linear SVM prediction function.
 */
typedef struct
{
  uint32_t        nbOfSupportVectors;     /**< Number of support vectors */
  uint32_t        vectorDimension;        /**< Dimension of vector space */
  float32_t       intercept;              /**< Intercept */
  const float32_t *dualCoefficients;      /**< Dual coefficients */
  const float32_t *supportVectors;        /**< Support vectors */
  const int32_t   *classes;               /**< The two SVM classes */
} arm_svm_linear_instance_f32;


/**
 * @brief Instance structure for polynomial SVM prediction function.
 */
typedef struct
{
  uint32_t        nbOfSupportVectors;     /**< Number of support vectors */
  uint32_t        vectorDimension;        /**< Dimension of vector space */
  float32_t       intercept;              /**< Intercept */
  const float32_t *dualCoefficients;      /**< Dual coefficients */
  const float32_t *supportVectors;        /**< Support vectors */
  const int32_t   *classes;               /**< The two SVM classes */
  int32_t         degree;                 /**< Polynomial degree */
  float32_t       coef0;                  /**< Polynomial constant */
  float32_t       gamma;                  /**< Gamma factor */
} arm_svm_polynomial_instance_f32;

/**
 * @brief Instance structure for rbf SVM prediction function.
 */
typedef struct
{
  uint32_t        nbOfSupportVectors;     /**< Number of support vectors */
  uint32_t        vectorDimension;        /**< Dimension of vector space */
  float32_t       intercept;              /**< Intercept */
  const float32_t *dualCoefficients;      /**< Dual coefficients */
  const float32_t *supportVectors;        /**< Support vectors */
  const int32_t   *classes;               /**< The two SVM classes */
  float32_t       gamma;                  /**< Gamma factor */
} arm_svm_rbf_instance_f32;

/**
 * @brief Instance structure for sigmoid SVM prediction function.
 */
typedef struct
{
  uint32_t        nbOfSupportVectors;     /**< Number of support vectors */
  uint32_t        vectorDimension;        /**< Dimension of vector space */
  float32_t       intercept;              /**< Intercept */
  const float32_t *dualCoefficients;      /**< Dual coefficients */
  const float32_t *supportVectors;        /**< Support vectors */
  const int32_t   *classes;               /**< The two SVM classes */
  float32_t       coef0;                  /**< Independant constant */
  float32_t       gamma;                  /**< Gamma factor */
} arm_svm_sigmoid_instance_f32;

/**
 * @brief        SVM linear instance init function
 * @param[in]    S                      Parameters for SVM functions
 * @param[in]    nbOfSupportVectors     Number of support vectors
 * @param[in]    vectorDimension        Dimension of vector space
 * @param[in]    intercept              Intercept
 * @param[in]    dualCoefficients       Array of dual coefficients
 * @param[in]    supportVectors         Array of support vectors
 * @param[in]    classes                Array of 2 classes ID
 * @return none.
 *
 */


void arm_svm_linear_init_f32(arm_svm_linear_instance_f32 *S, 
  uint32_t nbOfSupportVectors,
  uint32_t vectorDimension,
  float32_t intercept,
  const float32_t *dualCoefficients,
  const float32_t *supportVectors,
  const int32_t  *classes);

/**
 * @brief SVM linear prediction
 * @param[in]    S          Pointer to an instance of the linear SVM structure.
 * @param[in]    in         Pointer to input vector
 * @param[out]   pResult    Decision value
 * @return none.
 *
 */
  
void arm_svm_linear_predict_f32(const arm_svm_linear_instance_f32 *S, 
   const float32_t * in, 
   int32_t * pResult);


/**
 * @brief        SVM polynomial instance init function
 * @param[in]    S                      points to an instance of the polynomial SVM structure.
 * @param[in]    nbOfSupportVectors     Number of support vectors
 * @param[in]    vectorDimension        Dimension of vector space
 * @param[in]    intercept              Intercept
 * @param[in]    dualCoefficients       Array of dual coefficients
 * @param[in]    supportVectors         Array of support vectors
 * @param[in]    classes                Array of 2 classes ID
 * @param[in]    degree                 Polynomial degree
 * @param[in]    coef0                  coeff0 (scikit-learn terminology)
 * @param[in]    gamma                  gamma (scikit-learn terminology)
 * @return none.
 *
 */


void arm_svm_polynomial_init_f32(arm_svm_polynomial_instance_f32 *S, 
  uint32_t nbOfSupportVectors,
  uint32_t vectorDimension,
  float32_t intercept,
  const float32_t *dualCoefficients,
  const float32_t *supportVectors,
  const int32_t   *classes,
  int32_t      degree,
  float32_t coef0,
  float32_t gamma
  );

/**
 * @brief SVM polynomial prediction
 * @param[in]    S          Pointer to an instance of the polynomial SVM structure.
 * @param[in]    in         Pointer to input vector
 * @param[out]   pResult    Decision value
 * @return none.
 *
 */
void arm_svm_polynomial_predict_f32(const arm_svm_polynomial_instance_f32 *S, 
   const float32_t * in, 
   int32_t * pResult);


/**
 * @brief        SVM radial basis function instance init function
 * @param[in]    S                      points to an instance of the polynomial SVM structure.
 * @param[in]    nbOfSupportVectors     Number of support vectors
 * @param[in]    vectorDimension        Dimension of vector space
 * @param[in]    intercept              Intercept
 * @param[in]    dualCoefficients       Array of dual coefficients
 * @param[in]    supportVectors         Array of support vectors
 * @param[in]    classes                Array of 2 classes ID
 * @param[in]    gamma                  gamma (scikit-learn terminology)
 * @return none.
 *
 */

void arm_svm_rbf_init_f32(arm_svm_rbf_instance_f32 *S, 
  uint32_t nbOfSupportVectors,
  uint32_t vectorDimension,
  float32_t intercept,
  const float32_t *dualCoefficients,
  const float32_t *supportVectors,
  const int32_t   *classes,
  float32_t gamma
  );

/**
 * @brief SVM rbf prediction
 * @param[in]    S         Pointer to an instance of the rbf SVM structure.
 * @param[in]    in        Pointer to input vector
 * @param[out]   pResult   decision value
 * @return none.
 *
 */
void arm_svm_rbf_predict_f32(const arm_svm_rbf_instance_f32 *S, 
   const float32_t * in, 
   int32_t * pResult);

/**
 * @brief        SVM sigmoid instance init function
 * @param[in]    S                      points to an instance of the rbf SVM structure.
 * @param[in]    nbOfSupportVectors     Number of support vectors
 * @param[in]    vectorDimension        Dimension of vector space
 * @param[in]    intercept              Intercept
 * @param[in]    dualCoefficients       Array of dual coefficients
 * @param[in]    supportVectors         Array of support vectors
 * @param[in]    classes                Array of 2 classes ID
 * @param[in]    coef0                  coeff0 (scikit-learn terminology)
 * @param[in]    gamma                  gamma (scikit-learn terminology)
 * @return none.
 *
 */

void arm_svm_sigmoid_init_f32(arm_svm_sigmoid_instance_f32 *S, 
  uint32_t nbOfSupportVectors,
  uint32_t vectorDimension,
  float32_t intercept,
  const float32_t *dualCoefficients,
  const float32_t *supportVectors,
  const int32_t   *classes,
  float32_t coef0,
  float32_t gamma
  );

/**
 * @brief SVM sigmoid prediction
 * @param[in]    S        Pointer to an instance of the rbf SVM structure.
 * @param[in]    in       Pointer to input vector
 * @param[out]   pResult  Decision value
 * @return none.
 *
 */
void arm_svm_sigmoid_predict_f32(const arm_svm_sigmoid_instance_f32 *S, 
   const float32_t * in, 
   int32_t * pResult);



/**
 * @brief Instance structure for Naive Gaussian Bayesian estimator.
 */
typedef struct
{
  uint32_t vectorDimension;  /**< Dimension of vector space */
  uint32_t numberOfClasses;  /**< Number of different classes  */
  const float32_t *theta;          /**< Mean values for the Gaussians */
  const float32_t *sigma;          /**< Variances for the Gaussians */
  const float32_t *classPriors;    /**< Class prior probabilities */
  float32_t epsilon;         /**< Additive value to variances */
} arm_gaussian_naive_bayes_instance_f32;

/**
 * @brief Naive Gaussian Bayesian Estimator
 *
 * @param[in]  S         points to a naive bayes instance structure
 * @param[in]  in        points to the elements of the input vector.
 * @param[in]  pBuffer   points to a buffer of length numberOfClasses
 * @return The predicted class
 *
 */


uint32_t arm_gaussian_naive_bayes_predict_f32(const arm_gaussian_naive_bayes_instance_f32 *S, 
   const float32_t * in, 
   float32_t *pBuffer);

/**
 * @brief Computation of the LogSumExp
 *
 * In probabilistic computations, the dynamic of the probability values can be very
 * wide because they come from gaussian functions.
 * To avoid underflow and overflow issues, the values are represented by their log.
 * In this representation, multiplying the original exp values is easy : their logs are added.
 * But adding the original exp values is requiring some special handling and it is the
 * goal of the LogSumExp function.
 *
 * If the values are x1...xn, the function is computing:
 *
 * ln(exp(x1) + ... + exp(xn)) and the computation is done in such a way that
 * rounding issues are minimised.
 *
 * The max xm of the values is extracted and the function is computing:
 * xm + ln(exp(x1 - xm) + ... + exp(xn - xm))
 *
 * @param[in]  *in         Pointer to an array of input values.
 * @param[in]  blockSize   Number of samples in the input array.
 * @return LogSumExp
 *
 */


float32_t arm_logsumexp_f32(const float32_t *in, uint32_t blockSize);

/**
 * @brief Dot product with log arithmetic
 *
 * Vectors are containing the log of the samples
 *
 * @param[in]       pSrcA points to the first input vector
 * @param[in]       pSrcB points to the second input vector
 * @param[in]       blockSize number of samples in each vector
 * @param[in]       pTmpBuffer temporary buffer of length blockSize
 * @return The log of the dot product .
 *
 */


float32_t arm_logsumexp_dot_prod_f32(const float32_t * pSrcA,
  const float32_t * pSrcB,
  uint32_t blockSize,
  float32_t *pTmpBuffer);

/**
 * @brief Entropy
 *
 * @param[in]  pSrcA        Array of input values.
 * @param[in]  blockSize    Number of samples in the input array.
 * @return     Entropy      -Sum(p ln p)
 *
 */


float32_t arm_entropy_f32(const float32_t * pSrcA,uint32_t blockSize);


/**
 * @brief Entropy
 *
 * @param[in]  pSrcA        Array of input values.
 * @param[in]  blockSize    Number of samples in the input array.
 * @return     Entropy      -Sum(p ln p)
 *
 */


float64_t arm_entropy_f64(const float64_t * pSrcA, uint32_t blockSize);


/**
 * @brief Kullback-Leibler
 *
 * @param[in]  pSrcA         Pointer to an array of input values for probability distribution A.
 * @param[in]  pSrcB         Pointer to an array of input values for probability distribution B.
 * @param[in]  blockSize     Number of samples in the input array.
 * @return Kullback-Leibler  Divergence D(A || B)
 *
 */
float32_t arm_kullback_leibler_f32(const float32_t * pSrcA
  ,const float32_t * pSrcB
  ,uint32_t blockSize);


/**
 * @brief Kullback-Leibler
 *
 * @param[in]  pSrcA         Pointer to an array of input values for probability distribution A.
 * @param[in]  pSrcB         Pointer to an array of input values for probability distribution B.
 * @param[in]  blockSize     Number of samples in the input array.
 * @return Kullback-Leibler  Divergence D(A || B)
 *
 */
float64_t arm_kullback_leibler_f64(const float64_t * pSrcA, 
                const float64_t * pSrcB, 
                uint32_t blockSize);


/**
 * @brief Weighted sum
 *
 *
 * @param[in]    *in           Array of input values.
 * @param[in]    *weigths      Weights
 * @param[in]    blockSize     Number of samples in the input array.
 * @return Weighted sum
 *
 */
float32_t arm_weighted_sum_f32(const float32_t *in
  , const float32_t *weigths
  , uint32_t blockSize);


/**
 * @brief Barycenter
 *
 *
 * @param[in]    in         List of vectors
 * @param[in]    weights    Weights of the vectors
 * @param[out]   out        Barycenter
 * @param[in]    nbVectors  Number of vectors
 * @param[in]    vecDim     Dimension of space (vector dimension)
 * @return       None
 *
 */
void arm_barycenter_f32(const float32_t *in
  , const float32_t *weights
  , float32_t *out
  , uint32_t nbVectors
  , uint32_t vecDim);

/**
 * @brief        Euclidean distance between two vectors
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 *
 */

float32_t arm_euclidean_distance_f32(const float32_t *pA,const float32_t *pB, uint32_t blockSize);

/**
 * @brief        Bray-Curtis distance between two vectors
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 *
 */
float32_t arm_braycurtis_distance_f32(const float32_t *pA,const float32_t *pB, uint32_t blockSize);

/**
 * @brief        Canberra distance between two vectors
 *
 * This function may divide by zero when samples pA[i] and pB[i] are both zero.
 * The result of the computation will be correct. So the division per zero may be
 * ignored.
 *
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 *
 */
float32_t arm_canberra_distance_f32(const float32_t *pA,const float32_t *pB, uint32_t blockSize);


/**
 * @brief        Chebyshev distance between two vectors
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 *
 */
float32_t arm_chebyshev_distance_f32(const float32_t *pA,const float32_t *pB, uint32_t blockSize);


/**
 * @brief        Cityblock (Manhattan) distance between two vectors
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 *
 */
float32_t arm_cityblock_distance_f32(const float32_t *pA,const float32_t *pB, uint32_t blockSize);

/**
 * @brief        Correlation distance between two vectors
 *
 * The input vectors are modified in place !
 *
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 *
 */
float32_t arm_correlation_distance_f32(float32_t *pA,float32_t *pB, uint32_t blockSize);

/**
 * @brief        Cosine distance between two vectors
 *
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 *
 */

float32_t arm_cosine_distance_f32(const float32_t *pA,const float32_t *pB, uint32_t blockSize);

/**
 * @brief        Jensen-Shannon distance between two vectors
 *
 * This function is assuming that elements of second vector are > 0
 * and 0 only when the corresponding element of first vector is 0.
 * Otherwise the result of the computation does not make sense
 * and for speed reasons, the cases returning NaN or Infinity are not
 * managed.
 *
 * When the function is computing x log (x / y) with x 0 and y 0,
 * it will compute the right value (0) but a division per zero will occur
 * and shoudl be ignored in client code.
 *
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 *
 */

float32_t arm_jensenshannon_distance_f32(const float32_t *pA,const float32_t *pB,uint32_t blockSize);

/**
 * @brief        Minkowski distance between two vectors
 *
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    n          Norm order (>= 2)
 * @param[in]    blockSize  vector length
 * @return distance
 *
 */



float32_t arm_minkowski_distance_f32(const float32_t *pA,const float32_t *pB, int32_t order, uint32_t blockSize);

/**
 * @brief        Dice distance between two vectors
 *
 * @param[in]    pA              First vector of packed booleans
 * @param[in]    pB              Second vector of packed booleans
 * @param[in]    order           Distance order
 * @param[in]    blockSize       Number of samples
 * @return distance
 *
 */


float32_t arm_dice_distance(const uint32_t *pA, const uint32_t *pB, uint32_t numberOfBools);

/**
 * @brief        Hamming distance between two vectors
 *
 * @param[in]    pA              First vector of packed booleans
 * @param[in]    pB              Second vector of packed booleans
 * @param[in]    numberOfBools   Number of booleans
 * @return distance
 *
 */

float32_t arm_hamming_distance(const uint32_t *pA, const uint32_t *pB, uint32_t numberOfBools);

/**
 * @brief        Jaccard distance between two vectors
 *
 * @param[in]    pA              First vector of packed booleans
 * @param[in]    pB              Second vector of packed booleans
 * @param[in]    numberOfBools   Number of booleans
 * @return distance
 *
 */

float32_t arm_jaccard_distance(const uint32_t *pA, const uint32_t *pB, uint32_t numberOfBools);

/**
 * @brief        Kulsinski distance between two vectors
 *
 * @param[in]    pA              First vector of packed booleans
 * @param[in]    pB              Second vector of packed booleans
 * @param[in]    numberOfBools   Number of booleans
 * @return distance
 *
 */

float32_t arm_kulsinski_distance(const uint32_t *pA, const uint32_t *pB, uint32_t numberOfBools);

/**
 * @brief        Roger Stanimoto distance between two vectors
 *
 * @param[in]    pA              First vector of packed booleans
 * @param[in]    pB              Second vector of packed booleans
 * @param[in]    numberOfBools   Number of booleans
 * @return distance
 *
 */

float32_t arm_rogerstanimoto_distance(const uint32_t *pA, const uint32_t *pB, uint32_t numberOfBools);

/**
 * @brief        Russell-Rao distance between two vectors
 *
 * @param[in]    pA              First vector of packed booleans
 * @param[in]    pB              Second vector of packed booleans
 * @param[in]    numberOfBools   Number of booleans
 * @return distance
 *
 */

float32_t arm_russellrao_distance(const uint32_t *pA, const uint32_t *pB, uint32_t numberOfBools);

/**
 * @brief        Sokal-Michener distance between two vectors
 *
 * @param[in]    pA              First vector of packed booleans
 * @param[in]    pB              Second vector of packed booleans
 * @param[in]    numberOfBools   Number of booleans
 * @return distance
 *
 */

float32_t arm_sokalmichener_distance(const uint32_t *pA, const uint32_t *pB, uint32_t numberOfBools);

/**
 * @brief        Sokal-Sneath distance between two vectors
 *
 * @param[in]    pA              First vector of packed booleans
 * @param[in]    pB              Second vector of packed booleans
 * @param[in]    numberOfBools   Number of booleans
 * @return distance
 *
 */

float32_t arm_sokalsneath_distance(const uint32_t *pA, const uint32_t *pB, uint32_t numberOfBools);

/**
 * @brief        Yule distance between two vectors
 *
 * @param[in]    pA              First vector of packed booleans
 * @param[in]    pB              Second vector of packed booleans
 * @param[in]    numberOfBools   Number of booleans
 * @return distance
 *
 */

float32_t arm_yule_distance(const uint32_t *pA, const uint32_t *pB, uint32_t numberOfBools);


  /**
   * @ingroup groupInterpolation
   */

  /**
   * @defgroup BilinearInterpolate Bilinear Interpolation
   *
   * Bilinear interpolation is an extension of linear interpolation applied to a two dimensional grid.
   * The underlying function <code>f(x, y)</code> is sampled on a regular grid and the interpolation process
   * determines values between the grid points.
   * Bilinear interpolation is equivalent to two step linear interpolation, first in the x-dimension and then in the y-dimension.
   * Bilinear interpolation is often used in image processing to rescale images.
   * The CMSIS DSP library provides bilinear interpolation functions for Q7, Q15, Q31, and floating-point data types.
   *
   * <b>Algorithm</b>
   * \par
   * The instance structure used by the bilinear interpolation functions describes a two dimensional data table.
   * For floating-point, the instance structure is defined as:
   * <pre>
   *   typedef struct
   *   {
   *     uint16_t numRows;
   *     uint16_t numCols;
   *     float32_t *pData;
   * } arm_bilinear_interp_instance_f32;
   * </pre>
   *
   * \par
   * where <code>numRows</code> specifies the number of rows in the table;
   * <code>numCols</code> specifies the number of columns in the table;
   * and <code>pData</code> points to an array of size <code>numRows*numCols</code> values.
   * The data table <code>pTable</code> is organized in row order and the supplied data values fall on integer indexes.
   * That is, table element (x,y) is located at <code>pTable[x + y*numCols]</code> where x and y are integers.
   *
   * \par
   * Let <code>(x, y)</code> specify the desired interpolation point.  Then define:
   * <pre>
   *     XF = floor(x)
   *     YF = floor(y)
   * </pre>
   * \par
   * The interpolated output point is computed as:
   * <pre>
   *  f(x, y) = f(XF, YF) * (1-(x-XF)) * (1-(y-YF))
   *           + f(XF+1, YF) * (x-XF)*(1-(y-YF))
   *           + f(XF, YF+1) * (1-(x-XF))*(y-YF)
   *           + f(XF+1, YF+1) * (x-XF)*(y-YF)
   * </pre>
   * Note that the coordinates (x, y) contain integer and fractional components.
   * The integer components specify which portion of the table to use while the
   * fractional components control the interpolation processor.
   *
   * \par
   * if (x,y) are outside of the table boundary, Bilinear interpolation returns zero output.
   */


  /**
   * @addtogroup BilinearInterpolate
   * @{
   */

  /**
  * @brief  Floating-point bilinear interpolation.
  * @param[in,out] S  points to an instance of the interpolation structure.
  * @param[in]     X  interpolation coordinate.
  * @param[in]     Y  interpolation coordinate.
  * @return out interpolated value.
  */
  __STATIC_FORCEINLINE float32_t arm_bilinear_interp_f32(
  const arm_bilinear_interp_instance_f32 * S,
  float32_t X,
  float32_t Y)
  {
    float32_t out;
    float32_t f00, f01, f10, f11;
    float32_t *pData = S->pData;
    int32_t xIndex, yIndex, index;
    float32_t xdiff, ydiff;
    float32_t b1, b2, b3, b4;

    xIndex = (int32_t) X;
    yIndex = (int32_t) Y;

    /* Care taken for table outside boundary */
    /* Returns zero output when values are outside table boundary */
    if (xIndex < 0 || xIndex > (S->numCols - 2) || yIndex < 0 || yIndex > (S->numRows - 2))
    {
      return (0);
    }

    /* Calculation of index for two nearest points in X-direction */
    index = (xIndex ) + (yIndex ) * S->numCols;


    /* Read two nearest points in X-direction */
    f00 = pData[index];
    f01 = pData[index + 1];

    /* Calculation of index for two nearest points in Y-direction */
    index = (xIndex ) + (yIndex+1) * S->numCols;


    /* Read two nearest points in Y-direction */
    f10 = pData[index];
    f11 = pData[index + 1];

    /* Calculation of intermediate values */
    b1 = f00;
    b2 = f01 - f00;
    b3 = f10 - f00;
    b4 = f00 - f01 - f10 + f11;

    /* Calculation of fractional part in X */
    xdiff = X - xIndex;

    /* Calculation of fractional part in Y */
    ydiff = Y - yIndex;

    /* Calculation of bi-linear interpolated output */
    out = b1 + b2 * xdiff + b3 * ydiff + b4 * xdiff * ydiff;

    /* return to application */
    return (out);
  }


  /**
  * @brief  Q31 bilinear interpolation.
  * @param[in,out] S  points to an instance of the interpolation structure.
  * @param[in]     X  interpolation coordinate in 12.20 format.
  * @param[in]     Y  interpolation coordinate in 12.20 format.
  * @return out interpolated value.
  */
  __STATIC_FORCEINLINE q31_t arm_bilinear_interp_q31(
  arm_bilinear_interp_instance_q31 * S,
  q31_t X,
  q31_t Y)
  {
    q31_t out;                                   /* Temporary output */
    q31_t acc = 0;                               /* output */
    q31_t xfract, yfract;                        /* X, Y fractional parts */
    q31_t x1, x2, y1, y2;                        /* Nearest output values */
    int32_t rI, cI;                              /* Row and column indices */
    q31_t *pYData = S->pData;                    /* pointer to output table values */
    uint32_t nCols = S->numCols;                 /* num of rows */

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    rI = ((X & (q31_t)0xFFF00000) >> 20);

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    cI = ((Y & (q31_t)0xFFF00000) >> 20);

    /* Care taken for table outside boundary */
    /* Returns zero output when values are outside table boundary */
    if (rI < 0 || rI > (S->numCols - 2) || cI < 0 || cI > (S->numRows - 2))
    {
      return (0);
    }

    /* 20 bits for the fractional part */
    /* shift left xfract by 11 to keep 1.31 format */
    xfract = (X & 0x000FFFFF) << 11U;

    /* Read two nearest output values from the index */
    x1 = pYData[(rI) + (int32_t)nCols * (cI)    ];
    x2 = pYData[(rI) + (int32_t)nCols * (cI) + 1];

    /* 20 bits for the fractional part */
    /* shift left yfract by 11 to keep 1.31 format */
    yfract = (Y & 0x000FFFFF) << 11U;

    /* Read two nearest output values from the index */
    y1 = pYData[(rI) + (int32_t)nCols * (cI + 1)    ];
    y2 = pYData[(rI) + (int32_t)nCols * (cI + 1) + 1];

    /* Calculation of x1 * (1-xfract ) * (1-yfract) and acc is in 3.29(q29) format */
    out = ((q31_t) (((q63_t) x1  * (0x7FFFFFFF - xfract)) >> 32));
    acc = ((q31_t) (((q63_t) out * (0x7FFFFFFF - yfract)) >> 32));

    /* x2 * (xfract) * (1-yfract)  in 3.29(q29) and adding to acc */
    out = ((q31_t) ((q63_t) x2 * (0x7FFFFFFF - yfract) >> 32));
    acc += ((q31_t) ((q63_t) out * (xfract) >> 32));

    /* y1 * (1 - xfract) * (yfract)  in 3.29(q29) and adding to acc */
    out = ((q31_t) ((q63_t) y1 * (0x7FFFFFFF - xfract) >> 32));
    acc += ((q31_t) ((q63_t) out * (yfract) >> 32));

    /* y2 * (xfract) * (yfract)  in 3.29(q29) and adding to acc */
    out = ((q31_t) ((q63_t) y2 * (xfract) >> 32));
    acc += ((q31_t) ((q63_t) out * (yfract) >> 32));

    /* Convert acc to 1.31(q31) format */
    return ((q31_t)(acc << 2));
  }


  /**
  * @brief  Q15 bilinear interpolation.
  * @param[in,out] S  points to an instance of the interpolation structure.
  * @param[in]     X  interpolation coordinate in 12.20 format.
  * @param[in]     Y  interpolation coordinate in 12.20 format.
  * @return out interpolated value.
  */
  __STATIC_FORCEINLINE q15_t arm_bilinear_interp_q15(
  arm_bilinear_interp_instance_q15 * S,
  q31_t X,
  q31_t Y)
  {
    q63_t acc = 0;                               /* output */
    q31_t out;                                   /* Temporary output */
    q15_t x1, x2, y1, y2;                        /* Nearest output values */
    q31_t xfract, yfract;                        /* X, Y fractional parts */
    int32_t rI, cI;                              /* Row and column indices */
    q15_t *pYData = S->pData;                    /* pointer to output table values */
    uint32_t nCols = S->numCols;                 /* num of rows */

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    rI = ((X & (q31_t)0xFFF00000) >> 20);

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    cI = ((Y & (q31_t)0xFFF00000) >> 20);

    /* Care taken for table outside boundary */
    /* Returns zero output when values are outside table boundary */
    if (rI < 0 || rI > (S->numCols - 2) || cI < 0 || cI > (S->numRows - 2))
    {
      return (0);
    }

    /* 20 bits for the fractional part */
    /* xfract should be in 12.20 format */
    xfract = (X & 0x000FFFFF);

    /* Read two nearest output values from the index */
    x1 = pYData[((uint32_t)rI) + nCols * ((uint32_t)cI)    ];
    x2 = pYData[((uint32_t)rI) + nCols * ((uint32_t)cI) + 1];

    /* 20 bits for the fractional part */
    /* yfract should be in 12.20 format */
    yfract = (Y & 0x000FFFFF);

    /* Read two nearest output values from the index */
    y1 = pYData[((uint32_t)rI) + nCols * ((uint32_t)cI + 1)    ];
    y2 = pYData[((uint32_t)rI) + nCols * ((uint32_t)cI + 1) + 1];

    /* Calculation of x1 * (1-xfract ) * (1-yfract) and acc is in 13.51 format */

    /* x1 is in 1.15(q15), xfract in 12.20 format and out is in 13.35 format */
    /* convert 13.35 to 13.31 by right shifting  and out is in 1.31 */
    out = (q31_t) (((q63_t) x1 * (0x0FFFFF - xfract)) >> 4U);
    acc = ((q63_t) out * (0x0FFFFF - yfract));

    /* x2 * (xfract) * (1-yfract)  in 1.51 and adding to acc */
    out = (q31_t) (((q63_t) x2 * (0x0FFFFF - yfract)) >> 4U);
    acc += ((q63_t) out * (xfract));

    /* y1 * (1 - xfract) * (yfract)  in 1.51 and adding to acc */
    out = (q31_t) (((q63_t) y1 * (0x0FFFFF - xfract)) >> 4U);
    acc += ((q63_t) out * (yfract));

    /* y2 * (xfract) * (yfract)  in 1.51 and adding to acc */
    out = (q31_t) (((q63_t) y2 * (xfract)) >> 4U);
    acc += ((q63_t) out * (yfract));

    /* acc is in 13.51 format and down shift acc by 36 times */
    /* Convert out to 1.15 format */
    return ((q15_t)(acc >> 36));
  }


  /**
  * @brief  Q7 bilinear interpolation.
  * @param[in,out] S  points to an instance of the interpolation structure.
  * @param[in]     X  interpolation coordinate in 12.20 format.
  * @param[in]     Y  interpolation coordinate in 12.20 format.
  * @return out interpolated value.
  */
  __STATIC_FORCEINLINE q7_t arm_bilinear_interp_q7(
  arm_bilinear_interp_instance_q7 * S,
  q31_t X,
  q31_t Y)
  {
    q63_t acc = 0;                               /* output */
    q31_t out;                                   /* Temporary output */
    q31_t xfract, yfract;                        /* X, Y fractional parts */
    q7_t x1, x2, y1, y2;                         /* Nearest output values */
    int32_t rI, cI;                              /* Row and column indices */
    q7_t *pYData = S->pData;                     /* pointer to output table values */
    uint32_t nCols = S->numCols;                 /* num of rows */

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    rI = ((X & (q31_t)0xFFF00000) >> 20);

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    cI = ((Y & (q31_t)0xFFF00000) >> 20);

    /* Care taken for table outside boundary */
    /* Returns zero output when values are outside table boundary */
    if (rI < 0 || rI > (S->numCols - 2) || cI < 0 || cI > (S->numRows - 2))
    {
      return (0);
    }

    /* 20 bits for the fractional part */
    /* xfract should be in 12.20 format */
    xfract = (X & (q31_t)0x000FFFFF);

    /* Read two nearest output values from the index */
    x1 = pYData[((uint32_t)rI) + nCols * ((uint32_t)cI)    ];
    x2 = pYData[((uint32_t)rI) + nCols * ((uint32_t)cI) + 1];

    /* 20 bits for the fractional part */
    /* yfract should be in 12.20 format */
    yfract = (Y & (q31_t)0x000FFFFF);

    /* Read two nearest output values from the index */
    y1 = pYData[((uint32_t)rI) + nCols * ((uint32_t)cI + 1)    ];
    y2 = pYData[((uint32_t)rI) + nCols * ((uint32_t)cI + 1) + 1];

    /* Calculation of x1 * (1-xfract ) * (1-yfract) and acc is in 16.47 format */
    out = ((x1 * (0xFFFFF - xfract)));
    acc = (((q63_t) out * (0xFFFFF - yfract)));

    /* x2 * (xfract) * (1-yfract)  in 2.22 and adding to acc */
    out = ((x2 * (0xFFFFF - yfract)));
    acc += (((q63_t) out * (xfract)));

    /* y1 * (1 - xfract) * (yfract)  in 2.22 and adding to acc */
    out = ((y1 * (0xFFFFF - xfract)));
    acc += (((q63_t) out * (yfract)));

    /* y2 * (xfract) * (yfract)  in 2.22 and adding to acc */
    out = ((y2 * (yfract)));
    acc += (((q63_t) out * (xfract)));

    /* acc in 16.47 format and down shift by 40 to convert to 1.7 format */
    return ((q7_t)(acc >> 40));
  }

  /**
   * @} end of BilinearInterpolate group
   */


/* SMMLAR */
#define multAcc_32x32_keep32_R(a, x, y) \
    a = (q31_t) (((((q63_t) a) << 32) + ((q63_t) x * y) + 0x80000000LL ) >> 32)

/* SMMLSR */
#define multSub_32x32_keep32_R(a, x, y) \
    a = (q31_t) (((((q63_t) a) << 32) - ((q63_t) x * y) + 0x80000000LL ) >> 32)

/* SMMULR */
#define mult_32x32_keep32_R(a, x, y) \
    a = (q31_t) (((q63_t) x * y + 0x80000000LL ) >> 32)

/* SMMLA */
#define multAcc_32x32_keep32(a, x, y) \
    a += (q31_t) (((q63_t) x * y) >> 32)

/* SMMLS */
#define multSub_32x32_keep32(a, x, y) \
    a -= (q31_t) (((q63_t) x * y) >> 32)

/* SMMUL */
#define mult_32x32_keep32(a, x, y) \
    a = (q31_t) (((q63_t) x * y ) >> 32)


#if   defined ( __CC_ARM )
  /* Enter low optimization region - place directly above function definition */
  #if defined( __ARM_ARCH_7EM__ )
    #define LOW_OPTIMIZATION_ENTER \
       _Pragma ("push")         \
       _Pragma ("O1")
  #else
    #define LOW_OPTIMIZATION_ENTER
  #endif

  /* Exit low optimization region - place directly after end of function definition */
  #if defined ( __ARM_ARCH_7EM__ )
    #define LOW_OPTIMIZATION_EXIT \
       _Pragma ("pop")
  #else
    #define LOW_OPTIMIZATION_EXIT
  #endif

  /* Enter low optimization region - place directly above function definition */
  #define IAR_ONLY_LOW_OPTIMIZATION_ENTER

  /* Exit low optimization region - place directly after end of function definition */
  #define IAR_ONLY_LOW_OPTIMIZATION_EXIT

#elif defined (__ARMCC_VERSION ) && ( __ARMCC_VERSION >= 6010050 )
  #define LOW_OPTIMIZATION_ENTER
  #define LOW_OPTIMIZATION_EXIT
  #define IAR_ONLY_LOW_OPTIMIZATION_ENTER
  #define IAR_ONLY_LOW_OPTIMIZATION_EXIT

#elif defined ( __GNUC__ )
  #define LOW_OPTIMIZATION_ENTER \
       __attribute__(( optimize("-O1") ))
  #define LOW_OPTIMIZATION_EXIT
  #define IAR_ONLY_LOW_OPTIMIZATION_ENTER
  #define IAR_ONLY_LOW_OPTIMIZATION_EXIT

#elif defined ( __ICCARM__ )
  /* Enter low optimization region - place directly above function definition */
  #if defined ( __ARM_ARCH_7EM__ )
    #define LOW_OPTIMIZATION_ENTER \
       _Pragma ("optimize=low")
  #else
    #define LOW_OPTIMIZATION_ENTER
  #endif

  /* Exit low optimization region - place directly after end of function definition */
  #define LOW_OPTIMIZATION_EXIT

  /* Enter low optimization region - place directly above function definition */
  #if defined ( __ARM_ARCH_7EM__ )
    #define IAR_ONLY_LOW_OPTIMIZATION_ENTER \
       _Pragma ("optimize=low")
  #else
    #define IAR_ONLY_LOW_OPTIMIZATION_ENTER
  #endif

  /* Exit low optimization region - place directly after end of function definition */
  #define IAR_ONLY_LOW_OPTIMIZATION_EXIT

#elif defined ( __TI_ARM__ )
  #define LOW_OPTIMIZATION_ENTER
  #define LOW_OPTIMIZATION_EXIT
  #define IAR_ONLY_LOW_OPTIMIZATION_ENTER
  #define IAR_ONLY_LOW_OPTIMIZATION_EXIT

#elif defined ( __CSMC__ )
  #define LOW_OPTIMIZATION_ENTER
  #define LOW_OPTIMIZATION_EXIT
  #define IAR_ONLY_LOW_OPTIMIZATION_ENTER
  #define IAR_ONLY_LOW_OPTIMIZATION_EXIT

#elif defined ( __TASKING__ )
  #define LOW_OPTIMIZATION_ENTER
  #define LOW_OPTIMIZATION_EXIT
  #define IAR_ONLY_LOW_OPTIMIZATION_ENTER
  #define IAR_ONLY_LOW_OPTIMIZATION_EXIT
       
#elif defined ( _MSC_VER ) || defined(__GNUC_PYTHON__)
      #define LOW_OPTIMIZATION_ENTER
      #define LOW_OPTIMIZATION_EXIT
      #define IAR_ONLY_LOW_OPTIMIZATION_ENTER 
      #define IAR_ONLY_LOW_OPTIMIZATION_EXIT
#endif



/* Compiler specific diagnostic adjustment */
#if   defined ( __CC_ARM )

#elif defined ( __ARMCC_VERSION ) && ( __ARMCC_VERSION >= 6010050 )

#elif defined ( __GNUC__ )
#pragma GCC diagnostic pop

#elif defined ( __ICCARM__ )

#elif defined ( __TI_ARM__ )

#elif defined ( __CSMC__ )

#elif defined ( __TASKING__ )

#elif defined ( _MSC_VER )

#else
  #error Unknown compiler
#endif

#ifdef   __cplusplus
}
#endif


#endif /* _ARM_MATH_H */

/**
 *
 * End of file.
 */
