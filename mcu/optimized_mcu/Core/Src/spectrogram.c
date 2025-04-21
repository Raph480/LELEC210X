/*
 * spectrogram.c
 *
 *  Created on: Jun 4, 2021
 *      Author: math
 */

#include <stdio.h>
#include "spectrogram.h"
#include "spectrogram_tables.h"
#include "config.h"
#include "utils.h"
#include "arm_absmax_q15.h"
#include "system_stm32l4xx.h"

q15_t buf    [  SAMPLES_PER_MELVEC  ]; // Windowed samples
q15_t buf_fft[2*SAMPLES_PER_MELVEC  ]; // Double size (real|imag) buffer needed for arm_rfft_q15
q15_t buf_tmp[  SAMPLES_PER_MELVEC/2]; // Intermediate buffer for arm_mat_mult_fast_q15

// Convert 12-bit DC ADC samples to Q1.15 fixed point signal and remove DC component
void Spectrogram_Format(q15_t *buf)
{
	// STEP 0.1 : Remove DC Component
	//            --> Pointwise substract
	//            Complexity: O(N)
	//            Number of cycles: <TODO>

	// The output of the ADC is stored in an unsigned 12-bit format, so buf[i] is in [0 , 2**12 - 1]

	// Since we use a signed representation, we should now center the value around zero, we can do this by substracting 2**11.
	// Now the value of buf[i] is in [-2**11 , 2**11 - 1]

	// A constant value of 2**11 is substracted from each sample to center the signal around zero.
	for(uint16_t i=0; i < SAMPLES_PER_MELVEC; i++) { // Remove DC component
		buf[i] -= (1 << 11);
	}

	// STEP 0.2 : Increase fixed-point scale
	//            --> Pointwise shift
	//            Complexity: O(N)
	//            Number of cycles: <TODO>

	// In order to better use the scale of the signed 16-bit format (1 bit of sign and 15 integer bits), we can multiply by 2**(16-12) = 2**4
	// That way, the value of buf[i] is in [-2**15 , 2**15 - 1]

	// Since ADC values are 12-bit unsigned integers, they are left-shifted by 4 to better use the 16-bit signed fixed-point format.
	arm_shift_q15(buf, 4, buf, SAMPLES_PER_MELVEC);

}

// Compute MEL-vector
void Spectrogram_Compute(q15_t *samples, q15_t *melvec) {

	// STEP 1  : Windowing of input samples
	//           --> Pointwise product
	//           Complexity: O(N)
	//           Number of cycles: <TODO>

    start_cycle_count();
    arm_mult_q15(samples, hamming_window, buf, SAMPLES_PER_MELVEC);
    stop_cycle_count("Step 1");

	// STEP 2  : Discrete Fourier Transform
	//           --> In-place Fast Fourier Transform (FFT) on a real signal
	//           --> For our spectrogram, we only keep only positive frequencies (symmetry) in the next operations.
	//           Complexity: O(Nlog(N))
	//           Number of cycles: <TODO>

    start_cycle_count();
    arm_rfft_instance_q15 rfft_inst;
    arm_rfft_init_q15(&rfft_inst, SAMPLES_PER_MELVEC, 0, 1);
    arm_rfft_q15(&rfft_inst, buf, buf_fft);
    stop_cycle_count("Step 2");

    // STEP 3  : Compute the complex magnitude of the FFT
	//           Because the FFT can output a great proportion of very small values,
	//           we should rescale all values by their maximum to avoid loss of precision when computing the complex magnitude
	//           In this implementation, we use integer division and multiplication to rescale values, which are very costly.

	// STEP 3.1: Find the extremum value (maximum of absolute values)
	//           Complexity: O(N)
	//           Number of cycles: <TODO>

    start_cycle_count();
    q15_t vmax;
    uint32_t pIndex=0;
    arm_absmax_q15(buf_fft, SAMPLES_PER_MELVEC, &vmax, &pIndex);
    stop_cycle_count("Step 3.1");

	// STEP 3.2: Normalize the vector - Dynamic range increase
	//           Complexity: O(N)
	//           Number of cycles: <TODO>

    start_cycle_count();
    for (int i=0; i < SAMPLES_PER_MELVEC; i++) {
        buf[i] = (q15_t) (((q31_t) buf_fft[i] << 15) /((q31_t)vmax));
    }
    stop_cycle_count("Step 3.2");

	// STEP 3.3: Compute the complex magnitude
	//           --> The output buffer is now two times smaller because (real|imag) --> (mag)
	//           Complexity: O(N)
	//           Number of cycles: <TODO>

    start_cycle_count();
    arm_cmplx_mag_q15(buf, buf, SAMPLES_PER_MELVEC/2);
    stop_cycle_count("Step 3.3");

	// STEP 3.4: Denormalize the vector
	//           Complexity: O(N)
	//           Number of cycles: <TODO>

    start_cycle_count();
    for (int i=0; i < SAMPLES_PER_MELVEC/2; i++) {
        buf[i] = (q15_t) ((((q31_t) buf[i]) * ((q31_t) vmax) ) >> 15 );
    }
    stop_cycle_count("Step 3.4");

    // STEP 4: Apply Mel transform using optimized sparse multiplication

    start_cycle_count();  // Start counting cycles
    for (int i = 0; i < MELVEC_LENGTH; i++) {
        int16_t *hz2mel_row = &hz2mel_mat[i*SAMPLES_PER_MELVEC/2 +start_index[i]];  // Correct start in sparse matrix
        int16_t *fftmag_row = &buf[start_index[i]];  // Corresponding FFT values
        int count = nonzero_count[i];  // Number of nonzero elements


        q63_t tmp_result;
        // Compute dot product
        arm_dot_prod_q15(hz2mel_row, fftmag_row, count, &tmp_result);
        melvec[i] = (int16_t)(tmp_result >> 15);  // Convert back to q15 format
    }

    stop_cycle_count("Step 4");
    /*DEBUG_PRINT("MelVec (Original Version): \r\n");
	for (int i = 0; i < MELVEC_LENGTH; i++) {
		DEBUG_PRINT("%d ", melvec[i]);
	}
	DEBUG_PRINT("\r\n");
	*/
}



