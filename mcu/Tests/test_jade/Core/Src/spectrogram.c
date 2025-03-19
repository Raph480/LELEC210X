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
	// STEP 0.1 : Increase fixed-point scale
	//            --> Pointwise shift
	//            Complexity: O(N)
	//            Number of cycles: <TODO>

	// The output of the ADC is stored in an unsigned 12-bit format, so buf[i] is in [0 , 2**12 - 1]
	// In order to better use the scale of the signed 16-bit format (1 bit of sign and 15 integer bits), we can multiply by 2**(15-12) = 2**3
	// That way, the value of buf[i] is in [0 , 2**15 - 1]

	// /!\ When multiplying/dividing by a power 2, always prefer shifting left/right instead, ARM instructions to do so are more efficient.
	// Here we should shift left by 3.

	// Since ADC values are 12-bit unsigned integers, they are left-shifted by 3 to better use the 16-bit signed fixed-point format.
	arm_shift_q15(buf, 3, buf, SAMPLES_PER_MELVEC);

	// STEP 0.2 : Remove DC Component
	//            --> Pointwise substract
	//            Complexity: O(N)
	//            Number of cycles: <TODO>

	// Since we use a signed representation, we should now center the value around zero, we can do this by substracting 2**14.
	// Now the value of buf[i] is in [-2**14 , 2**14 - 1]

	// A constant value of 2**14 is substracted from each sample to center the signal around zero.
	for(uint16_t i=0; i < SAMPLES_PER_MELVEC; i++) { // Remove DC component
		buf[i] -= (1 << 14);
	}
}


// Compute spectrogram of samples and transform into MEL vectors.
// Compute spectrogram of samples and transform into MEL vectors.
/*void Spectrogram_Compute(q15_t *samples, q15_t *melvec)
{
    // STEP 1: Windowing of input samples
    start_cycle_count();  // Start counting cycles
    arm_mult_q15(samples, hamming_window, buf, SAMPLES_PER_MELVEC);
    uint32_t cycles_elapsed = stop_cycle_count("Step 1");
    //uint32_t elapsed_time_ms = (float)cycles_elapsed / (float)SystemCoreClock * 1000.0f;
    //float cycle_time_ns = (float)(elapsed_time_ms * 1000000) / (float)cycles_elapsed;  // Time per cycle in nanoseconds
    //DEBUG_PRINT("Cycle elapsed Step 1: %.6ld \r\n", cycles_elapsed);
    //DEBUG_PRINT("Time per cycle (Step 1): %.6f ns\r\n", cycle_time_ns);
    //DEBUG_PRINT("Spectrogram Compute Step 1: %.2f ms\r\n", elapsed_time_ms);

    // STEP 2: Discrete Fourier Transform (FFT)
    start_cycle_count();  // Start counting cycles
    arm_rfft_instance_q15 rfft_inst;
    arm_rfft_init_q15(&rfft_inst, SAMPLES_PER_MELVEC, 0, 1);
    arm_rfft_q15(&rfft_inst, buf, buf_fft);
    cycles_elapsed = stop_cycle_count("Step 2");
    //elapsed_time_ms = (float)cycles_elapsed / (float)SystemCoreClock * 1000.0f;
    //cycle_time_ns = (float)(elapsed_time_ms * 1000000) / (float)cycles_elapsed;  // Time per cycle in nanoseconds
    //DEBUG_PRINT("Cycle elapsed Step 2: %.6ld \r\n", cycles_elapsed);
    //DEBUG_PRINT("Time per cycle (Step 2): %.6f ns\r\n", cycle_time_ns);
    //DEBUG_PRINT("Spectrogram Compute Step 2: %.2f ms\r\n", elapsed_time_ms);

    // STEP 3.1: Find maximum value in FFT output
    start_cycle_count();  // Start counting cycles
    q15_t vmax;
    uint32_t pIndex = 0;
    arm_absmax_q15(buf_fft, SAMPLES_PER_MELVEC, &vmax, &pIndex);

    // STEP 3.2: Normalize the FFT output by dividing by vmax
    for (int i = 0; i < SAMPLES_PER_MELVEC; i++) {
        buf[i] = (q15_t) (((q31_t) buf_fft[i] << 15) / ((q31_t)vmax));
    }

    // STEP 3.3: Compute complex magnitude
    arm_cmplx_mag_q15(buf, buf, SAMPLES_PER_MELVEC / 2);

    // STEP 3.4: Denormalize the vector
    for (int i = 0; i < SAMPLES_PER_MELVEC / 2; i++) {
        buf[i] = (q15_t) ((((q31_t) buf[i]) * ((q31_t) vmax)) >> 15);
    }

    cycles_elapsed = stop_cycle_count("Step 3");
    //elapsed_time_ms = (float)cycles_elapsed / (float)SystemCoreClock * 1000.0f;
    //cycle_time_ns = (float)(elapsed_time_ms * 1000000) / (float)cycles_elapsed;  // Time per cycle in nanoseconds
    //DEBUG_PRINT("Cycle elapsed Step 3: %.6ld \r\n", cycles_elapsed);
    //DEBUG_PRINT("Time per cycle (Step 3): %.6f ns\r\n", cycle_time_ns);
    //DEBUG_PRINT("Spectrogram Compute Step 3: %.2f ms\r\n", elapsed_time_ms);

    // STEP 4: Apply Mel transform using matrix multiplication
    start_cycle_count();  // Start counting cycles
    arm_matrix_instance_q15 hz2mel_inst, fftmag_inst, melvec_inst;
    arm_mat_init_q15(&hz2mel_inst, MELVEC_LENGTH, SAMPLES_PER_MELVEC / 2, hz2mel_mat);
    arm_mat_init_q15(&fftmag_inst, SAMPLES_PER_MELVEC / 2, 1, buf);
    arm_mat_init_q15(&melvec_inst, MELVEC_LENGTH, 1, melvec);
    arm_mat_mult_fast_q15(&hz2mel_inst, &fftmag_inst, &melvec_inst, buf_tmp);
    cycles_elapsed = stop_cycle_count("Step 4");
    //elapsed_time_ms = (float)cycles_elapsed / (float)SystemCoreClock * 1000.0f;
    //cycle_time_ns = (float)(elapsed_time_ms * 1000000) / (float)cycles_elapsed;  // Time per cycle in nanoseconds
    //DEBUG_PRINT("Cycle elapsed Step 4: %.6ld \r\n", cycles_elapsed);
    //DEBUG_PRINT("Time per cycle (Step 4): %.6f ns\r\n", cycle_time_ns);
    //DEBUG_PRINT("Spectrogram Compute Step 4: %.2f ms\r\n", elapsed_time_ms);
}
*/

/*

void Spectrogram_Compute(q15_t *samples, q15_t *melvec) {

    // Step 1: Windowing of samples
    start_cycle_count();
    arm_mult_q15(samples, hamming_window, buf, SAMPLES_PER_MELVEC);
    uint32_t cycles_elapsed = stop_cycle_count("Step 1");
    //uint32_t elapsed_time_ms = (float)cycles_elapsed / SystemCoreClock * 1000.0f;
    //float cycle_time_ns = (float)(elapsed_time_ms * 1000000) / (float)cycles_elapsed;  // Time per cycle in nanoseconds
    //DEBUG_PRINT("Cycle elapsed Step 1: %.6ld \r\n", cycles_elapsed);
    //DEBUG_PRINT("Time per cycle (Step 1): %.6f ns\r\n", cycle_time_ns);

    // Step 2: FFT Computation
    start_cycle_count();
    arm_rfft_instance_q15 rfft_inst;
    arm_rfft_init_q15(&rfft_inst, SAMPLES_PER_MELVEC, 0, 1);
    arm_rfft_q15(&rfft_inst, buf, buf_fft);
    cycles_elapsed = stop_cycle_count("Step 2");
    //elapsed_time_ms = (float)cycles_elapsed / SystemCoreClock * 1000.0f;
    //cycle_time_ns = (float)(elapsed_time_ms * 1000000) / (float)cycles_elapsed;  // Time per cycle in nanoseconds
    //DEBUG_PRINT("Cycle elapsed Step 2: %.6ld \r\n", cycles_elapsed);
    //DEBUG_PRINT("Time per cycle (Step 2): %.6f ns\r\n", cycle_time_ns);

    // Step 3.1: Find maximum value in FFT output
    start_cycle_count();
    q15_t vmax;
    uint32_t pIndex = 0;
    arm_absmax_q15(buf_fft, SAMPLES_PER_MELVEC, &vmax, &pIndex);

    // Step 3.2: Normalize FFT output by dividing by vmax
    for (int i = 0; i < SAMPLES_PER_MELVEC; i++) {
        buf[i] = (q15_t) (((q31_t) buf_fft[i] << 15) / ((q31_t)vmax));
    }

    // Step 3.3: Compute complex magnitude
    arm_cmplx_mag_q15(buf, buf, SAMPLES_PER_MELVEC / 2);

    // Step 3.4: Denormalize magnitude
    for (int i = 0; i < SAMPLES_PER_MELVEC / 2; i++) {
        buf[i] = (q15_t) ((((q31_t) buf[i]) * ((q31_t) vmax)) >> 15);
    }

    cycles_elapsed = stop_cycle_count("Step 3");
    //elapsed_time_ms = (float)cycles_elapsed / SystemCoreClock * 1000.0f;
    //cycle_time_ns = (float)(elapsed_time_ms * 1000000) / (float)cycles_elapsed;  // Time per cycle in nanoseconds
    //DEBUG_PRINT("Cycle elapsed Step 3: %.6ld \r\n", cycles_elapsed);
    //DEBUG_PRINT("Time per cycle (Step 3): %.6f ns\r\n", cycle_time_ns);

    // Step 4: Apply Mel transform using CSR
    start_cycle_count();
    for (int i = 0; i < MELVEC_LENGTH; i++) {
        q31_t sum = 0;
        for (int j = hz2mel_row_ptr[i]; j < hz2mel_row_ptr[i + 1]; j++) {
            sum = __SMLAD(hz2mel_values[j], buf[hz2mel_col_indices[j]], sum);
        }
        melvec[i] = (q15_t)(__SSAT(sum >> 15, 16));
    }
    cycles_elapsed = stop_cycle_count("Step 4 CSR");
    //elapsed_time_ms = (float)cycles_elapsed / SystemCoreClock * 1000.0f;
    //cycle_time_ns = (float)(elapsed_time_ms * 1000000) / (float)cycles_elapsed;  // Time per cycle in nanoseconds
    //DEBUG_PRINT("Cycle elapsed Step 4: %.6ld \r\n", cycles_elapsed);
    //DEBUG_PRINT("Time per cycle (Step 4): %.6f ns\r\n", cycle_time_ns);
}
*/


/*void Spectrogram_Compute(q15_t *samples, q15_t *melvec) {


    start_cycle_count();
    arm_mult_q15(samples, hamming_window, buf, SAMPLES_PER_MELVEC);
    uint32_t cycles_elapsed = stop_cycle_count("Step 1");

    start_cycle_count();
    arm_rfft_instance_q15 rfft_inst;
    arm_rfft_init_q15(&rfft_inst, SAMPLES_PER_MELVEC, 0, 1);
    arm_rfft_q15(&rfft_inst, buf, buf_fft);
    cycles_elapsed = stop_cycle_count("Step 2");

    start_cycle_count();
    q15_t vmax;
    uint32_t pIndex=0;
    arm_absmax_q15(buf_fft, SAMPLES_PER_MELVEC, &vmax, &pIndex);
    for (int i=0; i < SAMPLES_PER_MELVEC; i++) {
        buf[i] = (q15_t) (((q31_t) buf_fft[i] << 15) /((q31_t)vmax));
    }
    arm_cmplx_mag_q15(buf, buf, SAMPLES_PER_MELVEC/2);
    for (int i=0; i < SAMPLES_PER_MELVEC/2; i++) {
        buf[i] = (q15_t) ((((q31_t) buf[i]) * ((q31_t) vmax) ) >> 15 );
    }
    cycles_elapsed = stop_cycle_count("Step 3");

    // Step 4: Apply Mel transform using CSR without SIMD
    start_cycle_count();
    for (int i = 0; i < MELVEC_LENGTH; i++) {
        q31_t sum = 0;
        for (int j = hz2mel_row_ptr[i]; j < hz2mel_row_ptr[i + 1]; j += 2) {
            sum += (q31_t) hz2mel_values[j] * (q31_t) buf[hz2mel_col_indices[j]];
            if (j + 1 < hz2mel_row_ptr[i + 1]) {
                sum += (q31_t) hz2mel_values[j + 1] * (q31_t) buf[hz2mel_col_indices[j + 1]];
            }
        }
        melvec[i] = (q15_t) __SSAT(sum >> 15, 16);
    }
    cycles_elapsed = stop_cycle_count("Step 4 CSR");
}
*/

/*
void Spectrogram_Compute(q15_t *samples, q15_t *melvec) {


    start_cycle_count();
    arm_mult_q15(samples, hamming_window, buf, SAMPLES_PER_MELVEC);
    uint32_t cycles_elapsed = stop_cycle_count("Step 1");

    start_cycle_count();
    arm_rfft_instance_q15 rfft_inst;
    arm_rfft_init_q15(&rfft_inst, SAMPLES_PER_MELVEC, 0, 1);
    arm_rfft_q15(&rfft_inst, buf, buf_fft);
    cycles_elapsed = stop_cycle_count("Step 2");

    start_cycle_count();
    q15_t vmax;
    uint32_t pIndex=0;
    arm_absmax_q15(buf_fft, SAMPLES_PER_MELVEC, &vmax, &pIndex);
    for (int i=0; i < SAMPLES_PER_MELVEC; i++) {
        buf[i] = (q15_t) (((q31_t) buf_fft[i] << 15) /((q31_t)vmax));
    }
    arm_cmplx_mag_q15(buf, buf, SAMPLES_PER_MELVEC/2);
    for (int i=0; i < SAMPLES_PER_MELVEC/2; i++) {
        buf[i] = (q15_t) ((((q31_t) buf[i]) * ((q31_t) vmax) ) >> 15 );
    }
    cycles_elapsed = stop_cycle_count("Step 3");

    // Step 4: Apply Mel transform using CSR without SIMD
    start_cycle_count();
    for (int i = 0; i < MELVEC_LENGTH; i++) {
        q31_t sum = 0;
        int j;
        for (j = hz2mel_row_ptr[i]; j + 2 < hz2mel_row_ptr[i + 1]; j += 2) {
            sum = __SMLAD(*((q31_t*)&hz2mel_values[j]), *((q31_t*)&buf[hz2mel_col_indices[j]]), sum);
        }
        // Handle any remaining elements
        for (; j < hz2mel_row_ptr[i + 1]; j++) {
            sum += (q31_t) hz2mel_values[j] * (q31_t) buf[hz2mel_col_indices[j]];
        }
        melvec[i] = (q15_t) __SSAT(sum >> 15, 16);
    }
    cycles_elapsed = stop_cycle_count("Step 4 SIMD CSR");


}
*/

//Version avec uniquement CSR:
/*
void Spectrogram_Compute(q15_t *samples, q15_t *melvec)
{
    // STEP 1: Windowing of input samples
    start_cycle_count();  // Start counting cycles
    arm_mult_q15(samples, hamming_window, buf, SAMPLES_PER_MELVEC);
    uint32_t cycles_elapsed = stop_cycle_count("Step 1");
    //uint32_t elapsed_time_ms = (float)cycles_elapsed / (float)SystemCoreClock * 1000.0f;
    //float cycle_time_ns = (float)(elapsed_time_ms * 1000000) / (float)cycles_elapsed;  // Time per cycle in nanoseconds
    //DEBUG_PRINT("Cycle elapsed Step 1: %.6ld \r\n", cycles_elapsed);
    //DEBUG_PRINT("Time per cycle (Step 1): %.6f ns\r\n", cycle_time_ns);
    //DEBUG_PRINT("Spectrogram Compute Step 1: %.2f ms\r\n", elapsed_time_ms);

    // STEP 2: Discrete Fourier Transform (FFT)
    start_cycle_count();  // Start counting cycles
    arm_rfft_instance_q15 rfft_inst;
    arm_rfft_init_q15(&rfft_inst, SAMPLES_PER_MELVEC, 0, 1);
    arm_rfft_q15(&rfft_inst, buf, buf_fft);
    cycles_elapsed = stop_cycle_count("Step 2");
    //elapsed_time_ms = (float)cycles_elapsed / (float)SystemCoreClock * 1000.0f;
    //cycle_time_ns = (float)(elapsed_time_ms * 1000000) / (float)cycles_elapsed;  // Time per cycle in nanoseconds
    //DEBUG_PRINT("Cycle elapsed Step 2: %.6ld \r\n", cycles_elapsed);
    //DEBUG_PRINT("Time per cycle (Step 2): %.6f ns\r\n", cycle_time_ns);
    //DEBUG_PRINT("Spectrogram Compute Step 2: %.2f ms\r\n", elapsed_time_ms);

    // STEP 3.1: Find maximum value in FFT output
    start_cycle_count();  // Start counting cycles
    q15_t vmax;
    uint32_t pIndex = 0;
    arm_absmax_q15(buf_fft, SAMPLES_PER_MELVEC, &vmax, &pIndex);

    // STEP 3.2: Normalize the FFT output by dividing by vmax
    for (int i = 0; i < SAMPLES_PER_MELVEC; i++) {
        buf[i] = (q15_t) (((q31_t) buf_fft[i] << 15) / ((q31_t)vmax));
    }

    // STEP 3.3: Compute complex magnitude
    arm_cmplx_mag_q15(buf, buf, SAMPLES_PER_MELVEC / 2);

    // STEP 3.4: Denormalize the vector
    for (int i = 0; i < SAMPLES_PER_MELVEC / 2; i++) {
        buf[i] = (q15_t) ((((q31_t) buf[i]) * ((q31_t) vmax)) >> 15);
    }

    cycles_elapsed = stop_cycle_count("Step 3");
    //elapsed_time_ms = (float)cycles_elapsed / (float)SystemCoreClock * 1000.0f;
    //cycle_time_ns = (float)(elapsed_time_ms * 1000000) / (float)cycles_elapsed;  // Time per cycle in nanoseconds
    //DEBUG_PRINT("Cycle elapsed Step 3: %.6ld \r\n", cycles_elapsed);
    //DEBUG_PRINT("Time per cycle (Step 3): %.6f ns\r\n", cycle_time_ns);
    //DEBUG_PRINT("Spectrogram Compute Step 3: %.2f ms\r\n", elapsed_time_ms);

    // STEP 4: Apply Mel transform using matrix multiplication
    start_cycle_count();  // Start counting cycles

    // Initialize output vector
    for (int row = 0; row < MELVEC_LENGTH; row++) {
        q31_t acc = 0;  // Accumulate in higher precision
        for (int j = hz2mel_row_ptr[row]; j < hz2mel_row_ptr[row + 1]; j += 2) {
            int col1 = hz2mel_col_indices[j];
            int col2 = hz2mel_col_indices[j + 1];
            acc += ((q31_t) hz2mel_values[j] * buf[col1]) >> 15;
            acc += ((q31_t) hz2mel_values[j + 1] * buf[col2]) >> 15;
        }
        melvec[row] = (q15_t) acc;
    }


    cycles_elapsed = stop_cycle_count("Step 4");
}
*/


//Double unrolling loop 12000 cycles

void Spectrogram_Compute(q15_t *samples, q15_t *melvec) {


    start_cycle_count();
    arm_mult_q15(samples, hamming_window, buf, SAMPLES_PER_MELVEC);
    uint32_t cycles_elapsed = stop_cycle_count("Step 1");

    start_cycle_count();
    arm_rfft_instance_q15 rfft_inst;
    arm_rfft_init_q15(&rfft_inst, SAMPLES_PER_MELVEC, 0, 1);
    arm_rfft_q15(&rfft_inst, buf, buf_fft);
    cycles_elapsed = stop_cycle_count("Step 2");

    start_cycle_count();
    q15_t vmax;
    uint32_t pIndex=0;
    arm_absmax_q15(buf_fft, SAMPLES_PER_MELVEC, &vmax, &pIndex);
    for (int i=0; i < SAMPLES_PER_MELVEC; i++) {
        buf[i] = (q15_t) (((q31_t) buf_fft[i] << 15) /((q31_t)vmax));
    }
    arm_cmplx_mag_q15(buf, buf, SAMPLES_PER_MELVEC/2);
    for (int i=0; i < SAMPLES_PER_MELVEC/2; i++) {
        buf[i] = (q15_t) ((((q31_t) buf[i]) * ((q31_t) vmax) ) >> 15 );
    }
    cycles_elapsed = stop_cycle_count("Step 3");


    // Step 4: Apply Mel transform using optimized CSR with double loop unrolling
    start_cycle_count();

    for (int i = 0; i < MELVEC_LENGTH; i++) {
        q31_t sum = 0;  // 32-bit accumulator
        int j;

        // Process four elements per iteration using unrolling
        for (j = hz2mel_row_ptr[i]; j + 3 < hz2mel_row_ptr[i + 1]; j += 4) {
            // Load 4 values for hz2mel_values and buf into 32-bit variables
            q31_t hz_vals1 = *((q31_t*)&hz2mel_values[j]);   // First two values
            q31_t buf_vals1 = *((q31_t*)&buf[hz2mel_col_indices[j]]);
            sum = __SMLAD(hz_vals1, buf_vals1, sum);  // Accumulate two multiplications

            q31_t hz_vals2 = *((q31_t*)&hz2mel_values[j + 2]);  // Next two values
            q31_t buf_vals2 = *((q31_t*)&buf[hz2mel_col_indices[j + 2]]);
            sum = __SMLAD(hz_vals2, buf_vals2, sum);  // Accumulate two more multiplications
        }

        // Handle the remaining 1–3 elements
        for (; j < hz2mel_row_ptr[i + 1]; j++) {
            sum += (q31_t) hz2mel_values[j] * (q31_t) buf[hz2mel_col_indices[j]];
        }

        melvec[i] = (q15_t) __SSAT(sum >> 15, 16);
    }

    cycles_elapsed = stop_cycle_count("Step 4 CSR Double Unrolling");




}



/*void Spectrogram_Compute(q15_t *samples, q15_t *melvec) {
    // Function to compute the spectrogram and apply the Mel transform
    // samples: Input signal samples (Q15 format)
    // melvec: Output Mel-spectrogram vector (Q15 format)

    start_cycle_count();  // Start cycle counter for performance measurement
    arm_mult_q15(samples, hamming_window, buf, SAMPLES_PER_MELVEC);
    // Apply a Hamming window to the input samples (element-wise multiplication)

    uint32_t cycles_elapsed = stop_cycle_count("Step 1");
    // Stop cycle counter and store elapsed cycles for Step 1 (Hamming window)

    start_cycle_count();  // Start cycle counter for Step 2
    arm_rfft_instance_q15 rfft_inst;  // Declare an instance of the RFFT structure
    arm_rfft_init_q15(&rfft_inst, SAMPLES_PER_MELVEC, 0, 1);
    // Initialize the real FFT instance for a forward transform

    arm_rfft_q15(&rfft_inst, buf, buf_fft);
    // Perform the real FFT (RFFT) on the windowed signal

    cycles_elapsed = stop_cycle_count("Step 2");
    // Stop cycle counter and store elapsed cycles for Step 2 (FFT computation)

    start_cycle_count();  // Start cycle counter for Step 3
    q15_t vmax;  // Variable to store the maximum absolute value
    uint32_t pIndex = 0;  // Variable to store the index of vmax

    arm_absmax_q15(buf_fft, SAMPLES_PER_MELVEC, &vmax, &pIndex);
    // Find the maximum absolute value (vmax) in the FFT output

    for (int i = 0; i < SAMPLES_PER_MELVEC; i++) {
        buf[i] = (q15_t)(((q31_t)buf_fft[i] << 15) / ((q31_t)vmax));
        // Normalize FFT output by dividing each value by vmax (scaling to Q15)
    }

    arm_cmplx_mag_q15(buf, buf, SAMPLES_PER_MELVEC / 2);
    // Compute the magnitude of the complex FFT output (half-spectrum)

    for (int i = 0; i < SAMPLES_PER_MELVEC / 2; i++) {
        buf[i] = (q15_t)((((q31_t)buf[i]) * ((q31_t)vmax)) >> 15);
        // Scale back the magnitude values using vmax
    }

    cycles_elapsed = stop_cycle_count("Step 3");
    // Stop cycle counter and store elapsed cycles for Step 3 (normalization & magnitude computation)

    // Step 4: Apply Mel transform using optimized CSR with double loop unrolling
    start_cycle_count();  // Start cycle counter for Step 4

    for (int i = 0; i < MELVEC_LENGTH; i++) {
        q31_t sum = 0;  // 32-bit accumulator for sum of weighted values
        int j;

        // Process elements using loop unrolling and fusion for efficiency
        for (j = hz2mel_row_ptr[i]; j + 3 < hz2mel_row_ptr[i + 1]; j += 4) {

            q31_t hz_vals1 = *((q31_t*)&hz2mel_values[j]);
            q31_t buf_vals1 = *((q31_t*)&buf[hz2mel_col_indices[j]]);
            sum = __SMLAD(hz_vals1, buf_vals1, sum);
            // Perform multiply-accumulate (MAC) on two elements at a time

            q31_t hz_vals2 = *((q31_t*)&hz2mel_values[j + 2]);
            q31_t buf_vals2 = *((q31_t*)&buf[hz2mel_col_indices[j + 2]]);
            sum = __SMLAD(hz_vals2, buf_vals2, sum);
            // Perform MAC on the next two elements
        }

        // Process any remaining elements that couldn't be handled in the unrolled loop
        for (; j < hz2mel_row_ptr[i + 1]; j++) {
            sum += (q31_t)hz2mel_values[j] * (q31_t)buf[hz2mel_col_indices[j]];
            // Compute remaining elements normally
        }

        melvec[i] = (q15_t)__SSAT(sum >> 15, 16);
        // Saturate and convert the final sum to Q15 format
    }

    cycles_elapsed = stop_cycle_count("Step 4 Optimized CSR with Loop Fusion");
    // Stop cycle counter and store elapsed cycles for Step 4 (Mel transform)

}
*/

//Triangle overlap method
/*void Spectrogram_Compute(q15_t *samples, q15_t *melvec) {
    // Step 1: Apply Hamming Window
    start_cycle_count();
    arm_mult_q15(samples, hamming_window, buf, SAMPLES_PER_MELVEC);
    uint32_t cycles_elapsed = stop_cycle_count("Step 1");

    // Step 2: Compute FFT
    start_cycle_count();
    arm_rfft_instance_q15 rfft_inst;
    arm_rfft_init_q15(&rfft_inst, SAMPLES_PER_MELVEC, 0, 1);
    arm_rfft_q15(&rfft_inst, buf, buf_fft);
    cycles_elapsed = stop_cycle_count("Step 2");

    // Step 3: Compute Magnitude Spectrum
    start_cycle_count();
    arm_cmplx_mag_q15(buf_fft, buf, SAMPLES_PER_MELVEC / 2);
    cycles_elapsed = stop_cycle_count("Step 3");

    // Step 4: Apply Mel filterbank using triangle-based accumulation
    start_cycle_count();

    // Initialize mel vector to zero
    for (int i = 0; i < MELVEC_LENGTH; i++) {
        melvec[i] = 0;
    }

    // Process each frequency bin
    for (int f = 0; f < SAMPLES_PER_MELVEC / 2; f++) {
        // Identify the two Mel filters this bin belongs to
        int lower_bin = mel_filter_indices[f];  // Left Mel filter
        int upper_bin = lower_bin + 1;          // Right Mel filter

        q15_t weight_upper = (q15_t)(mel_filter_weights[f] >> 15);  // Right shift to bring into Q15 range
        q15_t weight_lower = (q15_t)((1 << 15) - weight_upper); // Complement weight


        // Distribute magnitude to both Mel filters
        melvec[lower_bin] += (buf[f] * weight_lower) >> 15;
        melvec[upper_bin] += (buf[f] * weight_upper) >> 15;
    }

    cycles_elapsed = stop_cycle_count("Step 4 Triangle Accumulation");
}
*/





/*void Spectrogram_Compute(q15_t *samples, q15_t *melvec)
{
    // STEP 1: Windowing of input samples
    start_cycle_count();  // Start counting cycles
    arm_mult_q15(samples, hamming_window, buf, SAMPLES_PER_MELVEC);
    uint32_t cycles_elapsed = stop_cycle_count("Step 1");

    // STEP 2: Discrete Fourier Transform (FFT)
    start_cycle_count();  // Start counting cycles
    arm_rfft_instance_q15 rfft_inst;
    arm_rfft_init_q15(&rfft_inst, SAMPLES_PER_MELVEC, 0, 1);
    arm_rfft_q15(&rfft_inst, buf, buf_fft);
    cycles_elapsed = stop_cycle_count("Step 2");

    // STEP 3.1: Find maximum value in FFT output
    start_cycle_count();  // Start counting cycles
    q15_t vmax;
    uint32_t pIndex = 0;
    arm_absmax_q15(buf_fft, SAMPLES_PER_MELVEC, &vmax, &pIndex);

    // STEP 3.2: Normalize the FFT output by dividing by vmax
    for (int i = 0; i < SAMPLES_PER_MELVEC; i++) {
        buf[i] = (q15_t) (((q31_t) buf_fft[i] << 15) / ((q31_t)vmax));
    }

    // STEP 3.3: Compute complex magnitude
    arm_cmplx_mag_q15(buf, buf, SAMPLES_PER_MELVEC / 2);

    // STEP 3.4: Denormalize the vector
    for (int i = 0; i < SAMPLES_PER_MELVEC / 2; i++) {
        buf[i] = (q15_t) ((((q31_t) buf[i]) * ((q31_t) vmax)) >> 15);
    }

    cycles_elapsed = stop_cycle_count("Step 3");

    // STEP 4 (Original Version): Apply Mel transform using matrix multiplication
    start_cycle_count();  // Start counting cycles
    arm_matrix_instance_q15 hz2mel_inst, fftmag_inst, melvec_inst;
    arm_mat_init_q15(&hz2mel_inst, MELVEC_LENGTH, SAMPLES_PER_MELVEC / 2, hz2mel_mat);
    arm_mat_init_q15(&fftmag_inst, SAMPLES_PER_MELVEC / 2, 1, buf);
    arm_mat_init_q15(&melvec_inst, MELVEC_LENGTH, 1, melvec);
    arm_mat_mult_fast_q15(&hz2mel_inst, &fftmag_inst, &melvec_inst, buf_tmp);
    cycles_elapsed = stop_cycle_count("Step 4 Original Version");

    // Print results for the original version
    DEBUG_PRINT("MelVec (Original Version): \r\n");
    for (int i = 0; i < MELVEC_LENGTH; i++) {
        DEBUG_PRINT("%d ", melvec[i]);
    }
    DEBUG_PRINT("\r\n");

    // STEP 4 (Optimized Version): Apply Mel transform using optimized CSR with loop unrolling
    start_cycle_count();  // Start counting cycles

    // Optimized CSR version with unrolling factor 12 (as per your request)
    for (int i = 0; i < MELVEC_LENGTH; i++) {
        q31_t sum = 0;  // 32-bit accumulator
        int j;

        // Process elements using unrolling with loop fusion
        for (j = hz2mel_row_ptr[i]; j + 3 < hz2mel_row_ptr[i + 1]; j += 4) {
            // Load 4 values for hz2mel_values and buf into 32-bit variables
            q31_t hz_vals1 = *((q31_t*)&hz2mel_values[j]);   // First two values
            q31_t buf_vals1 = *((q31_t*)&buf[hz2mel_col_indices[j]]);
            sum = __SMLAD(hz_vals1, buf_vals1, sum);  // Accumulate first two multiplications

            q31_t hz_vals2 = *((q31_t*)&hz2mel_values[j + 2]);  // Next two values
            q31_t buf_vals2 = *((q31_t*)&buf[hz2mel_col_indices[j + 2]]);
            sum = __SMLAD(hz_vals2, buf_vals2, sum);  // Accumulate second two multiplications
        }

        // Handle the remaining elements in a single loop, processing up to 3 elements
        for (; j < hz2mel_row_ptr[i + 1]; j++) {
            sum += (q31_t)hz2mel_values[j] * (q31_t)buf[hz2mel_col_indices[j]];
        }

        melvec[i] = (q15_t)__SSAT(sum >> 15, 16);
    }

    cycles_elapsed = stop_cycle_count("Step 4 Optimized CSR with Loop Fusion");

    // Print results for the optimized version
    DEBUG_PRINT("MelVec (Optimized Version): \r\n");
    for (int i = 0; i < MELVEC_LENGTH; i++) {
        DEBUG_PRINT("%d ", melvec[i]);
    }
    DEBUG_PRINT("\r\n");
}*/

/*
void Spectrogram_Compute(q15_t *samples, q15_t *melvec) {

    start_cycle_count();
    arm_mult_q15(samples, hamming_window, buf, SAMPLES_PER_MELVEC);
    uint32_t cycles_elapsed = stop_cycle_count("Step 1");

    start_cycle_count();
    arm_rfft_instance_q15 rfft_inst;
    arm_rfft_init_q15(&rfft_inst, SAMPLES_PER_MELVEC, 0, 1);
    arm_rfft_q15(&rfft_inst, buf, buf_fft);
    cycles_elapsed = stop_cycle_count("Step 2");

    start_cycle_count();
    q15_t vmax;
    uint32_t pIndex = 0;
    arm_absmax_q15(buf_fft, SAMPLES_PER_MELVEC, &vmax, &pIndex);
    for (int i = 0; i < SAMPLES_PER_MELVEC; i++) {
        buf[i] = (q15_t)(((q31_t) buf_fft[i] << 15) / ((q31_t)vmax));
    }
    arm_cmplx_mag_q15(buf, buf, SAMPLES_PER_MELVEC / 2);
    for (int i = 0; i < SAMPLES_PER_MELVEC / 2; i++) {
        buf[i] = (q15_t)((((q31_t) buf[i]) * ((q31_t) vmax)) >> 15);
    }
    cycles_elapsed = stop_cycle_count("Step 3");

    // Step 4: Optimized Mel transform using CSR with reduced memory accesses
    start_cycle_count();

    for (int i = 0; i < MELVEC_LENGTH; i++) {
        q63_t sum = 0;  // Accumulate in 64-bit to prevent overflows
        int row_start = hz2mel_row_ptr[i];
        int row_end = hz2mel_row_ptr[i + 1];

        for (int j = row_start; j < row_end; j++) {
            // Load buffer value **only once** per iteration
            q31_t buf_val = (q31_t)buf[hz2mel_col_indices[j]];
            sum += (q31_t)hz2mel_values[j] * buf_val;
        }

        melvec[i] = (q15_t)__SSAT(sum >> 15, 16);
    }

    cycles_elapsed = stop_cycle_count("Step 4 Optimized CSR with Minimal Memory Access");
}

*/




// Optimized with ARM Cortex-M SIMD instructions: https://arm-software.github.io/CMSIS_5/Core/html/group__intrinsic__SIMD__gr.html#gae0c86f3298532183f3a29f5bb454d354
// uint32_t __SMLAD	(	uint32_t 	val1, uint32_t 	val2, uint32_t 	val3 )	
/*

Q setting dual 16-bit signed multiply with single 32-bit accumulator.

This function enables you to perform two signed 16-bit multiplications, adding both results to a 32-bit accumulate operand.
The Q bit is set if the addition overflows. Overflow cannot occur during the multiplications.

Parameters
val1	first 16-bit operands for each multiplication.
val2	second 16-bit operands for each multiplication.
val3	accumulate value.
Returns
the product of each multiplication added to the accumulate value, as a 32-bit integer.
Operation:
p1 = val1[15:0]  * val2[15:0]
p2 = val1[31:16] * val2[31:16]
res[31:0] = p1 + p2 + val3[31:0]

*/

/*

uint32_t __SSAT16	(	uint32_t 	val1,
const uint32_t 	val2 
)		
Q setting dual 16-bit saturate.

This function enables you to saturate two signed 16-bit values to a selected signed range.
The Q bit is set if either operation saturates.

Parameters
val1	two signed 16-bit values to be saturated.
val2	bit position for saturation, an integral constant expression in the range 1 to 16.
Returns
the sum of the absolute differences of the following bytes, added to the accumulation value:
the signed saturation of the low halfword in val1, saturated to the bit position specified in val2 and returned in the low halfword of the return value.
the signed saturation of the high halfword in val1, saturated to the bit position specified in val2 and returned in the high halfword of the return value.
Operation:
Saturate halfwords in val1 to the signed range specified by the bit position in val2

*/

//required header ?: Required headers: arm_math.h

/*
void mel_transform_simd(q15_t *hz2mel_mat, q15_t *fftmag, q15_t *melvec)
{
    start_cycle_count();

    for (int i = 0; i < MELVEC_LENGTH; i++) {//This loop iterates over each MEL coefficient (row of hz2mel_mat)
        q31_t sum = 0;  // sum is a 32-bit accumulator that stores the dot product result
        for (int j = 0; j < SAMPLES_PER_MELVEC / 2; j += 2) {//This loop computes the dot product between a row of hz2mel_mat and fftmag, using SIMD instructions
            sum = __SMLAD(*((q31_t*)&hz2mel_mat[i * (SAMPLES_PER_MELVEC / 2) + j]),
                          *((q31_t*)&fftmag[j]), sum); //Performs two parallel 16-bit multiplications: A_low * B_low and A_high * B_high, and adds the results to sum (32-bit accumulator).
        }
		//Since ARM SIMD instructions operate on 32-bit registers, we treat two consecutive q15_t elements as a q31_t
        melvec[i] = (q15_t)(__SSAT(sum >> 15, 16));  // Saturate and scale
    }

    uint32_t cycles_elapsed = stop_cycle_count("Step 4 SIMD");
    float elapsed_time_ms = (float)cycles_elapsed / SystemCoreClock * 1000.0f;
    DEBUG_PRINT("Cycle elapsed: %.6ld \r\n", cycles_elapsed);
    DEBUG_PRINT("Spectrogram Compute Step 4 SIMD: %.2f ms\r\n", elapsed_time_ms);

}


void Spectrogram_Compute_simd(q15_t *samples, q15_t *melvec) {


    start_cycle_count();
    arm_mult_q15(samples, hamming_window, buf, SAMPLES_PER_MELVEC);
    uint32_t cycles_elapsed = stop_cycle_count("Step 1");

    start_cycle_count();
    arm_rfft_instance_q15 rfft_inst;
    arm_rfft_init_q15(&rfft_inst, SAMPLES_PER_MELVEC, 0, 1);
    arm_rfft_q15(&rfft_inst, buf, buf_fft);
    cycles_elapsed = stop_cycle_count("Step 2");

    start_cycle_count();
    q15_t vmax;
    uint32_t pIndex=0;
    arm_absmax_q15(buf_fft, SAMPLES_PER_MELVEC, &vmax, &pIndex);
    for (int i=0; i < SAMPLES_PER_MELVEC; i++) {
        buf[i] = (q15_t) (((q31_t) buf_fft[i] << 15) /((q31_t)vmax));
    }
    arm_cmplx_mag_q15(buf, buf, SAMPLES_PER_MELVEC/2);
    for (int i=0; i < SAMPLES_PER_MELVEC/2; i++) {
        buf[i] = (q15_t) ((((q31_t) buf[i]) * ((q31_t) vmax) ) >> 15 );
    }
    cycles_elapsed = stop_cycle_count("Step 3");

    start_cycle_count();
    for (int i = 0; i < MELVEC_LENGTH; i++) {
        q31_t sum = 0;
        for (int j = 0; j < SAMPLES_PER_MELVEC / 2; j += 2) {
            sum = __SMLAD(*((q31_t*)&hz2mel_mat[i * (SAMPLES_PER_MELVEC / 2) + j]),
                          *((q31_t*)&buf[j]), sum);
        }
        melvec[i] = (q15_t)(__SSAT(sum >> 15, 16));
    }
    cycles_elapsed = stop_cycle_count("Step 4 SIMD");

}


void Spectrogram_Compute_simd_csr(q15_t *samples, q15_t *melvec) {
	DEBUG_PRINT("Melvec before modification with SIMD!!!!\n\r");
		for (int i = 0; i < MELVEC_LENGTH; i++) {
		   DEBUG_PRINT("%d ", melvec[i]);  // Printing as q15_t (integer representation)
		   }
		DEBUG_PRINT("\r\n");
    // Step 1: Windowing of samples
    start_cycle_count();
    arm_mult_q15(samples, hamming_window, buf, SAMPLES_PER_MELVEC);
    uint32_t cycles_elapsed = stop_cycle_count("Step 1");
    DEBUG_PRINT("Cycle elapsed: %.6ld \r\n", cycles_elapsed);

    DEBUG_PRINT("Windowed Samples (buf):\r\n");
    for (int i = 0; i < SAMPLES_PER_MELVEC; i++) {
        DEBUG_PRINT("%d ", buf[i]);  // Print windowed samples (q15_t values)
    }
    DEBUG_PRINT("\r\n");

    // Step 2: FFT Computation
    start_cycle_count();
    arm_rfft_instance_q15 rfft_inst;
    arm_rfft_init_q15(&rfft_inst, SAMPLES_PER_MELVEC, 0, 1);
    arm_rfft_q15(&rfft_inst, buf, buf_fft);
    cycles_elapsed = stop_cycle_count("Step 2");
    DEBUG_PRINT("Cycle elapsed: %.6ld \r\n", cycles_elapsed);

    DEBUG_PRINT("FFT Output (buf_fft):\r\n");
    for (int i = 0; i < SAMPLES_PER_MELVEC; i++) {
        DEBUG_PRINT("%d ", buf_fft[i]);  // Print FFT output (q15_t values)
    }
    DEBUG_PRINT("\r\n");

    // Step 3.1: Find maximum value in FFT output
    start_cycle_count();
    q15_t vmax;
    uint32_t pIndex = 0;
    arm_absmax_q15(buf_fft, SAMPLES_PER_MELVEC, &vmax, &pIndex);
    DEBUG_PRINT("Max FFT Value (vmax): %d \r\n", vmax);

    // Step 3.2: Normalize FFT output by dividing by vmax
    for (int i = 0; i < SAMPLES_PER_MELVEC; i++) {
        buf[i] = (q15_t) (((q31_t) buf_fft[i] << 15) / ((q31_t)vmax));
    }

    DEBUG_PRINT("Normalized FFT (buf):\r\n");
    for (int i = 0; i < SAMPLES_PER_MELVEC; i++) {
        DEBUG_PRINT("%d ", buf[i]);  // Print normalized FFT output
    }
    DEBUG_PRINT("\r\n");

    // Step 3.3: Compute complex magnitude
    arm_cmplx_mag_q15(buf, buf, SAMPLES_PER_MELVEC / 2);
    DEBUG_PRINT("Magnitude of FFT (buf after cmplx_mag):\r\n");
    for (int i = 0; i < SAMPLES_PER_MELVEC / 2; i++) {
        DEBUG_PRINT("%d ", buf[i]);  // Print magnitude of FFT output
    }
    DEBUG_PRINT("\r\n");

    // Step 3.4: Denormalize magnitude
    for (int i = 0; i < SAMPLES_PER_MELVEC / 2; i++) {
        buf[i] = (q15_t) ((((q31_t) buf[i]) * ((q31_t) vmax)) >> 15);
    }

    DEBUG_PRINT("Denormalized Magnitude (buf):\r\n");
    for (int i = 0; i < SAMPLES_PER_MELVEC / 2; i++) {
        DEBUG_PRINT("%d ", buf[i]);  // Print denormalized magnitude
    }
    DEBUG_PRINT("\r\n");

    cycles_elapsed = stop_cycle_count("Step 3");
    DEBUG_PRINT("Cycle elapsed: %.6ld \r\n", cycles_elapsed);

    // Step 4: Apply Mel transform using CSR
    start_cycle_count();
    for (int i = 0; i < MELVEC_LENGTH; i++) {
        q31_t sum = 0;
        for (int j = hz2mel_row_ptr[i]; j < hz2mel_row_ptr[i + 1]; j++) {
            sum = __SMLAD(hz2mel_values[j], buf[hz2mel_col_indices[j]], sum);
        }
        melvec[i] = (q15_t)(__SSAT(sum >> 15, 16));

        DEBUG_PRINT("Melvec[%d]: %d \r\n", i, melvec[i]);  // Print each Mel vector element
    }
    cycles_elapsed = stop_cycle_count("Step 4 CSR");
    DEBUG_PRINT("Cycle elapsed: %.6ld \r\n", cycles_elapsed);

    // Print computed melvec
    DEBUG_PRINT("Computed Mel Spectrogram with csr:\r\n");
    for (int i = 0; i < MELVEC_LENGTH; i++) {
        DEBUG_PRINT("%d ", melvec[i]);  // Print final Mel spectrogram vector (q15_t values)
    }
    DEBUG_PRINT("\r\n");
}
*/
