/*
 * config.h
 */

#ifndef INC_CONFIG_H_
#define INC_CONFIG_H_

#include <stdio.h>

// Runtime parameters
#define MAIN_APP 1
#define EVAL_RADIO 0

#define RUN_CONFIG MAIN_APP

#define DYNAMIC_CLOCK 0		// Set prescaler to /16 during acquisition and to /4 during transmission, DEPRECATED in final version

// Radio parameters
#define ENABLE_RADIO 1
#define SEND_8BIT_MELS 1 // Only send the 8 MSB bits of each number of the 16bit computed melspectrogram
#define BIT_SENSITIVITY 5 // If SEND_8BIT_MELS enabled, increase sensitivity

// General UART enable/disable (disable for low-power operation)
#define ENABLE_UART 1

// In continuous mode, we start and stop continuous acquisition on button press.
// In non-continuous mode, we send a single packet on button press.
#define CONTINUOUS_ACQ 1

#define BYPASS_BTN_PRESS 0

//For the threshold on the melvec:
#define BYPASS_THRESHOLD 0
#define MEL_N_STS 5    // Short-term sum over 5 frames
#define MEL_N_LTS 10000   // Long-term sum over 50 frames
#define K		  1.6		//factor to modulate the comparison


// Spectrogram parameters
#define SAMPLES_PER_MELVEC 512
#define MELVEC_LENGTH 10
#define N_MELVECS 20

// Enable performance measurements
#define PERF_COUNT 0

// Enable debug print
#define DEBUGP 1

#if (DEBUGP == 1)
#define DEBUG_PRINT(...) do{ printf(__VA_ARGS__ ); } while( 0 )
#else
#define DEBUG_PRINT(...) do{ } while ( 0 )
#endif



#endif /* INC_CONFIG_H_ */
