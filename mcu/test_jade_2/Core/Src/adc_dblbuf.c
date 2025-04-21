#include <adc_dblbuf.h>
#include "config.h"
#include "main.h"
#include "spectrogram.h"
#include "arm_math.h"
#include "utils.h"
#include "s2lp.h"
#include "packet.h"


static volatile uint16_t ADCDoubleBuf[2*ADC_BUF_SIZE]; /* ADC group regular conversion data (array of data) */
static volatile uint16_t* ADCData[2] = {&ADCDoubleBuf[0], &ADCDoubleBuf[ADC_BUF_SIZE]};
static volatile uint8_t ADCDataRdy[2] = {0, 0};

static volatile uint8_t cur_melvec = 0;
static q15_t mel_vectors[N_MELVECS][MELVEC_LENGTH];

static uint32_t packet_cnt = 0;

static volatile int32_t rem_n_bufs = 0;


//For the threshold:
//t_acquisition = 1s
#define N_STS	50		//short-term sum-window N_STS=50 is a window of 5ms=50*(1/10204) for a sampling frequency of 10204Hz
#define	N_LTS	8000	//Long-term sum-window (N_LTS>N_STS)  N_LTS=200 is a window of 20ms for a sampling frequency of 10204Hz
#define K		1.75		//factor to modulate the comparison

//For fs=3472
// N_STS = 18
// N_LTS =  2778

static float sts_sum = 0.0f;	//short-term sum
static float lts_sum = 0.0f;	//long-term sum
static float sts_buffer[N_STS] = {0}; //Circular buffer for STS at the start all values are set to 0
static float lts_buffer[N_LTS] = {0}; //Circular buffer for LTS
static uint16_t sts_idx = 0;
static uint16_t lts_idx = 0;


//For the threshold on the melvec:
#define MEL_N_STS 5    // Short-term sum over 5 frames
#define MEL_N_LTS 50   // Long-term sum over 50 frames

static float mel_sts_sum = 0.0f;
static float mel_lts_sum = 0.0f;
static float mel_sts_buffer[MEL_N_STS] = {0};  // Circular buffer for STS
static float mel_lts_buffer[MEL_N_LTS] = {0};  // Circular buffer for LTS
static uint16_t mel_sts_idx = 0;
static uint16_t mel_lts_idx = 0;

static const q15_t ones_vector[MELVEC_LENGTH] = { [0 ... MELVEC_LENGTH-1] = 0x7FFF };




//static uint8_t packet_detected = 0; //to know if the threshold was atteined or not
volatile int packet_detected = 0;

#define TOTAL_FRAMES 20  // Ensure we always send 20 frames

static int processing_active = 0; // Flag to continue processing 20 frames after detection


int StartADCAcq(int32_t n_bufs) {
	rem_n_bufs = n_bufs;
	cur_melvec = 0;
	if (rem_n_bufs != 0) {
		return HAL_ADC_Start_DMA(&hadc1, (uint32_t *)ADCDoubleBuf, 2*ADC_BUF_SIZE);
	} else {
		return HAL_OK;
	}
}

int IsADCFinished(void) {
	return (rem_n_bufs == 0);
}

static void StopADCAcq() {
	HAL_ADC_Stop_DMA(&hadc1);
}

static void print_spectrogram(void) {
#if (DEBUGP == 1)
	start_cycle_count();
	DEBUG_PRINT("Acquisition complete, sending the following FVs\r\n");
	for(unsigned int j=0; j < N_MELVECS; j++) {
		DEBUG_PRINT("FV #%u:\t", j+1);
		for(unsigned int i=0; i < MELVEC_LENGTH; i++) {
			DEBUG_PRINT("%.2f, ", q15_to_float(mel_vectors[j][i]));
		}
		DEBUG_PRINT("\r\n");
	}
	stop_cycle_count("Print FV");
#endif
}

static void print_encoded_packet(uint8_t *packet) {
#if (DEBUGP == 1)
	char hex_encoded_packet[2*PACKET_LENGTH+1];
	hex_encode(hex_encoded_packet, packet, PACKET_LENGTH);
	DEBUG_PRINT("DF:HEX:%s\r\n", hex_encoded_packet);
#endif
}

static void encode_packet(uint8_t *packet, uint32_t* packet_cnt) {
	// BE encoding of each mel coef
	for (size_t i = 0; i < N_MELVECS; i++) {
	    //DEBUG_PRINT("mel_vectors[%d]: ", i);  // Print the vector index
	    for (size_t j = 0; j < MELVEC_LENGTH; j++) {
	        (packet + PACKET_HEADER_LENGTH)[(i * MELVEC_LENGTH + j) * 2]   = mel_vectors[i][j] >> 8;
	        (packet + PACKET_HEADER_LENGTH)[(i * MELVEC_LENGTH + j) * 2 + 1] = mel_vectors[i][j] & 0xFF;

	        // Print all values of mel_vectors[i] in one line
	        //DEBUG_PRINT("%d ", mel_vectors[i][j]);  // Print the value of each element
	    }
	    //DEBUG_PRINT("\r\n");  // Move to the next line after printing the whole vector
	}
	// Write header and tag into the packet.
	make_packet(packet, PAYLOAD_LENGTH, 0, *packet_cnt);
	*packet_cnt += 1;

	// Print the packet count!
	DEBUG_PRINT("Packet sent: %lu\r\n", *packet_cnt);

	if (*packet_cnt == 0) {
		// Should not happen as packet_cnt is 32-bit and we send at most 1 packet per second.
		DEBUG_PRINT("Packet counter overflow.\r\n");
		Error_Handler();
	}
}

static void send_spectrogram() {
	uint8_t packet[PACKET_LENGTH];

	//start_cycle_count();
	encode_packet(packet, &packet_cnt);
	//stop_cycle_count("Encode packet");

	//start_cycle_count();

	RCC_ClkInitTypeDef clk_init = {0};
	uint32_t flash_latency;

	HAL_RCC_GetClockConfig(&clk_init, &flash_latency);
	clk_init.AHBCLKDivider = RCC_SYSCLK_DIV4; // ou RCC_SYSCLK_DIV1, _DIV4, etc.
	DEBUG_PRINT("Before clock to 24MHz.\r\n");
	HAL_RCC_ClockConfig(&clk_init, FLASH_LATENCY_0);

	DEBUG_PRINT("Clock to 24MHz.\r\n");

	S2LP_Send(packet, PACKET_LENGTH);

	HAL_RCC_GetClockConfig(&clk_init, &flash_latency);
	clk_init.AHBCLKDivider = RCC_SYSCLK_DIV16; // ou RCC_SYSCLK_DIV1, _DIV4, etc.
	HAL_RCC_ClockConfig(&clk_init, FLASH_LATENCY_0);

	DEBUG_PRINT("Clock to 3MHz.\r\n");

	//stop_cycle_count("Send packet");

	//print_encoded_packet(packet);
}



/*
// This function is called whenever the ADC completes a DMA transfer.
static void ADC_Callback(int buf_cplt) {
	int threshold_detected_in_frame =0;
    // Decreases the counter rem_n_bufs which tracks how many buffers remain to be processed.
    if (rem_n_bufs != -1) {
    	//DEBUG_PRINT("rem_n_bufs before decrement: %d\r\n", rem_n_bufs);
        rem_n_bufs--;
    }

    // If rem_n_bufs = 0, stop ADC acquisition.
    if (rem_n_bufs == 0) {
        DEBUG_PRINT("The acquisition stops because the counter is empty\n");
        StopADCAcq();
    } else if (ADCDataRdy[1 - buf_cplt]) { // Checks if the other buffer is still full.
        DEBUG_PRINT("Error: ADC Data buffer full\r\n");
        Error_Handler();
    }
    ADCDataRdy[buf_cplt] = 1; // Mark buffer as ready for processing.

    // Algorithm for threshold detection (dual running sum with soft threshold).
    for (int i = 0; i < ADC_BUF_SIZE; i++) {
        float sample = (float)ADCData[buf_cplt][i]; // Convert uint16_t to float.
        float sample_corrected = sample - 2048.0f; // Adjust baseline.
        float sample_mag = fabs(sample_corrected); // Get AC magnitude.

        // Short-term sum sliding window.
        sts_sum -= sts_buffer[sts_idx];
        sts_buffer[sts_idx] = sample_mag;
        sts_sum += sample_mag;
        sts_idx = (sts_idx + 1) % N_STS;

        // Long-term sum sliding window.
        lts_sum -= lts_buffer[lts_idx];
        lts_buffer[lts_idx] = sts_buffer[(sts_idx + N_STS - 1) % N_STS]; // Use last valid STS value.
        lts_sum += lts_buffer[lts_idx];
        lts_idx = (lts_idx + 1) % N_LTS;

        // Normalization.
        float norm_sts = sts_sum / N_STS;
        float norm_lts = (lts_sum / N_LTS) * K;

        // Packet detection.
        if (norm_sts > norm_lts) {
            packet_detected = 1;
            threshold_detected_in_frame = 1;

            //if (!processing_active) {
                //processing_active = 1;
                //cur_melvec = 0;  // Start from first frame.
            //}
            //DEBUG_PRINT("Packet detected is set to true for rem_n_bufs= %d\r\n", rem_n_bufs);
        }

    }

    if (threshold_detected_in_frame == 1){
    	DEBUG_PRINT("Packet detected is set to true for rem_n_bufs= %d\r\n", rem_n_bufs);
    }
	Spectrogram_Format((q15_t *)ADCData[buf_cplt]);
	Spectrogram_Compute((q15_t *)ADCData[buf_cplt], mel_vectors[cur_melvec]);

	//DEBUG_PRINT("Computed Spectrogram Frame %d:\r\n", cur_melvec);
	//for (int i = 0; i < MELVEC_LENGTH; i++) {
		//DEBUG_PRINT("%d ", mel_vectors[cur_melvec][i]);
	//}
	//DEBUG_PRINT("\r\n");

	cur_melvec++; // Move to next frame.

	// Stop processing after 20 frames.
	if (cur_melvec >= TOTAL_FRAMES) {
		if (packet_detected){
			send_spectrogram();
		}
		//packet_detected = 0; // Reset packet detection after sending.
		else{
			DEBUG_PRINT("20 frames reached, but threshold not detected, we do not send\r\n");
		}

		cur_melvec = 0;

	}


    ADCDataRdy[buf_cplt] = 0;
}
*/

//Version without arm functions
/*
static void ADC_Callback(int buf_cplt) {

    //int threshold_detected_in_frame = 0;

    // Decrease the counter tracking remaining buffers.
    if (rem_n_bufs != -1) {
        rem_n_bufs--;
    }

    // If no buffers remain, stop acquisition.
    if (rem_n_bufs == 0) {
        DEBUG_PRINT("Acquisition stopped: No more buffers to process.\n");
        StopADCAcq();
    }
    // Check if the other buffer is still full (potential overflow).
    else if (ADCDataRdy[1 - buf_cplt]) {
        DEBUG_PRINT("Error: ADC Data buffer full\r\n");
        Error_Handler();
    }

    // Mark the buffer as ready for processing.
    ADCDataRdy[buf_cplt] = 1;

    // **Compute Spectrogram**
    Spectrogram_Format((q15_t *)ADCData[buf_cplt]);
    Spectrogram_Compute((q15_t *)ADCData[buf_cplt], mel_vectors[cur_melvec]);

    start_cycle_count();
    // **Mel-Spectrogram Based Threshold Algorithm**
    float mel_energy = 0.0f;

    // Sum energy across Mel bands.
    for (int i = 0; i < MELVEC_LENGTH; i++) {
        mel_energy += mel_vectors[cur_melvec][i];
    }

    // Update Short-Term Sum (STS)
    mel_sts_sum -= mel_sts_buffer[mel_sts_idx];
    mel_sts_buffer[mel_sts_idx] = mel_energy;
    mel_sts_sum += mel_energy;
    mel_sts_idx = (mel_sts_idx + 1) % MEL_N_STS;

    // Update Long-Term Sum (LTS)
    mel_lts_sum -= mel_lts_buffer[mel_lts_idx];
    mel_lts_buffer[mel_lts_idx] = mel_sts_buffer[(mel_sts_idx + MEL_N_STS - 1) % MEL_N_STS];
    mel_lts_sum += mel_lts_buffer[mel_lts_idx];
    mel_lts_idx = (mel_lts_idx + 1) % MEL_N_LTS;

    // Compute Normalized Energy Threshold
    float norm_mel_sts = mel_sts_sum / MEL_N_STS;
    float norm_mel_lts = (mel_lts_sum / MEL_N_LTS) * K;

    // Packet Detection
    if (norm_mel_sts > norm_mel_lts) {
        packet_detected = 1;
        //threshold_detected_in_frame = 1;
        DEBUG_PRINT("Packet detected on mel-spectrogram energy at rem_n_bufs=%d\r\n", rem_n_bufs);
    }
    uint32_t cycles_Callback = stop_cycle_count("Fin Threshold\r\n");

    cur_melvec++; // Move to next frame.

    // **Send Spectrogram Only if a Packet is Detected**
    if (cur_melvec >= TOTAL_FRAMES) {
        if (packet_detected) {
            send_spectrogram();
        } else {
            DEBUG_PRINT("20 frames reached, but threshold not detected, skipping transmission.\r\n");
        }

        cur_melvec = 0;  // Reset for the next batch.
    }

    // Reset ADC buffer ready flag.
    ADCDataRdy[buf_cplt] = 0;

}*/

//Version with arm functions
static void ADC_Callback(int buf_cplt) {
    //int threshold_detected_in_frame = 0;

    // Decrease the counter tracking remaining buffers.
    if (rem_n_bufs != -1) {
        rem_n_bufs--;
    }

    // If no buffers remain, stop acquisition.
    if (rem_n_bufs == 0) {
        DEBUG_PRINT("Acquisition stopped: No more buffers to process.\n");
        StopADCAcq();
    }
    // Check if the other buffer is still full (potential overflow).
    else if (ADCDataRdy[1 - buf_cplt]) {
        DEBUG_PRINT("Error: ADC Data buffer full\r\n");
        Error_Handler();
    }

    // Mark the buffer as ready for processing.
    ADCDataRdy[buf_cplt] = 1;

    // **Compute Spectrogram**
    Spectrogram_Format((q15_t *)ADCData[buf_cplt]);
    Spectrogram_Compute((q15_t *)ADCData[buf_cplt], mel_vectors[cur_melvec]);

    //start_cycle_count();
    // **Mel-Spectrogram Based Threshold Algorithm**
    float mel_energy = 0.0f;

    // Sum energy across Mel bands.
    // Compute sum of mel energy using CMSIS-DSP dot product
    arm_dot_prod_q15(mel_vectors[cur_melvec], (q15_t*)ones_vector, MELVEC_LENGTH, &mel_energy);

    // Optimize STS update
    arm_sub_f32(&mel_sts_sum, &mel_sts_buffer[mel_sts_idx], &mel_sts_sum, 1);
    arm_copy_f32(&mel_energy, &mel_sts_buffer[mel_sts_idx], 1);
    arm_add_f32(&mel_sts_sum, &mel_energy, &mel_sts_sum, 1);
    mel_sts_idx = (mel_sts_idx + 1) % MEL_N_STS;

    // Optimize LTS update
    arm_sub_f32(&mel_lts_sum, &mel_lts_buffer[mel_lts_idx], &mel_lts_sum, 1);
    arm_copy_f32(&mel_sts_buffer[(mel_sts_idx + MEL_N_STS - 1) % MEL_N_STS], &mel_lts_buffer[mel_lts_idx], 1);
    arm_add_f32(&mel_lts_sum, &mel_lts_buffer[mel_lts_idx], &mel_lts_sum, 1);
    mel_lts_idx = (mel_lts_idx + 1) % MEL_N_LTS;

    // Compute Normalized Energy Threshold (we scale only one value so more efficient like this)
    float norm_mel_sts = mel_sts_sum / MEL_N_STS;
    float norm_mel_lts = (mel_lts_sum / MEL_N_LTS) * K;

    // Packet Detection
    if (norm_mel_sts > norm_mel_lts) {
        packet_detected = 1;
        //threshold_detected_in_frame = 1;
        //DEBUG_PRINT("Packet detected on mel-spectrogram energy at rem_n_bufs=%d\r\n", rem_n_bufs);
    }
    //uint32_t cycles_Callback = stop_cycle_count("Fin Threshold\r\n");

    cur_melvec++; // Move to next frame.

    // **Send Spectrogram Only if a Packet is Detected**
    if (cur_melvec >= TOTAL_FRAMES) {
        if (packet_detected) {
            send_spectrogram();
        } else {
            //DEBUG_PRINT("20 frames reached, but threshold not detected, skipping transmission.\r\n");
        }

        cur_melvec = 0;  // Reset for the next batch.
    }

    // Reset ADC buffer ready flag.
    ADCDataRdy[buf_cplt] = 0;
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
	ADC_Callback(1);
}

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc)
{
	start_cycle_count();
	ADC_Callback(0);
	stop_cycle_count("ConvHalfCpltCallback");
}
