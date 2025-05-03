#include <adc_dblbuf.h>
#include "config.h"
#include "main.h"
#include "spectrogram.h"
#include "arm_math.h"
#include "utils.h"
#include "s2lp.h"
#include "packet.h"


static volatile uint8_t cur_melvec = 0;


static volatile uint16_t ADCDoubleBuf[2*ADC_BUF_SIZE]; /* ADC group regular conversion data (array of data) */
static volatile uint16_t* ADCData[2] = {&ADCDoubleBuf[0], &ADCDoubleBuf[ADC_BUF_SIZE]};
static q15_t mel_vectors[N_MELVECS][MELVEC_LENGTH];

static volatile uint8_t ADCDataRdy[2] = {0, 0};

static uint32_t packet_cnt = 0;

static volatile int32_t rem_n_bufs = 0;

static float mel_sts_sum = 0.0f;
static float mel_lts_sum = 0.0f;
static float mel_sts_buffer[MEL_N_STS] = {0};  // Circular buffer for STS
static float mel_lts_buffer[MEL_N_LTS] = {0};  // Circular buffer for LTS
static uint16_t mel_sts_idx = 0;
static uint16_t mel_lts_idx = 0;


volatile int packet_detected = 0;


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

#if (DEBUGP == 1)
static void print_spectrogram(void) {

	start_cycle_count();
	DEBUG_PRINT("Acquisition complete, sending the following FVs\r\n");
	for(unsigned int j=0; j < N_MELVECS; j++) {
		DEBUG_PRINT("FV #%u:\t", j+1);
		for(unsigned int i=0; i < MELVEC_LENGTH; i++) {
			DEBUG_PRINT("%.5f, ", q15_to_float(mel_vectors[j][i]));
		}
		DEBUG_PRINT("\r\n");
	}
	stop_cycle_count("Print FV");
}

static void print_encoded_packet(uint8_t *packet) {
	char hex_encoded_packet[2*PACKET_LENGTH+1];
	hex_encode(hex_encoded_packet, packet, PACKET_LENGTH);
	DEBUG_PRINT("DF:HEX:%s\r\n", hex_encoded_packet);
}
#endif

static void encode_packet(uint8_t *packet, uint32_t* packet_cnt) {

	for (size_t i = 0; i < N_MELVECS; i++) {
		for (size_t j = 0; j < MELVEC_LENGTH; j++) {
			#if SEND_8BIT_MELS
				(packet + PACKET_HEADER_LENGTH)[i * MELVEC_LENGTH + j] = mel_vectors[i][j] >> (8 -BIT_SENSITIVITY);
			#else
				// BE encoding of each mel coef
				(packet + PACKET_HEADER_LENGTH)[(i * MELVEC_LENGTH + j) * 2]   = mel_vectors[i][j] >> 8;
				(packet + PACKET_HEADER_LENGTH)[(i * MELVEC_LENGTH + j) * 2 + 1] = mel_vectors[i][j] & 0xFF;
			#endif
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

	HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_8);

#if DYNAMIC_CLOCK

	RCC_ClkInitTypeDef clk_init = {0};
	uint32_t flash_latency;

	HAL_RCC_GetClockConfig(&clk_init, &flash_latency);
	clk_init.AHBCLKDivider = RCC_SYSCLK_DIV4; // ou RCC_SYSCLK_DIV1, _DIV4, etc.
	HAL_RCC_ClockConfig(&clk_init, FLASH_LATENCY_0);

	DEBUG_PRINT("Prescaler to /4.\r\n");

	S2LP_Send(packet, PACKET_LENGTH);

	HAL_RCC_GetClockConfig(&clk_init, &flash_latency);
	clk_init.AHBCLKDivider = RCC_SYSCLK_DIV16; // ou RCC_SYSCLK_DIV1, _DIV4, etc.
	HAL_RCC_ClockConfig(&clk_init, FLASH_LATENCY_0);

	DEBUG_PRINT("Prescaler to /16.\r\n");
#else

	S2LP_Send(packet, PACKET_LENGTH);

#endif

	HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_8);

	//stop_cycle_count("Send packet");
#if (DEBUGP == 1)
	print_encoded_packet(packet);
#endif
}



//Version with arm functions
static void ADC_Callback(int buf_cplt) {

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

    static const q15_t ones_vector[MELVEC_LENGTH] = { [0 ... MELVEC_LENGTH-1] = 0x7FFF };
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
    }
    //uint32_t cycles_Callback = stop_cycle_count("Fin Threshold\r\n");

    cur_melvec++; // Move to next frame.

	#if BYPASS_THRESHOLD
    	packet_detected = 1;
	#endif

    // **Send Spectrogram Only if a Packet is Detected**
    if (cur_melvec >= N_MELVECS) {
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
