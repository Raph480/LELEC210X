#include <adc_dblbuf.h>
#include "config.h"
#include "main.h"
#include "arm_math.h"
#include "utils.h"



static volatile uint16_t ADCDoubleBuf[2*ADC_BUF_SIZE]; /* ADC group regular conversion data (array of data) */
static volatile uint16_t* ADCData[2] = {&ADCDoubleBuf[0], &ADCDoubleBuf[ADC_BUF_SIZE]};
static volatile uint8_t ADCDataRdy[2] = {0, 0};


char hex_encoded_buffer[4*ADC_BUF_SIZE+1];

int StartADCAcq() {
	return HAL_ADC_Start_DMA(&hadc1, (uint32_t *)ADCDoubleBuf, 2*ADC_BUF_SIZE);
}


static void StopADCAcq() {
	HAL_ADC_Stop_DMA(&hadc1);
}


void print_buffer(uint16_t *buffer) {
	hex_encode(hex_encoded_buffer, (uint8_t*)buffer, 2*ADC_BUF_SIZE);
	printf("SND:HEX:%s\r\n", hex_encoded_buffer);
}

static void ADC_Callback(int buf_cplt) {
	if (ADCDataRdy[1-buf_cplt]) {
		DEBUG_PRINT("Error: ADC Data buffer full\r\n");
		Error_Handler();
	}
	ADCDataRdy[buf_cplt] = 1;
	print_buffer((q15_t *) ADCData[buf_cplt]);
	ADCDataRdy[buf_cplt] = 0;
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
	ADC_Callback(1);
}

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc)
{
	ADC_Callback(0);
}
