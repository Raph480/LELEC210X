#ifndef INC_ADC_DBLBUF_H_
#define INC_ADC_DBLBUF_H_

#include "main.h"
#include "config.h"
#include "arm_math.h"

// ADC parameters
#define ADC_BUF_SIZE SAMPLES_PER_MELVEC


int StartADCAcq();

void print_buffer(uint16_t *buffer);

extern ADC_HandleTypeDef hadc1;

#endif /* INC_ADC_DBLBUF_H_ */
