/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "aes.h"
#include "dma.h"
#include "spi.h"
#include "tim.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include "arm_math.h"
#include "adc_dblbuf.h"
#include "retarget.h"
#include "s2lp.h"
#include "spectrogram.h"
#include "eval_radio.h"
#include "packet.h"
#include "config.h"
#include "utils.h"
#include "usart.h"


/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

volatile uint8_t btn_press;

//AES Test
uint32_t input_data[4]= {0x6bc1bee2,0x2e409f96,0xe93d7e11,0x7393172a};
uint32_t encrypted_data[4];
uint32_t decrypted_data[4];
uint8_t test_msg_hw[] = "Hello World!abcdef";
size_t test_msg_len_hw = sizeof(test_msg_hw) - 1;
uint8_t tag_hw[16];
uint8_t test_msg[] = "Hello World!abcdef";
size_t test_msg_len = sizeof(test_msg) - 1;
uint8_t tag[16];

// DMA buffers
//uint8_t padded_msg_dma[1024]; // Adjust size based on message length
//uint8_t state_dma[1024];      // Adjust size based on message length

//For the spectrogram
// Global buffers for testing
q15_t test_samples[SAMPLES_PER_MELVEC];
q15_t melvec_ref[MELVEC_LENGTH];
q15_t melvec_opt[MELVEC_LENGTH];
q15_t melvec_csr[MELVEC_LENGTH];


#define SAMPLE_RATE 16000  // Example: 16 kHz sampling rate
#define SIGNAL_FREQ 1000   // Example: 1 kHz sine wave



/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
//state is not the good type of buffer, look with statew and so on, look at the description of the function
//We can give the entire message to the function but the message must be a multiple of 16 bytes, so we may need to add zeros to the end
//the tag is obtain by taking the last 16 bytes (and not the first 16 bytes like it is done for now)

/*
void tag_cbc_mac_hardware(uint8_t *tag, const uint8_t *msg, size_t msg_len) {
    size_t padded_len = (msg_len + 15) & ~15; // Round up to next multiple of 16
    uint8_t padded_msg[padded_len] __attribute__((aligned(4))); // Ensure 32-bit alignment
    uint8_t state[padded_len] __attribute__((aligned(4))); // Ensure 32-bit alignment
    size_t i, j;

    // Manually copy the message and pad with zeros
    for (i = 0; i < msg_len; i++) {
        padded_msg[i] = msg[i];
    }
    for (; i < padded_len; i++) {
        padded_msg[i] = 0; // Zero-padding
    }

    // Encrypt the entire message in one call
    if (HAL_CRYP_AESCBC_Encrypt(&hcryp, padded_msg,(uint16_t) padded_len, state, HAL_MAX_DELAY) != HAL_OK) {
        DEBUG_PRINT("AES encryption error\n");
    }

    // Copy the last block (CBC-MAC tag)
    for (j = 0; j < 16; j++) {
        tag[j] = state[padded_len - 16 + j];}

}

*/

/*
void tag_cbc_mac_hardware(uint8_t *tag, const uint8_t *msg, size_t msg_len) {
    size_t padded_len = (msg_len + 15) & ~15; // Round up to next multiple of 16
    uint8_t padded_msg[padded_len] __attribute__((aligned(4))); // Ensure 32-bit alignment
    size_t i;

    // Manually copy the message and pad with zeros
    for (i = 0; i < msg_len; i++) {
        padded_msg[i] = msg[i];
    }
    for (; i < padded_len; i++) {
        padded_msg[i] = 0; // Zero-padding
    }

    // Start interrupt-based AES CBC encryption
    if (HAL_CRYP_AESCBC_Encrypt_IT(&hcryp, padded_msg, padded_len, state_dma) != HAL_OK) {
        DEBUG_PRINT("AES encryption error\n");
    }

    // MCU enters sleep state while waiting for interrupt to signal encryption completion
    while (HAL_CRYP_GetState(&hcryp) != HAL_CRYP_STATE_READY) {
    	DEBUG_PRINT("Waiting for encryption...\n");
        __WFI(); // Sleep until operation finishes and interrupt is triggered
    }

    // Copy the last block (CBC-MAC tag)
    for (i = 0; i < 16; i++) {
        tag[i] = state_dma[padded_len - 16 + i];
    }
}

*/




void generate_test_data() {
    for (int i = 0; i < SAMPLES_PER_MELVEC; i++) {
        // Compute phase increment correctly (scaled to Q15 full range)
        uint16_t phase = (uint16_t)(((uint32_t)i * SIGNAL_FREQ * 65536) / SAMPLE_RATE);

        // Get sine value in Q15 format
        test_samples[i] = arm_sin_q15(phase);
    }
}



void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
	if (GPIO_Pin == B1_Pin) {
		btn_press = 1;
	}
	else if (GPIO_Pin == RADIO_INT_Pin)
		S2LP_IRQ_Handler();
}

static void acquire_and_send_packet() {

	//uint8_t plaintext[800]; //Example buffer for feature vectors
	//uint8_t ciphertext[800]; //Encryption buffer

	if (StartADCAcq(N_MELVECS) != HAL_OK) {
		DEBUG_PRINT("Error while enabling the DMA\r\n");
	}

	while (!IsADCFinished()) {

		__WFI();
	}



}


void run(void)
{
	btn_press = 0;

	while (1)
	{
	  while (!btn_press) {

		  HAL_GPIO_WritePin(GPIOB, LD2_Pin, GPIO_PIN_SET);
		  HAL_Delay(200);
		  HAL_GPIO_WritePin(GPIOB, LD2_Pin, GPIO_PIN_RESET);
		  HAL_Delay(200);
	  }
	  btn_press = 0;
#if (CONTINUOUS_ACQ == 1)
	  while (!btn_press) {
		  //start_cycle_count();
		  acquire_and_send_packet();
		  //stop_cycle_count("End Acquire Send Packet");

	  }
	  btn_press = 0;
#elif (CONTINUOUS_ACQ == 0)
	  acquire_and_send_packet();
#else
#error "Wrong value for CONTINUOUS_ACQ."
#endif
	}
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  DEBUG_PRINT("Hello\n");
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_8, GPIO_PIN_RESET);
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */


  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */
  MX_LPUART1_UART_Init();
  RetargetInit(&hlpuart1);
  DEBUG_PRINT("Hello world\r\n");


  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_SPI1_Init();
  MX_TIM3_Init();
  MX_ADC1_Init();
  MX_AES_Init();
  /* USER CODE BEGIN 2 */
  //Test encryption
  //HAL_CRYP_AESCBC_Encrypt(&hcryp, input_data, 4, encrypted_data, 1000);
  //HAL_CRYP_AESCBC_Decrypt(&hcryp, encrypted_data, 4, decrypted_data, 1000);


  //tag_cbc_mac_hardware(tag_hw, test_msg_hw, test_msg_len_hw);
  //tag_cbc_mac(tag, test_msg, test_msg_len);










#if ENABLE_RADIO
  // Enable S2LP Radio
  HAL_StatusTypeDef err = S2LP_Init(&hspi1);
  if (err)  {
	  DEBUG_PRINT("[S2LP] Error while initializing: %u\r\n", err);
	  Error_Handler();
  } else {
	  DEBUG_PRINT("[S2LP] Init OK\r\n");

  }
#endif

  if (HAL_ADCEx_Calibration_Start(&hadc1, ADC_SINGLE_ENDED) != HAL_OK) {
	  DEBUG_PRINT("Error while calibrating the ADC\r\n");
	  Error_Handler();
  }
  if (HAL_TIM_Base_Start(&htim3) != HAL_OK) {
	  DEBUG_PRINT("Error while enabling timer TIM3\r\n");
	  Error_Handler();
  }

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
#if (RUN_CONFIG == MAIN_APP)
  run();
#elif (RUN_CONFIG == EVAL_RADIO)
  eval_radio();
#else
#error "Wrong value for RUN_CONFIG."
#endif

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_MSI;
  RCC_OscInitStruct.MSIState = RCC_MSI_ON;
  RCC_OscInitStruct.MSICalibrationValue = 0;
  RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_10;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_MSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV16;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  __disable_irq();
  DEBUG_PRINT("Entering error Handler\r\n");
  while (1)
  {
	  // Blink LED3 (red)
	  HAL_GPIO_WritePin(GPIOB, LD3_Pin, GPIO_PIN_SET);
	  for (volatile int i=0; i < SystemCoreClock/200; i++);
	  HAL_GPIO_WritePin(GPIOB, LD3_Pin, GPIO_PIN_RESET);
	  for (volatile int i=0; i < SystemCoreClock/200; i++);
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: DEBUG_PRINT("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
