/*
 * packet.c
 */

#include "aes_ref.h"
#include "config.h"
#include "packet.h"
#include "main.h"
#include "utils.h"
#include "aes.h"

const uint8_t AES_Key[16]  = {
                            0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00};

// DMA buffers
uint8_t padded_msg_dma[1024]; // Adjust size based on message length
uint8_t state_dma[1024];      // Adjust size based on message length

void tag_cbc_mac(uint8_t *tag, const uint8_t *msg, size_t msg_len) {
	start_cycle_count();
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
    stop_cycle_count("Hardware encryption");
}



int make_packet(uint8_t *packet, size_t payload_len, uint8_t sender_id, uint32_t serial) {
	// Assumes payload is already in place in the packet
	// Packet structure :
	/***************************************************************************
	 *    Field       	Length (bytes)      Encoding        Description
	 ***************************************************************************
	 *  r 					1 								Reserved, set to 0.
	 * 	emitter_id 			1 					BE 			Unique id of the sensor node.
	 *	payload_length 		2 					BE 			Length of app_data (in bytes).
	 *	packet_serial 		4 					BE 			Unique and incrementing id of the packet.
	 *	app_data 			any 							The feature vectors.
	 *	tag 				16 								Message authentication code (MAC).
	 *
	 *	Note : BE refers to Big endian
	 *		 	Use the structure 	packet[x] = y; 	to set a byte of the packet buffer
	 *		 	To perform bit masking of the specific bytes you want to set, you can use
	 *		 		- bitshift operator (>>),
	 *		 		- and operator (&) with hex value, e.g.to perform 0xFF
	 *		 	This will be helpful when setting fields that are on multiple bytes.
	*/

    size_t packet_len = payload_len + PACKET_HEADER_LENGTH + PACKET_TAG_LENGTH;

	packet[0] = 0x0; // Reserved

	packet[1] = sender_id; // Emitter ID

	packet[2] = (payload_len >> 8) & 0xFF; // Payload length MSB
	packet[3] = payload_len & 0xFF; // Payload length LSB

	packet[4] = (serial >> 24) & 0xFF; // Packet serial MSB
	packet[5] = (serial >> 16) & 0xFF; // Packet serial
	packet[6] = (serial >> 8) & 0xFF; // Packet serial
	packet[7] = serial & 0xFF; // Packet serial

	//Authentication tag
    tag_cbc_mac(packet + payload_len + PACKET_HEADER_LENGTH, packet, payload_len + PACKET_HEADER_LENGTH);

    return packet_len;
}
