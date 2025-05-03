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

/*
void tag_cbc_mac(uint8_t *tag, const uint8_t *msg, size_t msg_len) {
	// Allocate a buffer of the key size to store the input and result of AES
	// uint32_t[4] is 4*(32/8)= 16 bytes long
	start_cycle_count();

	uint32_t statew[4] = {0}; // 16-byte buffer initialized to zero
	// state is a pointer to the start of the buffer
	//It's initialized to zero, meaning the first block of CBC (IV) is all zeros.
	uint8_t *state = (uint8_t*) statew; // Pointer to the 16-byte buffer

    size_t i;


	int n = msg_len/16; //Calculates how many full 16-byte blocks exist in msg.
	if (msg_len%16 != 0){ //If there's a remaining (incomplete) block, we add one more block.
		n++;
	}

	for(i=0;i<n;i++){ // Iterate over message blocks
		for (int j = 0; j < 16; j++)
		{
			if(i*16 + j < msg_len) // Check if we're still inside msg length
			{
				state[j] = state[j] ^ msg[i*16 + j]; // XOR message block into state
			}

		}
		AES128_encrypt(state,AES_Key);

	}



    // Copy the result of CBC-MAC-AES to the tag.
    for (int j=0; j<16; j++) {
        tag[j] = state[j];
    }
    /*

	uint32_t statew[4] = {0}; // 16-byte buffer for CBC-MAC result
	uint8_t *state = (uint8_t*)statew; //Pointer to state buffer

	// Process message using AES-CBC
	if (HAL_CRYP_AESCBC_Encrypt(&hcryp, (uint32_t*)msg, msg_len/4, (uint32_t*)state, 1000) != HAL_OK){
		DEBUG_PRINT("AES encryption error\n");
	}

	//The last block of the encrypted data output is the CBC-MAC
	for (size_t i = 0; i < 16; i++){
		tag[i] = state[i];
	}

    //uint32_t cycles_elapsed = stop_cycle_count("Software encryption");
}
*/

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
    uint32_t cycles_elapsed = stop_cycle_count("Hardware encryption");
}


// Assumes payload is already in place in the packet
int make_packet(uint8_t *packet, size_t payload_len, uint8_t sender_id, uint32_t serial) {
    size_t packet_len = payload_len + PACKET_HEADER_LENGTH + PACKET_TAG_LENGTH;
    // Initially, the whole packet header is set to 0s
    //memset(packet, 0, PACKET_HEADER_LENGTH);
    // So is the tag
	//memset(packet + payload_len + PACKET_HEADER_LENGTH, 0, PACKET_TAG_LENGTH);

	packet[0] = 0x0; // Reserved

	packet[1] = sender_id; // Emitter ID

	packet[2] = (payload_len >> 8) & 0xFF; // Payload length MSB
	packet[3] = payload_len & 0xFF; // Payload length LSB

	packet[4] = (serial >> 24) & 0xFF; // Packet serial MSB
	packet[5] = (serial >> 16) & 0xFF; // Packet serial
	packet[6] = (serial >> 8) & 0xFF; // Packet serial
	packet[7] = serial & 0xFF; // Packet serial


	// TO DO :  replace the two previous command by properly
	//			setting the packet header with the following structure :
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

	// For the tag field, you have to calculate the tag. The function call below is correct but
	// tag_cbc_mac function, calculating the tag, is not implemented.
    tag_cbc_mac(packet + payload_len + PACKET_HEADER_LENGTH, packet, payload_len + PACKET_HEADER_LENGTH);

    return packet_len;
}
