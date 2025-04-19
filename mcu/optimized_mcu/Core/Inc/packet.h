/*
 * packet.h
 */

#ifndef INC_PACKET_H_
#define INC_PACKET_H_

#include <stdint.h>
#include <stdlib.h>


#define PACKET_HEADER_LENGTH (1+1+2+4)
#define PACKET_TAG_LENGTH 16

#if SEND_8BIT_MELS|COMPUTE_8BIT_MELS
	#define PAYLOAD_LENGTH (sizeof(q7_t) * N_MELVECS * MELVEC_LENGTH)
#else
	#define PAYLOAD_LENGTH (sizeof(q15_t) * N_MELVECS * MELVEC_LENGTH)
#endif

#define PACKET_LENGTH (PACKET_HEADER_LENGTH + PAYLOAD_LENGTH + PACKET_TAG_LENGTH)

int make_packet(uint8_t *packet, size_t payload_len, uint8_t sender_id, uint32_t serial);

#endif /* INC_PACKET_H_ */
