/*
 * config.h
 */

#ifndef INC_CONFIG_H_
#define INC_CONFIG_H_

#include <stdio.h>


#define SAMPLES_PER_MELVEC 32700



// Enable debug print
#define DEBUGP 0

#if (DEBUGP == 1)
#define DEBUG_PRINT(...) do{ printf(__VA_ARGS__ ); } while( 0 )
#else
#define DEBUG_PRINT(...) do{ } while ( 0 )
#endif



#endif /* INC_CONFIG_H_ */
