#ifndef _CPUIMAGE_FASTMATH_HEADER_
#define _CPUIMAGE_FASTMATH_HEADER_
#include <stdlib.h>
#include <math.h>
 
static inline float fastSin(
	float x) {
    float y;
    x *= -0.318309886f;
    y = x + 25165824.0f;
    x -= y - 25165824.0f;
    x *= (x < 0 ? -x : x) - 1;
    return x * (3.5841304553896f * (x < 0 ? -x : x) + 3.1039673861526f);
} 

static inline float fastCos(
	float x)
{
    return PT_sinf(x + 1.570796327f);
}
#endif