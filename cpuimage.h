#ifndef _CPUIMAGE_HEADER_
#define _CPUIMAGE_HEADER_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "fastmath.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER) && (_MSC_VER >= 1310) /*Visual Studio: A few warning types are not desired here.*/
#pragma warning(disable : 4996)             /*VS does not like fopen, but fopen_s is not standard C so unusable here*/
#endif                                      /*_MSC_VER */

#ifndef clamp
#define clamp(value, min, max) ((value) > (max) ? (max) : (value) < (min) ? (min) : (value))
#endif
#ifndef ClampToByte
#define ClampToByte(v) (unsigned char)(((unsigned)(int)(v)) < (255) ? (v) : (v < 0) ? (0) : (255))
#endif
#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef min3
#define min3(a, b, c) min(min((a), (b)), (c))
#endif
#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef max3
#define max3(a, b, c) max(max((a), (b)), (c))
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

typedef struct
{
    int levelMinimum;
    int levelMiddle;
    int levelMaximum;
    int minOutput;
    int maxOutput;
    bool Enable;
} cpuLevelParams;

void rgb2yiq(unsigned char *R, unsigned char *G, unsigned char *B, short *Y, short *I, short *Q);

void yiq2rgb(short *Y, short *I, short *Q, unsigned char *R, unsigned char *G, unsigned char *B);

void rgb2hsv(const unsigned char *R, const unsigned char *G, const unsigned char *B, unsigned char *H, unsigned char *S,
             unsigned char *V);

void hsv2rgb(const unsigned char *H, const unsigned char *S, const unsigned char *V, unsigned char *R, unsigned char *G,
             unsigned char *B);

void rgb2ycbcr(unsigned char R, unsigned char G, unsigned char B, unsigned char *y, unsigned char *cb, unsigned char *cr);

void ycbcr2rgb(unsigned char y, unsigned char Cb, unsigned char Cr, unsigned char *R, unsigned char *G, unsigned char *B);

//--------------------------Color adjustments--------------------------
void CPUImageGrayscaleFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride);

// float redAdjustment = 1.0f, float greenAdjustment = 1.0f, float blueAdjustment = 1.0f
void CPUImageRGBFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float redAdjustment,
                       float greenAdjustment, float blueAdjustment);

// float thresholdMultiplier = 1.0f
void CPUImageAverageLuminanceThresholdFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride,
                                             float thresholdMultiplier);

void CPUImageAverageColor(unsigned char *Input, int Width, int Height, int Stride, unsigned char *AverageR,
                          unsigned char *AverageG, unsigned char *AverageB, unsigned char *AverageA);

void CPUImageLuminosity(unsigned char *Input, int Width, int Height, int Stride, unsigned char *Luminance);

// float intensity = 1.0f
void CPUImageColorMatrixFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride,
                               float *colorMatrix, float intensity);

//int intensity = 100
void CPUImageSepiaFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, int intensity);

// unsigned char colorToReplaceR = 0, unsigned char colorToReplaceG = 160, unsigned char colorToReplaceB = 0, float thresholdSensitivity = 0.2f, float smoothing = 0.1f
void CPUImageChromaKeyFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride,
                             unsigned char colorToReplaceR, unsigned char colorToReplaceG,
                             unsigned char colorToReplaceB, float thresholdSensitivity, float smoothing);

// int intensity = 100
void CPUImageLookupFilter(unsigned char *Input, unsigned char *Output, unsigned char *lookupTable, int Width, int Height,
                          int Stride, int intensity);

// float saturation = 1.0
void CPUImageSaturationFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride,
                              float saturation);

// float gamma = 1.0f
void CPUImageGammaFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float gamma);

//  float contrast = 1.0f
void CPUImageContrastFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float contrast);

//float exposure = 0.0f
void CPUImageExposureFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float exposure);

//int brightness = 0.0f
void CPUImageBrightnessFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride,
                              int brightness);

//unsigned char firstColorR = 0, unsigned char firstColorG = 0, unsigned char firstColorB = 0.5 * 255, unsigned char secondColorR = 1.0f * 255, unsigned char secondColorG = 0, unsigned char secondColorB = 0, int intensity = 100
void CPUImageFalseColorFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride,
                              unsigned char firstColorR, unsigned char firstColorG, unsigned char firstColorB,
                              unsigned char secondColorR, unsigned char secondColorG, unsigned char secondColorB,
                              int intensity);

// float distance = 0.3, float slope = 0, int intensity = 100
void CPUImageHazeFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float distance,
                        float slope, int intensity);

// float opacity = 1.0f
void CPUImageOpacityFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float opacity);

void CPUImageLevelsFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride,
                          cpuLevelParams *redLevelParams, cpuLevelParams *greenLevelParams,
                          cpuLevelParams *blueLevelParams);

// float hueAdjust = 90.0f
void CPUImageHueFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float hueAdjust);

// float shadowTintR = 1.0f, float shadowTintG = 0.0f, float shadowTintB = 0.0f, float highlightTintR = 0.0f, float highlightTintG = 0.0f, float highlightTintB = 1.0f, float shadowTintIntensity = 0.0f, float highlightTintIntensity = 0.0f
void CPUImageHighlightShadowTintFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride,
                                       float shadowTintR, float shadowTintG, float shadowTintB, float highlightTintR,
                                       float highlightTintG, float highlightTintB, float shadowTintIntensity,
                                       float highlightTintIntensity);

//  float shadows = 0.0f, float highlights = 1.0f
void CPUImageHighlightShadowFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride,
                                   float shadows, float highlights);

//  unsigned char filterColorR = 0.6 * 255, unsigned char filterColorG = 0.45 * 255, unsigned char filterColorB = 0.3 * 255, int intensity = 100
void CPUImageMonochromeFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride,
                              unsigned char filterColorR, unsigned char filterColorG, unsigned char filterColorB,
                              int intensity);

void CPUImageColorInvertFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride);

// unsigned char colorAlpha = 255
void CPUImageSolidColorGenerator(unsigned char *Output, int Width, int Height, int Stride, unsigned char colorR,
                                 unsigned char colorG, unsigned char colorB, unsigned char colorAlpha);

// unsigned char threshold = 127
void CPUImageLuminanceThresholdFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride,
                                      unsigned char threshold);

// float temperature = 5000, float tint = 0
void CPUImageWhiteBalanceFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride,
                                float temperature, float tint);

//float vibrance = 1.2
void CPUImageVibranceFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float vibrance);

// float skinToneAdjust = 0.3f, float skinHue = 0.05f, float skinHueThreshold = 80.0f, float maxHueShift = 0.25f, float maxSaturationShift = 0.4f, int upperSkinToneColor = 0
void CPUImageSkinToneFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride,
                            float skinToneAdjust, float skinHue, float skinHueThreshold, float maxHueShift,
                            float maxSaturationShift, int upperSkinToneColor);

//float fraction = 0.05f
void CPUImageAutoLevel(const unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float fraction);

//--------------------------Color adjustments--------------------------

//--------------------------Image processing--------------------------
void CPUImageGaussianBlurFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride,
                                float GaussianSigma);

// float GaussianSigma = 4, int intensity = 100
void CPUImageUnsharpMaskFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride,
                               float GaussianSigma, int intensity);

//int Radius = 3
void CPUImageBoxBlurFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, int Radius);

// float Radius = 4, int sharpness = 1, int intensity = 100
void CPUImageSharpenFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float Radius,
                           int sharpness, int intensity);

void CPUImageResamplingFilter(unsigned char *Input, unsigned int Width, unsigned int Height, unsigned int Stride,
                              unsigned char *Output, int newWidth, int newHeight, int dstStride);

void CPUImageCropFilter(const unsigned char *Input, int Width, int Height, int srcStride, unsigned char *Output, int cropX,
                        int cropY, int dstWidth, int dstHeight, int dstStride);

void CPUImageSobelEdge(unsigned char *Input, unsigned char *Output, int Width, int Height);

int CPUImageHoughLines(unsigned char *Input, int Width, int Height, int lineIntensity, int Threshold, float resTheta,
                       int numLine, float *Radius, float *Theta);

void CPUImageDrawLine(unsigned char *canvas, int width, int height, int stride, int x1, int y1, int x2, int y2,
                      unsigned char R, unsigned char G, unsigned char B);
//--------------------------Image processing--------------------------

//--------------------------preImage processing--------------------------
//ref:https://github.com/scardine/image_size
bool CPUImageGetImageSize(const char *file_path, int *width, int *height, int *file_size);
//--------------------------preImage processing--------------------------

#ifdef __cplusplus
}
#endif
#endif