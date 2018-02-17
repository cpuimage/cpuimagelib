#include <math.h>
#include <string.h>
#include <sys/stat.h>
#ifdef __cplusplus
extern "C" {
#endif
#include "cpuimage.h"
/*

	LevelParams redLevelParams = {
	//色阶最小值
	0,
	//色阶中间值
	127,
	//色阶最大值
	255,
	//最小输出值
	0,
	//最大输出值
	255,
	//是否应用
	true,
	};
	LevelParams greenLevelParams = {
	//色阶最小值
	0,
	//色阶中间值
	127,
	//色阶最大值
	255,
	//最小输出值
	0,
	//最大输出值
	255,
	//是否应用
	true,
	};

	LevelParams blueLevelParams = {
	//色阶最小值
	0,
	//色阶中间值
	127,
	//色阶最大值
	255,
	//最小输出值
	0,
	//最大输出值
	255,
	//是否应用
	true,
	};*/

static inline unsigned long byteswap_ulong(unsigned long i)
{
	unsigned int j;
	j = (i << 24);
	j += (i << 8) & 0x00FF0000;
	j += (i >> 8) & 0x0000FF00;
	j += (i >> 24);
	return j;
}

static inline unsigned short byteswap_ushort(unsigned short i)
{
	unsigned short j;
	j = (i << 8);
	j += (i >> 8);
	return j;
}

static inline unsigned char step(unsigned char edge, unsigned char x)
{
	return (unsigned char)(x < edge ? 0 : 255);
}

static inline int Abs(int x)
{
	return (x ^ (x >> 31)) - (x >> 31);
}

unsigned char mix_u8(unsigned char a, unsigned char b, float alpha)
{
	return (unsigned char)ClampToByte(a * (1.0f - alpha) + b * alpha);
}

float dot(unsigned char R, unsigned char G, unsigned char B, float fR, float fG, float fB)
{
	return (float)(R * fR + G * fG + B * fB);
}

static inline float mix(float a, float b, float alpha) { return (a * (1.0f - alpha) + b * alpha); }

unsigned char degree(unsigned char InputColor, unsigned char OutputColor, float intensity)
{
	return (unsigned char)ClampToByte(((intensity * OutputColor) + (1.0f - intensity) * InputColor));
}

float smoothstep(float edgeMin, float edgeMax, float x)
{
	return clamp((x - edgeMin) / (edgeMax - edgeMin), 0.0f, 1.0f);
}

float vec2_distance(float vecX, float vecY, float otherX, float otherY)
{
	float dx = vecX - otherX;
	float dy = vecY - otherY;
	return sqrtf(dx * dx + dy * dy);
}

void CPUImageGrayscaleFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride)
{
	int Channels = Stride / Width;

	const int B_WT = (int)(0.114 * 256 + 0.5);
	const int G_WT = (int)(0.587 * 256 + 0.5);
	const int R_WT = 256 - B_WT - G_WT; //     int(0.299 * 256 + 0.5);
	int Channel = Stride / Width;
	if (Channel == 3)
	{
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *LinePS = Input + Y * Stride;
			unsigned char *LinePD = Output + Y * Width;
            int X = 0;
            for (; X < Width - 4; X += 4, LinePS += Channel * 4) {
                LinePD[X + 0] = (unsigned char) ((B_WT * LinePS[0] + G_WT * LinePS[1] + R_WT * LinePS[2]) >> 8);
                LinePD[X + 1] = (unsigned char) ((B_WT * LinePS[3] + G_WT * LinePS[4] + R_WT * LinePS[5]) >> 8);
                LinePD[X + 2] = (unsigned char) ((B_WT * LinePS[6] + G_WT * LinePS[7] + R_WT * LinePS[8]) >> 8);
                LinePD[X + 3] = (unsigned char) ((B_WT * LinePS[9] + G_WT * LinePS[10] + R_WT * LinePS[11]) >> 8);
            }
            for (; X < Width; X++, LinePS += Channel) {
                LinePD[X] = (unsigned char) (B_WT * LinePS[0] + G_WT * LinePS[1] + R_WT * LinePS[2]) >> 8;
            }
        }
    } else if (Channel == 4) {
        for (int Y = 0; Y < Height; Y++) {
            unsigned char *LinePS = Input + Y * Stride;
            unsigned char *LinePD = Output + Y * Width;
            int X = 0;
            for (; X < Width - 4; X += 4, LinePS += Channel * 4) {
                LinePD[X + 0] = (unsigned char) ((B_WT * LinePS[0] + G_WT * LinePS[1] + R_WT * LinePS[2]) >> 8);
                LinePD[X + 1] = (unsigned char) ((B_WT * LinePS[4] + G_WT * LinePS[5] + R_WT * LinePS[6]) >> 8);
                LinePD[X + 2] = (unsigned char) ((B_WT * LinePS[8] + G_WT * LinePS[9] + R_WT * LinePS[10]) >> 8);
                LinePD[X + 3] = (unsigned char) ((B_WT * LinePS[12] + G_WT * LinePS[13] + R_WT * LinePS[14]) >> 8);
            }
            for (; X < Width; X++, LinePS += Channel) {
                LinePD[X] = (unsigned char) ((B_WT * LinePS[0] + G_WT * LinePS[1] + R_WT * LinePS[2]) >> 8);
            }
        }
	}
	else if (Channel == 1)
	{
		if (Output != Input)
		{
			memcpy(Output, Input, Height * Stride);
		}
	}
}

/*
	CPUImageRGBFilter: Adjusts the individual RGB channels of an image
	red,green,blue : Normalized values by which each color channel is multiplied.The range is from 0.0 up, with 1.0 as the default.
	*/
void CPUImageRGBFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float redAdjustment, float greenAdjustment, float blueAdjustment)
{
	int Channels = Stride / Width;
	if (Channels == 1)
		return;
	unsigned char AdjustMapR[256] = {0};
	unsigned char AdjustMapG[256] = {0};
	unsigned char AdjustMapB[256] = {0};
	for (int pixel = 0; pixel < 256; pixel++)
	{
		AdjustMapR[pixel] = (unsigned char)(pixel * redAdjustment);
		AdjustMapG[pixel] = (unsigned char)(pixel * greenAdjustment);
		AdjustMapB[pixel] = (unsigned char)(pixel * blueAdjustment);
	}
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			pOutput[0] = AdjustMapR[pInput[0]];
			pOutput[1] = AdjustMapG[pInput[1]];
			pOutput[2] = AdjustMapB[pInput[2]];
			pInput += Channels;
			pOutput += Channels;
		}
	}
}
/*
	CPUImageAverageLuminanceThresholdFilter: This applies a thresholding operation where the threshold is continually adjusted based on the average luminance of the scene.
	thresholdMultiplier : This is a factor that the average luminance will be multiplied by in order to arrive at the final threshold to use.By default, this is 1.0.
	*/

void CPUImageAverageLuminanceThresholdFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float thresholdMultiplier)
{
	int Channels = Stride / Width;
	unsigned char Luminance = 0;
	if (Channels == 1)
	{
        int numberOfPixels = Width * Height;
		unsigned int histogramGray[256] = {0};
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pInput = Input + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				histogramGray[pInput[0]]++;
				pInput += Channels;
			}
		}
		unsigned int SumPix = 0;
		for (unsigned int i = 0; i < 256; i++)
		{
			SumPix += histogramGray[i] * i;
		}
		Luminance = (unsigned char)((SumPix / numberOfPixels) * thresholdMultiplier);

		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pOutput = Output + (Y * Stride);
			unsigned char *pInput = Input + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				pOutput[0] = step(Luminance, pInput[0]);
				pInput++;
				pOutput++;
			}
		}
	}
	else if (Channels == 3 || Channels == 4)
	{

        int numberOfPixels = Width * Height;
		unsigned int histogramLum[256] = {0};
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pInput = Input + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				const unsigned char R = pInput[0];
				const unsigned char G = pInput[1];
				const unsigned char B = pInput[2];
				histogramLum[((13926 * R + 46884 * G + 4725 * B) >> 16)]++;
				pInput += Channels;
			}
		}
		unsigned int Sum = 0;
		for (unsigned int i = 0; i < 256; i++)
		{
			Sum += histogramLum[i] * i;
		}
		Luminance = (unsigned char)((Sum / numberOfPixels) * thresholdMultiplier);
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pOutput = Output + (Y * Stride);
			unsigned char *pInput = Input + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				unsigned char luminance =(unsigned char)((13926 * pInput[0] + 46884 * pInput[1] + 4725 * pInput[2]) >> 16);
				pOutput[2] = pOutput[1] = pOutput[0] = step(Luminance, luminance);
				pInput += Channels;
				pOutput += Channels;
			}
		}
	}
}
/*
	CPUImageAverageColor: This processes an Input image and determines the average color of the scene, by averaging the RGBA components for each pixel in the image.A reduction process is used to progressively downsample the source image on the GPU, followed by a short averaging calculation on the CPU.The Output from this filter is meaningless, but you need to set the colorAverageProcessingFinishedBlock property to a block that takes in four color components and a frame time and does something with them.
	Shader :
	------------------------------------------------------------------------

	GPUImageColorAveragingVertexShaderString = SHADER_STRING
	(
	attribute vec4 position;
	attribute vec4 InputTextureCoordinate;

	uniform float texelWidth;
	uniform float texelHeight;

	varying vec2 upperLeftInputTextureCoordinate;
	varying vec2 upperRightInputTextureCoordinate;
	varying vec2 lowerLeftInputTextureCoordinate;
	varying vec2 lowerRightInputTextureCoordinate;

	void main()
	{
	gl_Position = position;

	upperLeftInputTextureCoordinate = InputTextureCoordinate.xy + vec2(-texelWidth, -texelHeight);
	upperRightInputTextureCoordinate = InputTextureCoordinate.xy + vec2(texelWidth, -texelHeight);
	lowerLeftInputTextureCoordinate = InputTextureCoordinate.xy + vec2(-texelWidth, texelHeight);
	lowerRightInputTextureCoordinate = InputTextureCoordinate.xy + vec2(texelWidth, texelHeight);
	}
	);


	GPUImageColorAveragingFragmentShaderString = SHADER_STRING
	(
	precision highp float;

	uniform sampler2D InputImageTexture;

	varying highp vec2 OutputTextureCoordinate;

	varying highp vec2 upperLeftInputTextureCoordinate;
	varying highp vec2 upperRightInputTextureCoordinate;
	varying highp vec2 lowerLeftInputTextureCoordinate;
	varying highp vec2 lowerRightInputTextureCoordinate;

	void main()
	{
	highp vec4 upperLeftColor = texture2D(InputImageTexture, upperLeftInputTextureCoordinate);
	highp vec4 upperRightColor = texture2D(InputImageTexture, upperRightInputTextureCoordinate);
	highp vec4 lowerLeftColor = texture2D(InputImageTexture, lowerLeftInputTextureCoordinate);
	highp vec4 lowerRightColor = texture2D(InputImageTexture, lowerRightInputTextureCoordinate);

	gl_FragColor = 0.25 * (upperLeftColor + upperRightColor + lowerLeftColor + lowerRightColor);
	}
	);
	------------------------------------------------------------------------
	*/
//与gpuimage算法不一致
void CPUImageAverageColor(unsigned char *Input, int Width, int Height, int Stride, unsigned char *AverageR, unsigned char *AverageG, unsigned char *AverageB, unsigned char *AverageA)
{
	int Channels = Stride / Width;
	if (Channels == 1)
	{
        int numberOfPixels = Width * Height;
		unsigned int histogramGray[256] = {0};
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pInput = Input + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				histogramGray[pInput[0]]++;
				pInput += Channels;
			}
		}
		unsigned int Sum = 0;
		for (unsigned int i = 0; i < 256; i++)
		{
			Sum += histogramGray[i] * i;
		}
		*AverageR = (unsigned char)(Sum / numberOfPixels);
		*AverageG = *AverageR;
		*AverageB = *AverageR;
		*AverageA = *AverageR;
	}
	else if (Channels == 3)
	{

        int numberOfPixels = Width * Height;
		unsigned int histogramRGB[768] = {0};
		unsigned int *histogramR = &histogramRGB[0];
		unsigned int *histogramG = &histogramRGB[256];
		unsigned int *histogramB = &histogramRGB[512];
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pInput = Input + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				const unsigned char R = pInput[0];
				const unsigned char G = pInput[1];
				const unsigned char B = pInput[2];
				histogramR[R]++;
				histogramG[G]++;
				histogramB[B]++;
				pInput += Channels;
			}
		}
		unsigned int SumPixR = 0, SumPixG = 0, SumPixB = 0;
		for (unsigned int i = 0; i < 256; i++)
		{
			SumPixR += histogramR[i] * i;
			SumPixG += histogramG[i] * i;
			SumPixB += histogramB[i] * i;
		}
		*AverageR =(unsigned char) (SumPixR / numberOfPixels);
		*AverageG =(unsigned char)(SumPixG / numberOfPixels);
		*AverageB =(unsigned char) (SumPixB / numberOfPixels);
		*AverageA = 255;
	}
	else if (Channels == 4)
	{

        int numberOfPixels = Width * Height;
		unsigned int histogramRGB[768 + 256] = {0};
		unsigned int *histogramR = &histogramRGB[0];
		unsigned int *histogramG = &histogramRGB[256];
		unsigned int *histogramB = &histogramRGB[512];
		unsigned int *histogramA = &histogramRGB[768];
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pInput = Input + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				const unsigned char R = pInput[0];
				const unsigned char G = pInput[1];
				const unsigned char B = pInput[2];
				const unsigned char A = pInput[3];
				histogramR[R]++;
				histogramG[G]++;
				histogramB[B]++;
				histogramA[A]++;
				pInput += Channels;
			}
		}
		unsigned int SumPixR = 0, SumPixG = 0, SumPixB = 0, SumPixA = 0;
		for (unsigned int i = 0; i < 256; i++)
		{
			SumPixR += histogramR[i] * i;
			SumPixG += histogramG[i] * i;
			SumPixB += histogramB[i] * i;
			SumPixA += histogramA[i] * i;
		}
		*AverageR =(unsigned char) (SumPixR / numberOfPixels);
		*AverageG =(unsigned char) (SumPixG / numberOfPixels);
		*AverageB =(unsigned char) (SumPixB / numberOfPixels);
		*AverageA =(unsigned char) (SumPixA / numberOfPixels);
	}
}

/*
	CPUImageLuminosity: Like the CPUImageAverageColor, this reduces an image to its average luminosity.You need to set the luminosityProcessingFinishedBlock to handle the Output of this filter, which just returns a luminosity value and a frame time.


	Shader :
	------------------------------------------------------------------------
	precision highp float;

	uniform sampler2D InputImageTexture;

	varying highp vec2 OutputTextureCoordinate;

	varying highp vec2 upperLeftInputTextureCoordinate;
	varying highp vec2 upperRightInputTextureCoordinate;
	varying highp vec2 lowerLeftInputTextureCoordinate;
	varying highp vec2 lowerRightInputTextureCoordinate;

	const highp vec3 W = vec3(0.2125, 0.7154, 0.0721);

	void main()
	{
	highp float upperLeftLuminance = dot(texture2D(InputImageTexture, upperLeftInputTextureCoordinate).rgb, W);
	highp float upperRightLuminance = dot(texture2D(InputImageTexture, upperRightInputTextureCoordinate).rgb, W);
	highp float lowerLeftLuminance = dot(texture2D(InputImageTexture, lowerLeftInputTextureCoordinate).rgb, W);
	highp float lowerRightLuminance = dot(texture2D(InputImageTexture, lowerRightInputTextureCoordinate).rgb, W);

	highp float luminosity = 0.25 * (upperLeftLuminance + upperRightLuminance + lowerLeftLuminance + lowerRightLuminance);
	gl_FragColor = vec4(luminosity, luminosity, luminosity, 1.0);
	}


	precision highp float;

	uniform sampler2D InputImageTexture;

	varying highp vec2 OutputTextureCoordinate;

	varying highp vec2 upperLeftInputTextureCoordinate;
	varying highp vec2 upperRightInputTextureCoordinate;
	varying highp vec2 lowerLeftInputTextureCoordinate;
	varying highp vec2 lowerRightInputTextureCoordinate;

	void main()
	{
	highp float upperLeftLuminance = texture2D(InputImageTexture, upperLeftInputTextureCoordinate).r;
	highp float upperRightLuminance = texture2D(InputImageTexture, upperRightInputTextureCoordinate).r;
	highp float lowerLeftLuminance = texture2D(InputImageTexture, lowerLeftInputTextureCoordinate).r;
	highp float lowerRightLuminance = texture2D(InputImageTexture, lowerRightInputTextureCoordinate).r;

	highp float luminosity = 0.25 * (upperLeftLuminance + upperRightLuminance + lowerLeftLuminance + lowerRightLuminance);
	gl_FragColor = vec4(luminosity, luminosity, luminosity, 1.0);
	}
	------------------------------------------------------------------------
	*/

void CPUImageLuminosity(unsigned char *Input, int Width, int Height, int Stride, unsigned char *Luminance)
{
	int Channels = Stride / Width;
	if (Channels == 1)
	{
        int numberOfPixels = Width * Height;
		unsigned int histogramGray[256] = {0};
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pInput = Input + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				histogramGray[pInput[0]]++;
				pInput++;
			}
		}
		unsigned int Sum = 0;
		for (int i = 0; i < 256; i++)
		{
			Sum += histogramGray[i] * i;
		}
		*Luminance = (unsigned char)(Sum / numberOfPixels);
	}
	else if (Channels == 3 || Channels == 4)
	{

        int numberOfPixels = Width * Height;
		unsigned int histogramLum[256] = {0};
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pInput = Input + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				const unsigned char R = pInput[0];
				const unsigned char G = pInput[1];
				const unsigned char B = pInput[2];
				histogramLum[((13926 * R + 46884 * G + 4725 * B) >> 16)]++;
				pInput += Channels;
			}
		}
		unsigned int Sum = 0;
		for (unsigned int i = 0; i < 256; i++)
		{
			Sum += histogramLum[i] * i;
		}
		*Luminance = (unsigned char)(Sum / numberOfPixels);
	}
}
/*
	CPUImageColorMatrixFilter: Transforms the colors of an image by applying a matrix to them
	colorMatrix : A 4x4 matrix used to transform each color in an image
	intensity : The degree to which the new transformed color replaces the original color for each pixel


	Shader :
	------------------------------------------------------------------------

	varying highp vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;

	uniform lowp mat4 colorMatrix;
	uniform lowp float intensity;

	void main()
	{
	lowp vec4 textureColor = texture2D(InputImageTexture, textureCoordinate);
	lowp vec4 OutputColor = textureColor * colorMatrix;

	gl_FragColor = (intensity * OutputColor) + ((1.0 - intensity) * textureColor);
	}
	------------------------------------------------------------------------
	*/

void CPUImageColorMatrixFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float *colorMatrix, float intensity)
{
	int Channels = Stride / Width;
	if (Channels == 1)
		return;
	unsigned char degreeMap[256 * 256] = {0};
	for (int pixel = 0; pixel < 256; pixel++)
	{
		unsigned char *pDegreeMap = degreeMap + pixel * 256;
		for (int out = 0; out < 256; out++)
		{
			pDegreeMap[0] = degree((unsigned char)pixel, (unsigned char)out, intensity);
			pDegreeMap++;
		}
	}
	unsigned char colorMatrixMap[256 * 16] = {0};
	unsigned char *colorMatrix0 = &colorMatrixMap[0];
	unsigned char *colorMatrix1 = &colorMatrixMap[256];
	unsigned char *colorMatrix2 = &colorMatrixMap[256 * 2];
	unsigned char *colorMatrix3 = &colorMatrixMap[256 * 3];
	unsigned char *colorMatrix4 = &colorMatrixMap[256 * 4];
	unsigned char *colorMatrix5 = &colorMatrixMap[256 * 5];
	unsigned char *colorMatrix6 = &colorMatrixMap[256 * 6];
	unsigned char *colorMatrix7 = &colorMatrixMap[256 * 7];
	unsigned char *colorMatrix8 = &colorMatrixMap[256 * 8];
	unsigned char *colorMatrix9 = &colorMatrixMap[256 * 9];
	unsigned char *colorMatrix10 = &colorMatrixMap[256 * 10];
	unsigned char *colorMatrix11 = &colorMatrixMap[256 * 11];
	unsigned char *colorMatrix12 = &colorMatrixMap[256 * 12];
	unsigned char *colorMatrix13 = &colorMatrixMap[256 * 13];
	unsigned char *colorMatrix14 = &colorMatrixMap[256 * 14];
	unsigned char *colorMatrix15 = &colorMatrixMap[256 * 15];
	for (int pixel = 0; pixel < 256; pixel++)
	{
		colorMatrix0[pixel] = ClampToByte(pixel * colorMatrix[0]);
		colorMatrix1[pixel] = ClampToByte(pixel * colorMatrix[1]);
		colorMatrix2[pixel] = ClampToByte(pixel * colorMatrix[2]);
		colorMatrix3[pixel] = ClampToByte(pixel * colorMatrix[3]);
		colorMatrix4[pixel] = ClampToByte(pixel * colorMatrix[4]);
		colorMatrix5[pixel] = ClampToByte(pixel * colorMatrix[5]);
		colorMatrix6[pixel] = ClampToByte(pixel * colorMatrix[6]);
		colorMatrix7[pixel] = ClampToByte(pixel * colorMatrix[7]);
		colorMatrix8[pixel] = ClampToByte(pixel * colorMatrix[8]);
		colorMatrix9[pixel] = ClampToByte(pixel * colorMatrix[9]);
		colorMatrix10[pixel] = ClampToByte(pixel * colorMatrix[10]);
		colorMatrix11[pixel] = ClampToByte(pixel * colorMatrix[11]);
		colorMatrix12[pixel] = ClampToByte(pixel * colorMatrix[12]);
		colorMatrix13[pixel] = ClampToByte(pixel * colorMatrix[13]);
		colorMatrix14[pixel] = ClampToByte(pixel * colorMatrix[14]);
		colorMatrix15[pixel] = ClampToByte(pixel * colorMatrix[15]);
	}
	if (Channels == 4)
	{
		unsigned char outR, outG, outB, outA;
		int WidthStep = Width * Channels;
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pOutput = Output + (Y * WidthStep);
			unsigned char *pInput = Input + (Y * WidthStep);
			for (int X = 0; X < Width; X++)
			{
				const unsigned char r = pInput[0];
				const unsigned char g = pInput[1];
				const unsigned char b = pInput[2];
				const unsigned char a = pInput[3];
				outR = ClampToByte(colorMatrix0[r] + colorMatrix1[g] + colorMatrix2[b] + colorMatrix3[a]);
				unsigned char *pDegreeMapR = degreeMap + (r << 8);
				pOutput[0] = pDegreeMapR[outR];
				outG = ClampToByte(colorMatrix4[r] + colorMatrix5[g] + colorMatrix6[b] + colorMatrix7[a]);
				unsigned char *pDegreeMapG = degreeMap + (g << 8);
				pOutput[1] = pDegreeMapG[outG];
				outB = ClampToByte(colorMatrix8[r] + colorMatrix9[g] + colorMatrix10[b] + colorMatrix11[a]);
				unsigned char *pDegreeMapB = degreeMap + (b << 8);
				pOutput[2] = pDegreeMapB[outB];
				outA = ClampToByte(colorMatrix12[r] + colorMatrix13[g] + colorMatrix14[b] + colorMatrix15[a]);
				unsigned char *pDegreeMapA = degreeMap + (a << 8);
				pOutput[3] = pDegreeMapA[outA];
				pInput += Channels;
				pOutput += Channels;
			}
		}
	}
	else if (Channels == 3)
	{
		//三个通道则，a为255不透明
		unsigned char outR, outG, outB;
		int WidthStep = Width * Channels;
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pOutput = Output + (Y * WidthStep);
			unsigned char *pInput = Input + (Y * WidthStep);
			for (int X = 0; X < Width; X++)
			{
				const unsigned char r = pInput[0];
				const unsigned char g = pInput[1];
				const unsigned char b = pInput[2];
				outR = ClampToByte(colorMatrix0[r] + colorMatrix1[g] + colorMatrix2[b]);
				unsigned char *pDegreeMapR = degreeMap + (r << 8);
				pOutput[0] = pDegreeMapR[outR];
				outG = ClampToByte(colorMatrix4[r] + colorMatrix5[g] + colorMatrix6[b]);
				unsigned char *pDegreeMapG = degreeMap + (g << 8);
				pOutput[1] = pDegreeMapG[outG];
				outB = ClampToByte(colorMatrix8[r] + colorMatrix9[g] + colorMatrix10[b]);
				unsigned char *pDegreeMapB = degreeMap + (b << 8);
				pOutput[2] = pDegreeMapB[outB];
				pInput += Channels;
				pOutput += Channels;
			}
		}
	}
}

/*
	CPUImageSepiaFilter: Simple sepia tone filter
	intensity : The degree to which the sepia tone replaces the normal image color(0.0 - 1.0, with 1.0 as the default)
	*/

void CPUImageSepiaFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, int intensity)
{
	int Channels = Stride / Width;
	if (Channels == 1)
		return;
	float fIntensity = intensity / 100.0f;

	float colorMatrix[4 * 4] = {
		0.3588f, 0.7044f, 0.1368f, 0.0f,
		0.2990f, 0.5870f, 0.1140f, 0.0f,
		0.2392f, 0.4696f, 0.0912f, 0.0f,
		0.f, 0.f, 0.f, 1.f};
	CPUImageColorMatrixFilter(Input, Output, Width, Height, Stride, colorMatrix, fIntensity);
}
/*

	CPUImageChromaKeyFilter: For a given color in the image, sets the alpha channel to 0. This is similar to the GPUImageChromaKeyBlendFilter, only instead of blending in a second image for a matching color this doesn't take in a second image and just turns a given color transparent.
	thresholdSensitivity : How close a color match needs to exist to the target color to be replaced(default of 0.4)
	smoothing : How smoothly to blend for the color match(default of 0.1)


	Shader :
	------------------------------------------------------------------------
	// Shader code based on Apple's CIChromaKeyFilter example: https://developer.apple.com/library/mac/#samplecode/CIChromaKeyFilter/Introduction/Intro.html
	(
	precision highp float;

	varying highp vec2 textureCoordinate;

	uniform float thresholdSensitivity;
	uniform float smoothing;
	uniform vec3 colorToReplace;
	uniform sampler2D InputImageTexture;
	uniform sampler2D InputImageTexture2;

	void main()
	{
	vec4 textureColor = texture2D(InputImageTexture, textureCoordinate);

	float maskY = 0.2989 * colorToReplace.r + 0.5866 * colorToReplace.g + 0.1145 * colorToReplace.b;
	float maskCr = 0.7132 * (colorToReplace.r - maskY);
	float maskCb = 0.5647 * (colorToReplace.b - maskY);

	float Y = 0.2989 * textureColor.r + 0.5866 * textureColor.g + 0.1145 * textureColor.b;
	float Cr = 0.7132 * (textureColor.r - Y);
	float Cb = 0.5647 * (textureColor.b - Y);

	//     float blendValue = 1.0 - smoothstep(thresholdSensitivity - smoothing, thresholdSensitivity , abs(Cr - maskCr) + abs(Cb - maskCb));
	float blendValue = smoothstep(thresholdSensitivity, thresholdSensitivity + smoothing, distance(vec2(Cr, Cb), vec2(maskCr, maskCb)));
	gl_FragColor = vec4(textureColor.rgb, textureColor.a * blendValue);
	}
	);
	------------------------------------------------------------------------

	*/

void CPUImageChromaKeyFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, unsigned char colorToReplaceR, unsigned char colorToReplaceG, unsigned char colorToReplaceB, float thresholdSensitivity, float smoothing)
{
	int Channels = Stride / Width;
	if (Channels == 1)
	{
		return;
	}

	unsigned char maskY =(unsigned char) ((19589 * colorToReplaceR + 38443 * colorToReplaceG + 7504 * colorToReplaceB) >> 16);

	unsigned char maskCr =(unsigned char) ((46740 * (colorToReplaceR - maskY) >> 16) + 128);

	unsigned char maskCb =(unsigned char) ((37008 * (colorToReplaceB - maskY) >> 16) + 128);
	int iThresholdSensitivity = (int)(thresholdSensitivity * 255.0f);
	int iSmoothing = (int)(smoothing * 256);
	if (Channels == 3)
	{
		short blendMap[256 * 256] = {0};
		for (int Cr = 0; Cr < 256; Cr++)
		{
			short *pBlendMap = blendMap + (Cr << 8);
			for (int Cb = 0; Cb < 256; Cb++)
			{
				pBlendMap[Cb] = (short)(255.0f * smoothstep((float)iThresholdSensitivity, (float)iThresholdSensitivity + iSmoothing, vec2_distance((float)Cr, (float)Cb, (float)maskCr, (float)maskCb)));
			}
		}
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pOutput = Output + (Y * Stride);
			unsigned char *pInput = Input + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				const unsigned char R = pInput[0];
				const unsigned char G = pInput[1];
				const unsigned char B = pInput[2];
				unsigned char y = (unsigned char)((19589 * R + 38443 * G + 7504 * B) >> 16);
				unsigned char Cr = (unsigned char)((46740 * (R - y) >> 16) + 128);
				unsigned char Cb = (unsigned char)((37008 * (B - y) >> 16) + 128);
				//乘以255取得mask,不乘以255则排除mask
				short *pBlendMap = blendMap + (Cr << 8);
				const short blendValue = pBlendMap[Cb];
				pOutput[0] =(unsigned char) (255 - (R * blendValue));
				pOutput[1] =(unsigned char) (255 - (G * blendValue));
				pOutput[2] =(unsigned char) (255 - (B * blendValue));
				pInput += Channels;
				pOutput += Channels;
			}
		}
	}
	else if (Channels == 4)
	{
		unsigned char blendMap[256 * 256] = {0};
		for (int Cr = 0; Cr < 256; Cr++)
		{
			unsigned char *pBlendMap = blendMap + (Cr << 8);
			for (int Cb = 0; Cb < 256; Cb++)
			{
				pBlendMap[Cb] = (unsigned char)(255.0f * smoothstep((float)iThresholdSensitivity, (float)iThresholdSensitivity + iSmoothing, vec2_distance((float)Cr, (float)Cb, (float)maskCr, (float)maskCb)));
			}
		}
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pOutput = Output + (Y * Stride);
			unsigned char *pInput = Input + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				const unsigned char R = pInput[0];
				const unsigned char G = pInput[1];
				const unsigned char B = pInput[2];
				const unsigned char A = pInput[3];
				unsigned char y = (unsigned char)((19589 * R + 38443 * G + 7504 * B) >> 16);
				unsigned char Cr =(unsigned char)((46740 * (R - y) >> 16) + 128);
				unsigned char Cb = (unsigned char)((37008 * (B - y) >> 16) + 128);
				//直接处理透明通道
				unsigned char *pBlendMap = blendMap + (Cr << 8);
				const unsigned char blendValue = pBlendMap[Cb];
				pOutput[3] = ClampToByte(A * blendValue);
				pInput += Channels;
				pOutput += Channels;
			}
		}
	}
} /*
	CPUImageLookupFilter: Uses an RGB color lookup image to remap the colors in an image.First, use your favourite photo editing application to apply a filter to lookup.png from GPUImage / framework / Resources.For this to work properly each pixel color must not depend on other pixels(e.g.blur will not work).If you need a more complex filter you can create as many lookup tables as required.Once ready, use your new lookup.png file as a second Input for GPUImageLookupFilter.

	Shader :
	------------------------------------------------------------------------
	varying highp vec2 textureCoordinate;
	varying highp vec2 textureCoordinate2; // TODO: This is not used

	uniform sampler2D InputImageTexture;
	uniform sampler2D InputImageTexture2; // lookup texture

	uniform lowp float intensity;

	void main()
	{
	highp vec4 textureColor = texture2D(InputImageTexture, textureCoordinate);

	highp float blueColor = textureColor.b * 63.0;

	highp vec2 quad1;
	quad1.y = floor(floor(blueColor) / 8.0);
	quad1.x = floor(blueColor) - (quad1.y * 8.0);

	highp vec2 quad2;
	quad2.y = floor(ceil(blueColor) / 8.0);
	quad2.x = ceil(blueColor) - (quad2.y * 8.0);

	highp vec2 texPos1;
	texPos1.x = (quad1.x * 0.125) + 0.5 / 512.0 + ((0.125 - 1.0 / 512.0) * textureColor.r);
	texPos1.y = (quad1.y * 0.125) + 0.5 / 512.0 + ((0.125 - 1.0 / 512.0) * textureColor.g);

	highp vec2 texPos2;
	texPos2.x = (quad2.x * 0.125) + 0.5 / 512.0 + ((0.125 - 1.0 / 512.0) * textureColor.r);
	texPos2.y = (quad2.y * 0.125) + 0.5 / 512.0 + ((0.125 - 1.0 / 512.0) * textureColor.g);

	lowp vec4 newColor1 = texture2D(InputImageTexture2, texPos1);
	lowp vec4 newColor2 = texture2D(InputImageTexture2, texPos2);

	lowp vec4 newColor = mix(newColor1, newColor2, fract(blueColor));
	gl_FragColor = mix(textureColor, vec4(newColor.rgb, textureColor.w), intensity);
	}
	------------------------------------------------------------------------
	*/

//lookup 512*512*3
void CPUImageLookupFilter(unsigned char *Input, unsigned char *Output, unsigned char *lookupTable, int Width, int Height, int Stride, int intensity)
{
	int Channels = Stride / Width;
	if (Channels == 1)
		return;
	float preMap[256 * 5] = {0};
	float *pixelColorMap = &preMap[0];
	float *quad1yMap = &preMap[256];
	float *quad1xMap = &preMap[256 + 256];
	float *quad2yMap = &preMap[256 + 256 + 256];
	float *quad2xMap = &preMap[256 + 256 + 256 + 256];

	unsigned short fractMap[256] = {0};
	unsigned short sIntensity =(unsigned short) max(min(intensity, 100), 0);
	int c1 = 256 * (100 - sIntensity) / 100;
	int c2 = 256 * (100 - (100 - sIntensity)) / 100;

	for (int b = 0; b < 256; b++)
	{
		pixelColorMap[b] = b * (63.0f / 255.0f);
		fractMap[b] = (unsigned short)(256 * (pixelColorMap[b] - truncf(pixelColorMap[b])));
		quad1yMap[b] = floorf(floorf(pixelColorMap[b]) * (1.0f / 8.0f));
		quad1xMap[b] = floorf(pixelColorMap[b]) - (quad1yMap[b] * 8.0f);
		quad2yMap[b] = floorf(ceilf(pixelColorMap[b]) * (1.0f / 8.0f));
		quad2xMap[b] = ceilf(pixelColorMap[b]) - (quad2yMap[b] * 8.0f);
		quad1yMap[b] = quad1yMap[b] * 64.0f + 0.5f;
		quad2yMap[b] = quad2yMap[b] * 64.0f + 0.5f;
		quad1xMap[b] = quad1xMap[b] * 64.0f + 0.5f;
		quad2xMap[b] = quad2xMap[b] * 64.0f + 0.5f;
	}

	int lookupChannels = 3;
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			unsigned char R = pInput[0];
			unsigned char G = pInput[1];
			unsigned char B = pInput[2];
			float green = pixelColorMap[G];
			float red = pixelColorMap[R];
			unsigned char *pLineLookup1 = &lookupTable[(((int)(quad1yMap[B] + green) << 9) + (int)(quad1xMap[B] + red)) * lookupChannels];
			unsigned char *pLineLookup2 = &lookupTable[(((int)(quad2yMap[B] + green) << 9) + (int)(quad2xMap[B] + red)) * lookupChannels];
			unsigned short fractB = fractMap[B];
			pOutput[0] = (unsigned char) ((int)(R * c1 + ((*pLineLookup1++ * (256 - fractB) + *pLineLookup2++ * fractB) >> 8) * c2) >> 8);
			pOutput[1] = (unsigned char)((int)(G * c1 + ((*pLineLookup1++ * (256 - fractB) + *pLineLookup2++ * fractB) >> 8) * c2) >> 8);
			pOutput[2] = (unsigned char)((int)(B * c1 + ((*pLineLookup1++ * (256 - fractB) + *pLineLookup2++ * fractB) >> 8) * c2) >> 8);
			pInput += Channels;
			pOutput += Channels;
		}
	}
}

/*
	CPUImageSaturationFilter: Adjusts the saturation of an image
	saturation : The degree of saturation or desaturation to apply to the image(0.0 - 2.0, with 1.0 as the default)

	Shader :
	------------------------------------------------------------------------------------------ -
	varying highp vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;
	uniform lowp float saturation;

	// Values from "Graphics Shaders: Theory and Practice" by Bailey and Cunningham
	const mediump vec3 luminanceWeighting = vec3(0.2125, 0.7154, 0.0721);

	void main()
	{
	lowp vec4 textureColor = texture2D(InputImageTexture, textureCoordinate);
	lowp float luminance = dot(textureColor.rgb, luminanceWeighting);
	lowp vec3 greyScaleColor = vec3(luminance);

	gl_FragColor = vec4(mix(greyScaleColor, textureColor.rgb, saturation), textureColor.w);
	}
	------------------------------------------------------------------------------------------ -
	*/

void CPUImageSaturationFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float saturation)
{
	int Channels = Stride / Width;
	if (Channels == 1)
		return;
	unsigned char SaturationMap[256 * 256] = {0};
	for (int grey = 0; grey < 256; grey++)
	{
		unsigned char *pSaturationMap = SaturationMap + (grey << 8);
		for (int Input = 0; Input < 256; Input++)
		{
			pSaturationMap[0] = (unsigned char)((mix_u8((unsigned char)grey,(unsigned char) Input, saturation) + Input) * 0.5f);
			pSaturationMap++;
		}
	}
	//0.2125*256.0*256.0=13926.4
	//0.7154*256.0*256.0=46884.4544
	//0.0721*256.0*256.0=4725.1456
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			unsigned char *pSaturationMap = SaturationMap + (((13926 * pInput[0] + 46884 * pInput[1] + 4725 * pInput[2]) >> 16) << 8);
			pOutput[0] = pSaturationMap[pInput[0]];
			pOutput[1] = pSaturationMap[pInput[1]];
			pOutput[2] = pSaturationMap[pInput[2]];
			pInput += Channels;
			pOutput += Channels;
		}
	}
}

/*

	CPUImageGammaFilter: Adjusts the gamma of an image
	gamma : The gamma adjustment to apply(0.0 - 3.0, with 1.0 as the default)

	Shader :
	------------------------------------------------------------------------------------------ -
	varying highp vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;
	uniform lowp float gamma;

	void main()
	{
	lowp vec4 textureColor = texture2D(InputImageTexture, textureCoordinate);

	gl_FragColor = vec4(pow(textureColor.rgb, vec3(gamma)), textureColor.w);
	}
	------------------------------------------------------------------------------------------ -
	*/

void CPUImageGammaFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float gamma)
{
	int Channels = Stride / Width;
	if (Channels == 1)
		return;
	unsigned char GammasMap[256] = {0};
	for (int pixel = 0; pixel < 256; pixel++)
	{
		GammasMap[pixel] = ClampToByte(pow(pixel, gamma));
	}
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			pOutput[0] = GammasMap[pInput[0]];
			pOutput[1] = GammasMap[pInput[1]];
			pOutput[2] = GammasMap[pInput[2]];
			pInput += Channels;
			pOutput += Channels;
		}
	}
}

/*

	CPUImageContrastFilter: Adjusts the contrast of the image
	contrast : The adjusted contrast(0.0 - 4.0, with 1.0 as the default)


	Shader :
	------------------------------------------------------------------------------------------ -
	varying highp vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;
	uniform lowp float contrast;

	void main()
	{
	lowp vec4 textureColor = texture2D(InputImageTexture, textureCoordinate);

	gl_FragColor = vec4(((textureColor.rgb - vec3(0.5)) * contrast + vec3(0.5)), textureColor.w);
	}
	------------------------------------------------------------------------------------------ -
	*/
void CPUImageContrastFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float contrast)
{
	int Channels = Stride / Width;
	if (Channels == 1)
		return;
	unsigned char contrastMap[256] = {0};
	for (int pixel = 0; pixel < 256; pixel++)
	{
		contrastMap[pixel] = ClampToByte((pixel - 127) * contrast + 127);
	}
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			pOutput[0] = contrastMap[pInput[0]];
			pOutput[1] = contrastMap[pInput[1]];
			pOutput[2] = contrastMap[pInput[2]];
			pInput += Channels;
			pOutput += Channels;
		}
	}
}
/*

	CPUImageExposureFilter: Adjusts the exposure of the image
	exposure : The adjusted exposure(-10.0 - 10.0, with 0.0 as the default)

	Shader :
	------------------------------------------------------------------------------------------ -
	varying highp vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;
	uniform highp float exposure;

	void main()
	{
	highp vec4 textureColor = texture2D(InputImageTexture, textureCoordinate);

	gl_FragColor = vec4(textureColor.rgb * pow(2.0, exposure), textureColor.w);
	}
	------------------------------------------------------------------------------------------ -
	*/
void CPUImageExposureFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float exposure)
{
	int Channels = Stride / Width;
	if (Channels == 1)
		return;
	unsigned char exposureMap[256] = {0};
	for (int pixel = 0; pixel < 256; pixel++)
	{
		exposureMap[pixel] = ClampToByte(pixel * pow(2.0, exposure));
	}
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			pOutput[0] = exposureMap[pInput[0]];
			pOutput[1] = exposureMap[pInput[1]];
			pOutput[2] = exposureMap[pInput[2]];
			pInput += Channels;
			pOutput += Channels;
		}
	}
}

/*

	CPUImageBrightnessFilter: Adjusts the brightness of the image
	brightness : The adjusted brightness(-1.0 - 1.0, with 0.0 as the default)

	Shader :
	------------------------------------------------------------------------------------------ -
	varying highp vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;
	uniform lowp float brightness;

	void main()
	{
	lowp vec4 textureColor = texture2D(InputImageTexture, textureCoordinate);

	gl_FragColor = vec4((textureColor.rgb + vec3(brightness)), textureColor.w);
	}
	------------------------------------------------------------------------------------------
	*/

void CPUImageBrightnessFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, int brightness)
{
	int Channels = Stride / Width;
	if (Channels == 1)
		return;

	unsigned char BrightnessMap[256] = {0};
	for (int pixel = 0; pixel < 256; pixel++)
	{
		BrightnessMap[pixel] = ClampToByte(pixel + brightness);
	}
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			pOutput[0] = BrightnessMap[pInput[0]];
			pOutput[1] = BrightnessMap[pInput[1]];
			pOutput[2] = BrightnessMap[pInput[2]];
			pInput += Channels;
			pOutput += Channels;
		}
	}
}
/*

	CPUImageFalseColorFilter: Uses the luminance of the image to mix between two user - specified colors
	firstColor : The first and second colors specify what colors replace the dark and light areas of the image, respectively.The defaults are(0.0, 0.0, 0.5) amd(1.0, 0.0, 0.0).
	secondColor :


	Shader :
	------------------------------------------------------------------------
	precision lowp float;

	varying highp vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;
	uniform float intensity;
	uniform vec3 firstColor;
	uniform vec3 secondColor;

	const mediump vec3 luminanceWeighting = vec3(0.2125, 0.7154, 0.0721);

	void main()
	{
	lowp vec4 textureColor = texture2D(InputImageTexture, textureCoordinate);
	float luminance = dot(textureColor.rgb, luminanceWeighting);

	gl_FragColor = vec4(mix(firstColor.rgb, secondColor.rgb, luminance), textureColor.a);
	}
	------------------------------------------------------------------------

	*/

void CPUImageFalseColorFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, unsigned char firstColorR, unsigned char firstColorG, unsigned char firstColorB, unsigned char secondColorR, unsigned char secondColorG, unsigned char secondColorB, int intensity)
{
	int Channels = Stride / Width;
	if (Channels == 1)
		return;
	unsigned short sIntensity =(unsigned short) max(min(intensity, 100), 0);
	int c1 = 256 * (100 - sIntensity) / 100;
	int c2 = 256 * (100 - (100 - sIntensity)) / 100;

	unsigned char ColorMapR[256] = {0};
	unsigned char ColorMapG[256] = {0};
	unsigned char ColorMapB[256] = {0};
	for (int pixel = 0; pixel < 256; pixel++)
	{
		float fPixel = pixel * (1.0f / 255.0f);
		ColorMapR[pixel] = mix_u8(firstColorR, secondColorR, fPixel);
		ColorMapG[pixel] = mix_u8(firstColorG, secondColorG, fPixel);
		ColorMapB[pixel] = mix_u8(firstColorB, secondColorB, fPixel);
	}
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			unsigned char luminanceWeighting =(unsigned char) ((13926 * pInput[0] + 46884 * pInput[1] + 4725 * pInput[2]) >> 16);

			pOutput[0] =(unsigned char) ((pInput[0] * c1 + ColorMapR[luminanceWeighting] * c2) >> 8);
			pOutput[1] =(unsigned char) ((pInput[1] * c1 + ColorMapG[luminanceWeighting] * c2) >> 8);
			pOutput[2] =(unsigned char)((pInput[2] * c1 + ColorMapB[luminanceWeighting] * c2) >> 8);
			pInput += Channels;
			pOutput += Channels;
		}
	}
}
/*

	CPUImageHazeFilter: Used to add or remove haze(similar to a UV filter)
	distance : Strength of the color applied.Default 0. Values between - .3 and .3 are best.
	slope : Amount of color change.Default 0. Values between - .3 and .3 are best.

	Shader :
	------------------------------------------------------------------------
	varying highp vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;

	uniform lowp float hazeDistance;
	uniform highp float slope;

	void main()
	{
	//todo reconsider precision modifiers
	highp vec4 color = vec4(1.0);//todo reimplement as a parameter

	highp float  d = textureCoordinate.y * slope + hazeDistance;

	highp vec4 c = texture2D(InputImageTexture, textureCoordinate); // consider using unpremultiply

	c = (c - d * color) / (1.0 - d);

	gl_FragColor = c; //consider using premultiply(c);
	}
	------------------------------------------------------------------------
	*/

//distance: Strength of the color applied.Default 0. Values between - .3 and .3 are best.
// slope : Amount of color change.Default 0. Values between - .3 and .3 are best.
void CPUImageHazeFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float distance, float slope, int intensity)
{
	int Channels = Stride / Width;
	if (Channels == 1)
		return;
	unsigned short sIntensity = (unsigned short)max(min(intensity, 100), 0);
	int c1 = 256 * (100 - sIntensity) / 100;
	int c2 = 256 * (100 - (100 - sIntensity)) / 100;
	short *distanceColorMap = (short *)malloc(Height * sizeof(short));
	short *patchDistanceMap = (short *)malloc(Height * sizeof(short));
	if (distanceColorMap == NULL || patchDistanceMap == NULL)
	{
		if (distanceColorMap)
		{
			free(distanceColorMap);
		}
		if (patchDistanceMap)
		{
			free(patchDistanceMap);
		}
		return;
	}
	float color = 1.0f;
	for (int i = 0; i < Height; i++)
	{
		float d = i * (1.0f / Height) * slope + distance;
		distanceColorMap[i] = ClampToByte(255.0 * d * color);
		patchDistanceMap[i] = (short)(256 * clamp(1.0f / (1.0f - d), 0.0f, 1.0f));
	}
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			pOutput[0] = (unsigned char)((int)(pInput[0] * c1 + ((ClampToByte(pInput[0] - distanceColorMap[Y]) * patchDistanceMap[Y]) >> 8) * c2) >> 8);
			pOutput[1] = (unsigned char)((int)(pInput[1] * c1 + ((ClampToByte(pInput[1] - distanceColorMap[Y]) * patchDistanceMap[Y]) >> 8) * c2) >> 8);
			pOutput[2] = (unsigned char)((int)(pInput[2] * c1 + ((ClampToByte(pInput[2] - distanceColorMap[Y]) * patchDistanceMap[Y]) >> 8) * c2) >> 8);
			pInput += Channels;
			pOutput += Channels;
		}
	}
	free(distanceColorMap);
	free(patchDistanceMap);
}

/*

	CPUImageOpacityFilter: Adjusts the alpha channel of the incoming image
	opacity : The value to multiply the incoming alpha channel for each pixel by(0.0 - 1.0, with 1.0 as the default)

	Shader :
	------------------------------------------------------------------------
	varying highp vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;
	uniform lowp float opacity;

	void main()
	{
	lowp vec4 textureColor = texture2D(InputImageTexture, textureCoordinate);

	gl_FragColor = vec4(textureColor.rgb, textureColor.a * opacity);
	}
	------------------------------------------------------------------------
	*/
void CPUImageOpacityFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float opacity)
{
	int Channels = Stride / Width;
	if (Channels == 1)
		return;
	if (Channels == 4)
	{
		unsigned char opacityMap[256] = {0};
		for (unsigned int pixel = 0; pixel < 256; pixel++)
		{
			opacityMap[pixel] = (unsigned char)(pixel * opacity);
		}
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pOutput = Output + (Y * Stride);
			unsigned char *pInput = Input + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				pOutput[3] = opacityMap[pInput[3]];
				pInput += Channels;
				pOutput += Channels;
			}
		}
	}
}

/*

	CPUImageLevelsFilter: Photoshop - like levels adjustment.The min, max, minOut and maxOut parameters are floats in the range[0, 1].If you have parameters from Photoshop in the range[0, 255] you must first convert them to be[0, 1].The gamma / mid parameter is a float >= 0. This matches the value from Photoshop.If you want to apply levels to RGB as well as individual channels you need to use this filter twice - first for the individual channels and then for all channels.
	*/

/*
	Shader :
	------------------------------------------------------------------------
	IOS版本

	//Gamma correction
	//Details: http://blog.mouaif.org/2009/01/22/photoshop-gamma-correction-shader/


	#define GammaCorrection(color, gamma)                                pow(color, 1.0 / gamma)


	** Levels control (Input (+gamma), Output)
	** Details: http://blog.mouaif.org/2009/01/28/levels-control-shader/


	#define LevelsControlInputRange(color, minInput, maxInput)                min(max(color - minInput, vec3(0.0)) / (maxInput - minInput), vec3(1.0))
	#define LevelsControlInput(color, minInput, gamma, maxInput)                GammaCorrection(LevelsControlInputRange(color, minInput, maxInput), gamma)
	#define LevelsControlOutputRange(color, minOutput, maxOutput)             mix(minOutput, maxOutput, color)
	#define LevelsControl(color, minInput, gamma, maxInput, minOutput, maxOutput)     LevelsControlOutputRange(LevelsControlInput(color, minInput, gamma, maxInput), minOutput, maxOutput)
	------------------------------------------------------------------------
	varying highp vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;
	uniform mediump vec3 levelMinimum;
	uniform mediump vec3 levelMiddle;
	uniform mediump vec3 levelMaximum;
	uniform mediump vec3 minOutput;
	uniform mediump vec3 maxOutput;

	void main()
	{
	mediump vec4 textureColor = texture2D(InputImageTexture, textureCoordinate);

	gl_FragColor = vec4(LevelsControl(textureColor.rgb, levelMinimum, levelMiddle, levelMaximum, minOutput, maxOutput), textureColor.a);
	}
	------------------------------------------------------------------------------------------ -
	安卓版本:
	------------------------------------------------------------------------------------------ -
	varying highp vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;
	uniform mediump vec3 levelMinimum;
	uniform mediump vec3 levelMiddle;
	uniform mediump vec3 levelMaximum;
	uniform mediump vec3 minOutput;
	uniform mediump vec3 maxOutput;

	void main()
	{
	mediump vec4 textureColor = texture2D(InputImageTexture, textureCoordinate);
	gl_FragColor = vec4(mix(minOutput, maxOutput, pow(min(max(textureColor.rgb - levelMinimum, vec3(0.0)) / (levelMaximum - levelMinimum), vec3(1.0)), 1.0 / levelMiddle)), textureColor.a);
	};
	------------------------------------------------------------------------------------------ -
	*/

void CPUImageLevelsFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, cpuLevelParams *redLevelParams, cpuLevelParams *greenLevelParams, cpuLevelParams *blueLevelParams)
{
	int Channels = Stride / Width;
	unsigned char LevelMapR[256] = {0};
	unsigned char LevelMapG[256] = {0};
	unsigned char LevelMapB[256] = {0};
	for (int pixel = 0; pixel < 256; pixel++)
	{
		if (redLevelParams->Enable)
		{
			LevelMapR[pixel] = (mix_u8( (unsigned char)redLevelParams->minOutput,  (unsigned char)redLevelParams->maxOutput, (powf(min(max(pixel - redLevelParams->levelMinimum, (0.0f)) / (redLevelParams->levelMaximum - redLevelParams->levelMinimum), (255)), 1.0f / (redLevelParams->levelMiddle * (1.0f / 255.0f))))));
		}
		else
		{
			LevelMapR[pixel] = (unsigned char) pixel;
		}
		if (greenLevelParams->Enable)
		{
			LevelMapG[pixel] = (mix_u8( (unsigned char)greenLevelParams->minOutput,  (unsigned char)greenLevelParams->maxOutput, (powf(min(max(pixel - greenLevelParams->levelMinimum, (0.0f)) / (greenLevelParams->levelMaximum - greenLevelParams->levelMinimum), (255)), 1.0f / (greenLevelParams->levelMiddle * (1.0f / 255.0f))))));
		}
		else
		{
			LevelMapG[pixel] = (unsigned char) pixel;
		}
		if (blueLevelParams->Enable)
		{
			LevelMapB[pixel] = (mix_u8( (unsigned char)blueLevelParams->minOutput,  (unsigned char)blueLevelParams->maxOutput, (powf(min(max(pixel - blueLevelParams->levelMinimum, (0.0f)) / (blueLevelParams->levelMaximum - blueLevelParams->levelMinimum), (255)), 1.0f / (blueLevelParams->levelMiddle * (1.0f / 255.0f))))));
		}
		else
		{
			LevelMapB[pixel] = (unsigned char) pixel;
		}
	}

	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			pOutput[0] = LevelMapR[pInput[0]];
			pOutput[1] = LevelMapG[pInput[1]];
			pOutput[2] = LevelMapB[pInput[2]];
			pInput += Channels;
			pOutput += Channels;
		}
	}
}

/*

	CPUImageHueFilter: Adjusts the hue of an image
	hue : The hue angle, in degrees. 90 degrees by default


	// Adapted from http://stackoverflow.com/questions/9234724/how-to-change-hue-of-a-texture-with-glsl - see for code and discussion


	Shader:
	------------------------------------------------------------------------
	precision highp float;
	varying highp vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;
	uniform mediump float hueAdjust;
	const highp  vec4  kRGBToYPrime = vec4(0.299, 0.587, 0.114, 0.0);
	const highp  vec4  kRGBToI = vec4(0.595716, -0.274453, -0.321263, 0.0);
	const highp  vec4  kRGBToQ = vec4(0.211456, -0.522591, 0.31135, 0.0);

	const highp  vec4  kYIQToR = vec4(1.0, 0.9563, 0.6210, 0.0);
	const highp  vec4  kYIQToG = vec4(1.0, -0.2721, -0.6474, 0.0);
	const highp  vec4  kYIQToB = vec4(1.0, -1.1070, 1.7046, 0.0);

	void main()
	{
	// Sample the Input pixel
	highp vec4 color = texture2D(InputImageTexture, textureCoordinate);

	// Convert to YIQ
	highp float   YPrime = dot(color, kRGBToYPrime);
	highp float   I = dot(color, kRGBToI);
	highp float   Q = dot(color, kRGBToQ);

	// Calculate the hue and chroma
	highp float   hue = atan(Q, I);
	highp float   chroma = sqrt(I * I + Q * Q);

	// Make the user's adjustments
	hue += (-hueAdjust); //why negative rotation?

	// Convert back to YIQ
	Q = chroma * sin(hue);
	I = chroma * cos(hue);

	// Convert back to RGB
	highp vec4    yIQ = vec4(YPrime, I, Q, 0.0);
	color.r = dot(yIQ, kYIQToR);
	color.g = dot(yIQ, kYIQToG);
	color.b = dot(yIQ, kYIQToB);

	// Save the result
	gl_FragColor = color;
	}
	------------------------------------------------------------------------ */

/* RGB to YIQ */
void rgb2yiq(unsigned char *R, unsigned char *G, unsigned char *B, short *Y, short *I, short *Q)
{
	*Y = (short) ((int)(0.299f * 65536) * *R + (int)(0.587f * 65536) * *G + (int)(0.114f * 65536) * *B) >> 16;
	*I = (short) ((int)(0.595f * 65536) * *R - (int)(0.274453f * 65536) * *G - (int)(0.321263f * 65536) * *B) >> 16;
	*Q = (short) ((int)(0.211456f * 65536) * *R - (int)(0.522591f * 65536) * *G + (int)(0.311135f * 65536) * *B) >> 16;
}

/* YIQ to RGB */
void yiq2rgb(short *Y, short *I, short *Q, unsigned char *R, unsigned char *G, unsigned char *B)
{
	*R = ClampToByte((int)(*Y + ((((int)(0.9563 * 65536)) * (*I)) + ((int)(0.6210 * 65536)) * (*Q))) >> 16);
	*G = ClampToByte((int)(*Y - ((((int)(0.2721 * 65536)) * (*I)) + ((int)(0.6474 * 65536)) * (*Q))) >> 16);
	*B = ClampToByte((int)(*Y + ((((int)(1.7046 * 65536)) * (*Q)) - ((int)(1.1070 * 65536)) * (*I))) >> 16);
}

void CPUImageHueFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float hueAdjust)
{
	int Channels = Stride / Width;
	hueAdjust = fmodf(hueAdjust, 360.0f) * 3.14159265358979323846f / 180.0f;
	float hueMap[256 * 256] = {0};
	float ChromaMap[256 * 256] = {0};

	for (int Q = 0; Q < 256; Q++)
	{
		float *pChromaMap = ChromaMap + Q * 256;
		for (int I = 0; I < 256; I++)
		{
			pChromaMap[0] = sqrtf((float)((I - 128) * (I - 128) + (Q - 128) * (Q - 128)));
			pChromaMap++;
		}
	}
	for (int Q = 0; Q < 256; Q++)
	{
		float *pHueMap = hueMap + Q * 256;
		for (int I = 0; I < 256; I++)
		{
			pHueMap[0] = atan2f((float)Q - 128, (float)I - 128);
			pHueMap++;
		}
	}
	float hue = 0;
	short YPrime = 0;
	short I = 0;
	short Q = 0;
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			// Convert to YIQ
			rgb2yiq(&pInput[0], &pInput[1], &pInput[2], &YPrime, &I, &Q);
			// Calculate the hue and chroma
			float *pHueMap = hueMap + ((Q + 128) << 8);
			hue = pHueMap[I + 128];
			float *pChromaMap = ChromaMap + ((Q + 128) << 8);
			float chroma = pChromaMap[I + 128];
			// Make the user's adjustments
			hue += (-hueAdjust); //why negative rotation?
			// Convert back to YIQ
			Q = (short)(chroma * sinf(hue));
			I = (short)(chroma * cosf(hue));
			yiq2rgb(&YPrime, &I, &Q, &pOutput[0], &pOutput[1], &pOutput[2]);
			pInput += Channels;
			pOutput += Channels;
		}
	}
}
/*

	CPUImageHighlightShadowTintFilter: Allows you to tint the shadows and highlights of an image independently using a color and intensity
	shadowTintColor : Shadow tint RGB color(GPUVector4).Default : {1.0f, 0.0f, 0.0f, 1.0f} (red).
	highlightTintColor : Highlight tint RGB color(GPUVector4).Default : {0.0f, 0.0f, 1.0f, 1.0f} (blue).
	shadowTintIntensity : Shadow tint intensity, from 0.0 to 1.0.Default : 0.0
	highlightTintIntensity : Highlight tint intensity, from 0.0 to 1.0, with 0.0 as the default.
	Shader :
	------------------------------------------------------------------------

	precision lowp float;

	varying highp vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;
	uniform lowp float shadowTintIntensity;
	uniform lowp float highlightTintIntensity;
	uniform highp vec4 shadowTintColor;
	uniform highp vec4 highlightTintColor;

	const mediump vec3 luminanceWeighting = vec3(0.2125, 0.7154, 0.0721);

	void main()
	{
	lowp vec4 textureColor = texture2D(InputImageTexture, textureCoordinate);
	highp float luminance = dot(textureColor.rgb, luminanceWeighting);

	highp vec4 shadowResult = mix(textureColor, max(textureColor, vec4(mix(shadowTintColor.rgb, textureColor.rgb, luminance), textureColor.a)), shadowTintIntensity);
	highp vec4 highlightResult = mix(textureColor, min(shadowResult, vec4(mix(shadowResult.rgb, highlightTintColor.rgb, luminance), textureColor.a)), highlightTintIntensity);

	gl_FragColor = vec4(mix(shadowResult.rgb, highlightResult.rgb, luminance), textureColor.a);
	}
	);
	------------------------------------------------------------------------

	*/

void CPUImageHighlightShadowTintFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float shadowTintR, float shadowTintG, float shadowTintB, float highlightTintR, float highlightTintG, float highlightTintB, float shadowTintIntensity, float highlightTintIntensity)
{
	int Channels = Stride / Width;
	if (Channels == 1)
	{
		return;
	}

	unsigned char HighlightShadowMapR[256 * 256] = {0};
	unsigned char HighlightShadowMapG[256 * 256] = {0};
	unsigned char HighlightShadowMapB[256 * 256] = {0};
	for (int lum = 0; lum < 256; lum++)
	{
		float luminance = (1.0f / 255.0f) * lum;
		unsigned char *pHighlightShadowMapR = HighlightShadowMapR + (lum << 8);
		unsigned char *pHighlightShadowMapG = HighlightShadowMapG + (lum << 8);
		unsigned char *pHighlightShadowMapB = HighlightShadowMapB + (lum << 8);
		for (int pixel = 0; pixel < 256; pixel++)
		{
			float fpixel = (1.0f / 255.0f) * pixel;
			float shadowResultR = mix(fpixel, max(fpixel, mix(shadowTintR, fpixel, luminance)), shadowTintIntensity);
			float shadowResultG = mix(fpixel, max(fpixel, mix(shadowTintG, fpixel, luminance)), shadowTintIntensity);
			float shadowResultB = mix(fpixel, max(fpixel, mix(shadowTintB, fpixel, luminance)), shadowTintIntensity);
			float highlightResultR = mix(fpixel, min(shadowResultR, mix(shadowResultR, highlightTintR, luminance)), highlightTintIntensity);
			float highlightResultG = mix(fpixel, min(shadowResultG, mix(shadowResultG, highlightTintG, luminance)), highlightTintIntensity);
			float highlightResultB = mix(fpixel, min(shadowResultB, mix(shadowResultB, highlightTintB, luminance)), highlightTintIntensity);
			pHighlightShadowMapR[pixel] = ClampToByte(255.0f * mix(shadowResultR, highlightResultR, luminance));
			pHighlightShadowMapG[pixel] = ClampToByte(255.0f * mix(shadowResultG, highlightResultG, luminance));
			pHighlightShadowMapB[pixel] = ClampToByte(255.0f * mix(shadowResultB, highlightResultB, luminance));
		}
	}
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			const unsigned char R = pInput[0];
			const unsigned char G = pInput[1];
			const unsigned char B = pInput[2];
			unsigned short lum = (unsigned short)((13926 * R + 46884 * G + 4725 * B) >> 16) << 8;
			unsigned char *pHighlightShadowMapR = HighlightShadowMapR + (lum);
			unsigned char *pHighlightShadowMapG = HighlightShadowMapG + (lum);
			unsigned char *pHighlightShadowMapB = HighlightShadowMapB + (lum);
			pOutput[0] = pHighlightShadowMapR[R];
			pOutput[1] = pHighlightShadowMapG[G];
			pOutput[2] = pHighlightShadowMapB[B];
			pInput += Channels;
			pOutput += Channels;
		}
	}
}
/*

	CPUImageHighlightShadowFilter: Adjusts the shadows and highlights of an image
	shadows : Increase to lighten shadows, from 0.0 to 1.0, with 0.0 as the default.
	highlights : Decrease to darken highlights, from 0.0 to 1.0, with 1.0 as the default.

	Shader :
	------------------------------------------------------------------------
	uniform sampler2D InputImageTexture;
	varying highp vec2 textureCoordinate;

	uniform lowp float shadows;
	uniform lowp float highlights;

	const mediump vec3 luminanceWeighting = vec3(0.3, 0.3, 0.3);

	void main()
	{
	lowp vec4 source = texture2D(InputImageTexture, textureCoordinate);
	mediump float luminance = dot(source.rgb, luminanceWeighting);

	mediump float shadow = clamp((pow(luminance, 1.0 / (shadows + 1.0)) + (-0.76)*pow(luminance, 2.0 / (shadows + 1.0))) - luminance, 0.0, 1.0);
	mediump float highlight = clamp((1.0 - (pow(1.0 - luminance, 1.0 / (2.0 - highlights)) + (-0.8)*pow(1.0 - luminance, 2.0 / (2.0 - highlights)))) - luminance, -1.0, 0.0);
	lowp vec3 result = vec3(0.0, 0.0, 0.0) + ((luminance + shadow + highlight) - 0.0) * ((source.rgb - vec3(0.0, 0.0, 0.0)) / (luminance - 0.0));

	gl_FragColor = vec4(result.rgb, source.a);
	}
	------------------------------------------------------------------------
	*/
void CPUImageHighlightShadowFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float shadows, float highlights)
{
	int Channels = Stride / Width;
	if (Channels == 1)
	{
		return;
	}
	short luminanceWeightingMap[256] = {0};
	short shadowMap[256] = {0};
	short highlightMap[256] = {0};

	int divLuminance[256 * 256] = {0};
	for (int pixel = 0; pixel < 256; pixel++)
	{
		luminanceWeightingMap[pixel] = (short)(pixel * 0.3f);
		float luminance = (1.0f / 255.0f) * pixel;
		shadowMap[pixel] = (short)(255.0f * clamp((powf(luminance, 1.0f / (shadows + 1.0f)) + (-0.76f) * powf(luminance, 2.0f / (shadows + 1.0f))) - luminance, 0.0f, 1.0f));
		highlightMap[pixel] = (short)(255.0f * clamp((1.0f - (powf(1.0f - luminance, 1.0f / (2.0f - highlights)) + (-0.8f) * powf(1.0f - luminance, 2.0f / (2.0f - highlights)))) - luminance, -1.0f, 0.0f));
	}
	for (int luminance = 0; luminance < 256; luminance++)
	{
		int *pDivLuminance = divLuminance + luminance * 256;
		for (int pixel = 0; pixel < 256; pixel++)
		{
			pDivLuminance[0] = (int)(255.0f * pixel * (1.0f / luminance));
			pDivLuminance++;
		}
	}
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			const short luminance = luminanceWeightingMap[pInput[0]] + luminanceWeightingMap[pInput[1]] + luminanceWeightingMap[pInput[2]];
			const short shadow = shadowMap[luminance];
			const short highlight = highlightMap[luminance];
			short lshpixel = (luminance + shadow + highlight);
			int *pDivLuminance = divLuminance + (luminance << 8);
			pOutput[0] =(unsigned char) ((lshpixel * pDivLuminance[pInput[0]]) >> 8);
			pOutput[1] =(unsigned char) ((lshpixel * pDivLuminance[pInput[1]]) >> 8);
			pOutput[2] =(unsigned char) ((lshpixel * pDivLuminance[pInput[2]]) >> 8);
			pInput += Channels;
			pOutput += Channels;
		}
	}
}

/*

	CPUImageMonochromeFilter: Converts the image to a single - color version, based on the luminance of each pixel
	intensity : The degree to which the specific color replaces the normal image color(0.0 - 1.0, with 1.0 as the default)
	color : The color to use as the basis for the effect, with(0.6, 0.45, 0.3, 1.0) as the default.

	Shader :
	------------------------------------------------------------------------------------------ -
	precision lowp float;

	varying highp vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;
	uniform float intensity;
	uniform vec3 filterColor;

	const mediump vec3 luminanceWeighting = vec3(0.2125, 0.7154, 0.0721);

	void main()
	{
	//desat, then apply overlay blend
	lowp vec4 textureColor = texture2D(InputImageTexture, textureCoordinate);
	float luminance = dot(textureColor.rgb, luminanceWeighting);

	lowp vec4 desat = vec4(vec3(luminance), 1.0);

	//overlay
	lowp vec4 OutputColor = vec4(
	(desat.r < 0.5 ? (2.0 * desat.r * filterColor.r) : (1.0 - 2.0 * (1.0 - desat.r) * (1.0 - filterColor.r))),
	(desat.g < 0.5 ? (2.0 * desat.g * filterColor.g) : (1.0 - 2.0 * (1.0 - desat.g) * (1.0 - filterColor.g))),
	(desat.b < 0.5 ? (2.0 * desat.b * filterColor.b) : (1.0 - 2.0 * (1.0 - desat.b) * (1.0 - filterColor.b))),
	1.0
	);

	//which is better, or are they equal?
	gl_FragColor = vec4(mix(textureColor.rgb, OutputColor.rgb, intensity), textureColor.a);
	}
	------------------------------------------------------------------------------------------ -
	*/
void CPUImageMonochromeFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, unsigned char filterColorR, unsigned char filterColorG, unsigned char filterColorB, int intensity)
{
	int Channels = Stride / Width;
	if (Channels == 1)
	{
		return;
	}
	unsigned short sIntensity = (unsigned short)max(min(intensity, 100), 0);
	int c1 = 256 * (100 - sIntensity) / 100;
	int c2 = 256 * (100 - (100 - sIntensity)) / 100;
	float fColorR = (float)filterColorR * (1.0f / 255.0f);
	float fColorG = (float)filterColorG * (1.0f / 255.0f);
	float fColorB = (float)filterColorB * (1.0f / 255.0f);
	unsigned char filterColorRMap[256] = {0};
	unsigned char filterColorGMap[256] = {0};
	unsigned char filterColorBMap[256] = {0};
	for (int luminance = 0; luminance < 256; luminance++)
	{
		float lum = (1.0f / 255.0f) * luminance;
		filterColorRMap[luminance] = (unsigned char)(255.0f * (lum < 0.5f ? (2.0f * lum * fColorR) : (1.0f - 2.0f * (1.0f - lum) * (1.0f - fColorR))));
		filterColorGMap[luminance] = (unsigned char)(255.0f * (lum < 0.5f ? (2.0f * lum * fColorG) : (1.0f - 2.0f * (1.0f - lum) * (1.0f - fColorG))));
		filterColorBMap[luminance] = (unsigned char)(255.0f * (lum < 0.5f ? (2.0f * lum * fColorB) : (1.0f - 2.0f * (1.0f - lum) * (1.0f - fColorB))));
	}
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			unsigned char lum =(unsigned char) ((13926 * pInput[0] + 46884 * pInput[1] + 4725 * pInput[2]) >> 16);
			pOutput[0] =(unsigned char)((int)(pInput[0] * c1 + filterColorRMap[lum] * c2) >> 8);
			pOutput[1] =(unsigned char) ((int)(pInput[1] * c1 + filterColorGMap[lum] * c2) >> 8);
			pOutput[2] =(unsigned char) ((int)(pInput[2] * c1 + filterColorBMap[lum] * c2) >> 8);
			pInput += Channels;
			pOutput += Channels;
		}
	}
}
/*

	CPUImageColorInvertFilter: Inverts the colors of an image


	Shader :
	------------------------------------------------------------------------------------------ -
	varying vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;

	void main()
	{
	vec4 textureColor = texture2D(InputImageTexture, textureCoordinate);

	gl_FragColor = vec4((1.0 - textureColor.rgb), textureColor.w);
	}
	------------------------------------------------------------------------------------------ -
	*/
void CPUImageColorInvertFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride)
{
	int Channels = Stride / Width;
	if (Channels == 1)
	{
		return;
	}
	unsigned char invertMap[256] = {0};
	for (int pixel = 0; pixel < 256; pixel++)
	{
		invertMap[pixel] =(unsigned char) (255 - pixel);
	}
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			pOutput[0] = invertMap[pInput[0]];
			pOutput[1] = invertMap[pInput[1]];
			pOutput[2] = invertMap[pInput[2]];
			pInput += Channels;
			pOutput += Channels;
		}
	}
}

/*

	CPUImageSolidColorGenerator: This Outputs a generated image with a solid color.You need to define the image size using - forceProcessingAtSize :
	color : The color, in a four component format, that is used to fill the image.

	Shader :
	------------------------------------------------------------------------

	precision lowp float;

	varying highp vec2 textureCoordinate;
	uniform sampler2D InputImageTexture;
	uniform vec4 color;
	uniform float useExistingAlpha;

	void main()
	{
	lowp vec4 textureColor = texture2D(InputImageTexture, textureCoordinate);
	gl_FragColor = vec4(color.rgb, max(textureColor.a, 1.0 - useExistingAlpha));
	}
	------------------------------------------------------------------------
	*/

void CPUImageSolidColorGenerator(unsigned char *Output, int Width, int Height, int Stride, unsigned char colorR, unsigned char colorG, unsigned char colorB, unsigned char colorAlpha)
{
	int Channels = Stride / Width;
	if (Channels == 4)
	{
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pOutput = Output + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				pOutput[0] = colorR;
				pOutput[1] = colorG;
				pOutput[2] = colorB;
				pOutput[3] = colorAlpha;
				pOutput += Channels;
			}
		}
	}
	else if (Channels == 3)
	{
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pOutput = Output + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				pOutput[0] = colorR;
				pOutput[1] = colorG;
				pOutput[2] = colorB;
				pOutput += Channels;
			}
		}
	}
}

/*

	CPUImageLuminanceThresholdFilter: Pixels with a luminance above the threshold will appear white, and those below will be black
	threshold : The luminance threshold, from 0.0 to 1.0, with a default of 0.5

	Shader :
	------------------------------------------------------------------------
	varying highp vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;
	uniform highp float threshold;

	const highp vec3 W = vec3(0.2125, 0.7154, 0.0721);

	void main()
	{
	highp vec4 textureColor = texture2D(InputImageTexture, textureCoordinate);
	highp float luminance = dot(textureColor.rgb, W);
	highp float thresholdResult = step(threshold, luminance);

	gl_FragColor = vec4(vec3(thresholdResult), textureColor.w);
	}
	------------------------------------------------------------------------
	*/

void CPUImageLuminanceThresholdFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, unsigned char threshold)
{
	int Channels = Stride / Width;
	if (Channels == 1)
	{

		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pOutput = Output + (Y * Stride);
			unsigned char *pInput = Input + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				pOutput[0] = step(threshold, pInput[0]);
				pInput++;
				pOutput++;
			}
		}
		return;
	}

	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			unsigned char luminance = (unsigned char)((13926 * pInput[0] + 46884 * pInput[1] + 4725 * pInput[2]) >> 16);
			pOutput[2] = pOutput[1] = pOutput[0] = step(threshold, luminance);
			pInput += Channels;
			pOutput += Channels;
		}
	}
}
/*

	CPUImageWhiteBalanceFilter: Adjusts the white balance of an image.
	temperature : The temperature to adjust the image by, in ºK.A value of 4000 is very cool and 7000 very warm.The default value is 5000. Note that the scale between 4000 and 5000 is nearly as visually significant as that between 5000 and 7000.
	tint : The tint to adjust the image by.A value of - 200 is very green and 200 is very pink.The default value is 0.


	Shader :
	------------------------------------------------------------------------
	uniform sampler2D InputImageTexture;
	varying highp vec2 textureCoordinate;

	uniform lowp float temperature;
	uniform lowp float tint;

	const lowp vec3 warmFilter = vec3(0.93, 0.54, 0.0);

	const mediump mat3 RGBtoYIQ = mat3(0.299, 0.587, 0.114, 0.596, -0.274, -0.322, 0.212, -0.523, 0.311);
	const mediump mat3 YIQtoRGB = mat3(1.0, 0.956, 0.621, 1.0, -0.272, -0.647, 1.0, -1.105, 1.702);

	void main()
	{
	lowp vec4 source = texture2D(InputImageTexture, textureCoordinate);

	mediump vec3 yiq = RGBtoYIQ * source.rgb; //adjusting tint
	yiq.b = clamp(yiq.b + tint*0.5226*0.1, -0.5226, 0.5226);
	lowp vec3 rgb = YIQtoRGB * yiq;

	lowp vec3 processed = vec3(
	(rgb.r < 0.5 ? (2.0 * rgb.r * warmFilter.r) : (1.0 - 2.0 * (1.0 - rgb.r) * (1.0 - warmFilter.r))), //adjusting temperature
	(rgb.g < 0.5 ? (2.0 * rgb.g * warmFilter.g) : (1.0 - 2.0 * (1.0 - rgb.g) * (1.0 - warmFilter.g))),
	(rgb.b < 0.5 ? (2.0 * rgb.b * warmFilter.b) : (1.0 - 2.0 * (1.0 - rgb.b) * (1.0 - warmFilter.b))));

	gl_FragColor = vec4(mix(rgb, processed, temperature), source.a);
	}
	------------------------------------------------------------------------
	*/

void CPUImageWhiteBalanceFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float temperature, float tint)
{
	int Channels = Stride / Width;
	if (Channels == 1)
	{
		return;
	}
	float Temperature = temperature;
	Temperature = Temperature < 5000 ? (float)(0.0004 * (Temperature - 5000.0)) : (float)(0.00006 * (Temperature - 5000.0));

	float Tint = tint;
	Tint = (float)(Tint / 100.0f);

	short YPrime = 0;
	short I = 0;
	short Q = 0;

	float warmFilterR = 0.93f;
	float warmFilterG = 0.54f;
	float warmFilterB = 0;
	int plusTint = (int)(Tint * 255.0f * 0.5226f * 0.1f);

	short QTint[256] = {0};
	unsigned char processedMap[256 * 3] = {0};
	unsigned char *processedRMap = &processedMap[0];
	unsigned char *processedGMap = &processedMap[256];
	unsigned char *processedBMap = &processedMap[512];
	for (int pixel = 0; pixel < 256; pixel++)
	{
		float fpixel = pixel * (1.0f / 255.0f);
		QTint[pixel] = (short)clamp((float)(pixel - 127 + plusTint), -127.0f, 127.0f);
		float processedR = (fpixel < 0.5f ? (2.0f * fpixel * warmFilterR) : (1.0f - 2.0f * (1.0f - fpixel) * (1.0f - warmFilterR)));
		float processedG = (fpixel < 0.5f ? (2.0f * fpixel * warmFilterG) : (1.0f - 2.0f * (1.0f - fpixel) * (1.0f - warmFilterG)));
		float processedB = (fpixel < 0.5f ? (2.0f * fpixel * warmFilterB) : (1.0f - 2.0f * (1.0f - fpixel) * (1.0f - warmFilterB)));
		processedRMap[pixel] = ClampToByte(255.0f * mix(fpixel, processedR, Temperature));
		processedGMap[pixel] = ClampToByte(255.0f * mix(fpixel, processedG, Temperature));
		processedBMap[pixel] = ClampToByte(255.0f * mix(fpixel, processedB, Temperature));
	}
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			const unsigned char R = pInput[0];
			const unsigned char G = pInput[1];
			const unsigned char B = pInput[2];

			YPrime =(short) (((int)(0.299f * 65536) * R + (int)(0.587f * 65536) * G + (int)(0.114f * 65536) * B) >> 16);
			I = (short)(((int)(0.595f * 65536) * R - (int)(0.274453f * 65536) * G - (int)(0.321263f * 65536) * B) >> 16);
			Q = (short)(((int)(0.211456f * 65536) * R - (int)(0.522591f * 65536) * G + (int)(0.311135f * 65536) * B) >> 16);
			//adjusting tint
			Q = QTint[Q + 127];
			//adjusting temperature
			pOutput[0] = processedRMap[ClampToByte(YPrime + (((int)(0.9563 * 65536) * I + (int)(0.6210 * 65536) * Q) >> 16))];
			pOutput[1] = processedGMap[ClampToByte(YPrime - (((int)(0.2721 * 65536) * I + (int)(0.6474 * 65536) * Q) >> 16))];
			pOutput[2] = processedBMap[ClampToByte(YPrime + (((int)(1.7046 * 65536) * Q - (int)(1.1070 * 65536) * I) >> 16))];

			pInput += Channels;
			pOutput += Channels;
		}
	}
}

/*

	CPUImageVibranceFilter: Adjusts the vibrance of an image
	vibrance : The vibrance adjustment to apply, using 0.0 as the default, and a suggested min / max of around - 1.2 and 1.2, respectively.

	Shader :
	------------------------------------------------------------------------
	varying highp vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;
	uniform lowp float vibrance;

	void main() {
	lowp vec4 color = texture2D(InputImageTexture, textureCoordinate);
	lowp float average = (color.r + color.g + color.b) / 3.0;
	lowp float mx = max(color.r, max(color.g, color.b));
	lowp float amt = (mx - average) * (-vibrance * 3.0);
	color.rgb = mix(color.rgb, vec3(mx), amt);
	gl_FragColor = color;
	}
	------------------------------------------------------------------------
	*/

void CPUImageVibranceFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float vibrance)
{
	int Channels = Stride / Width;
	if (Channels == 1)
	{
		return;
	}
	int iVibrance = (int)(-(vibrance * 256));
	unsigned char mx = 0;
	int amt = 0;
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			const unsigned char r = pInput[0];
			const unsigned char g = pInput[1];
			const unsigned char b = pInput[2];
			mx = max3(r, g, b);
			amt = (3 * mx - (r + g + b)) * iVibrance;
			pOutput[0] = ClampToByte((r * (255 * 256 - amt) + mx * amt) >> 16);
			pOutput[1] = ClampToByte((g * (255 * 256 - amt) + mx * amt) >> 16);
			pOutput[2] = ClampToByte((b * (255 * 256 - amt) + mx * amt) >> 16);

			pInput += Channels;
			pOutput += Channels;
		}
	}
}

/*

	CPUImageSkinToneFilter: A skin - tone adjustment filter that affects a unique range of light skin - tone colors and adjusts the pink / green or pink / orange range accordingly.Default values are targetted at fair caucasian skin, but can be adjusted as required.
	skinToneAdjust : Amount to adjust skin tone.Default : 0.0, suggested min / max : -0.3 and 0.3 respectively.
	skinHue : Skin hue to be detected.Default : 0.05 (fair caucasian to reddish skin).
	skinHueThreshold : Amount of variance in skin hue.Default : 40.0.
	maxHueShift : Maximum amount of hue shifting allowed.Default : 0.25.
	maxSaturationShift = Maximum amount of saturation to be shifted(when using orange).Default : 0.4.
	upperSkinToneColor = GPUImageSkinToneUpperColorGreen or GPUImageSkinToneUpperColorOrange

	//upperSkinToneColor  is 0 or 1
	Shader :
	------------------------------------------------------------------------

	varying highp vec2 textureCoordinate;

	uniform sampler2D InputImageTexture;

	// [-1;1] <=> [pink;orange]
	uniform highp float skinToneAdjust; // will make reds more pink

	// Other parameters
	uniform mediump float skinHue;
	uniform mediump float skinHueThreshold;
	uniform mediump float maxHueShift;
	uniform mediump float maxSaturationShift;
	uniform int upperSkinToneColor;

	// RGB <-> HSV conversion, thanks to http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
	highp vec3 rgb2hsv(highp vec3 c)
	{
	highp vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
	highp vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
	highp vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

	highp float d = q.x - min(q.w, q.y);
	highp float e = 1.0e-10;
	return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
	}

	// HSV <-> RGB conversion, thanks to http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
	highp vec3 hsv2rgb(highp vec3 c)
	{
	highp vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	highp vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
	return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
	}


	// Main
	void main()
	{

	// Sample the Input pixel
	highp vec4 colorRGB = texture2D(InputImageTexture, textureCoordinate);

	// Convert color to HSV, extract hue
	highp vec3 colorHSV = rgb2hsv(colorRGB.rgb);
	highp float hue = colorHSV.x;

	// check how far from skin hue
	highp float dist = hue - skinHue;
	if (dist > 0.5)
	dist -= 1.0;
	if (dist < -0.5)
	dist += 1.0;
	dist = abs(dist) / 0.5; // normalized to [0,1]

	// Apply Gaussian like filter
	highp float weight = exp(-dist*dist*skinHueThreshold);
	weight = clamp(weight, 0.0, 1.0);

	// Using pink/green, so only adjust hue
	if (upperSkinToneColor == 0) {
	colorHSV.x += skinToneAdjust * weight * maxHueShift;
	// Using pink/orange, so adjust hue < 0 and saturation > 0
	}
	else if (upperSkinToneColor == 1) {
	// We want more orange, so increase saturation
	if (skinToneAdjust > 0.0)
	colorHSV.y += skinToneAdjust * weight * maxSaturationShift;
	// we want more pinks, so decrease hue
	else
	colorHSV.x += skinToneAdjust * weight * maxHueShift;
	}

	// final color
	highp vec3 finalColorRGB = hsv2rgb(colorHSV.rgb);

	// display
	gl_FragColor = vec4(finalColorRGB, 1.0);
	}
	);
	------------------------------------------------------------------------
	*/

void rgb2hsv(const unsigned char *R, const unsigned char *G, const unsigned char *B, unsigned char *H, unsigned char *S, unsigned char *V)
{
	int r = *R;
	int g = *G;
	int b = *B;

	int h, s;
	int nMax = max3(r, g, b);
	int nMin = min3(r, g, b);
	int diff = nMax - nMin;

	if (diff == 0)
	{
		h = 0;
		s = 0;
	}
	else
	{
		if (nMin == b)
		{
			h = 60 * (g - r) / diff + 60;
		}
		else if (nMin == r)
		{
			h = 60 * (b - g) / diff + 180;
		}
		else
		{
			h = 60 * (r - b) / diff + 300;
		}
		// normalize 0`359
		//if (h < 0)    h += 360; if (h >= 360) h -= 360;
		if (!((unsigned)(int)(h) < (360)))
		{
			if (h < 0)
				h += 360;
			else
				h -= 360;
		}
		if (nMax == 0)
		{
			s = 0;
		}
		else
		{
			s = 255 * diff / nMax;
		}
	}

	*H = (unsigned char)(h >> 1); // 0`179
	*S = (unsigned char)s;		  // 0`255
	*V = (unsigned char)nMax;					  // 0`255
}

void hsv2rgb(const unsigned char *H, const unsigned char *S, const unsigned char *V, unsigned char *R, unsigned char *G, unsigned char *B)
{
	if (*S > 0)
	{
		int r, g, b;
		r = *V;
		g = *V;
		b = *V;
		float h = *H * (6.0f / 180.0f); // 0`180 -> 0.0`1.0
		int i = (int)h;
		int f = (int)(256 * (h - (float)i));
		int VS = (*V * *S) >> 8;
		int VSF = VS * f;
		switch (i)
		{
		case 0:
			b -= VS;
			g = b + (VSF >> 8);
			break;
		case 1:
			r = *V - (VSF >> 8);
			b -= VS;
			break;
		case 2:
			r -= VS;
			b = r + (VSF >> 8);
			break;
		case 3:
			r -= VS;
			g -= (VSF >> 8);
			break;
		case 4:
			g -= VS;
			r = g + (VSF >> 8);
			break;
		case 5:
			g -= VS;
			b -= (VSF >> 8);
			break;
        default:
            break;
		}
		*R = (unsigned char)(r);
		*G = (unsigned char)(g);
		*B = (unsigned char)(b);
	}
	else
	{
		*R = *V;
		*G = *V;
		*B = *V;
	}
}

void CPUImageSkinToneFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float skinToneAdjust, float skinHue, float skinHueThreshold, float maxHueShift, float maxSaturationShift, int upperSkinToneColor)
{
	int Channels = Stride / Width;
	if (Channels == 1)
	{
		return;
	}
	int maxSatShiftAdjust = (int)(maxSaturationShift * 255.0f * skinToneAdjust);
	float maxHueShiftAdjust = maxHueShift * skinToneAdjust;
	unsigned char hueMap[256] = {0};
	int satMap[256] = {0};
	for (unsigned int H = 0; H < 256; H++)
	{
		satMap[H] = 0;
		float hue = H * (1.0f / 255.0f);
		// check how far from skin hue
		float dist = hue - skinHue;
		if (dist > 0.5f)
			dist -= 1.0f;
		if (dist < -0.5f)
			dist += 1.0f;
		dist = (float)(fabsf(dist) * (1.0f / 0.5f)); // normalized to [0,1]
		// Apply Gaussian like filter
		float weightMap = clamp(expf(-dist * dist * skinHueThreshold), 0.0f, 1.0f);

		// Using pink/green, so only adjust hue
		if (upperSkinToneColor == 0)
		{
			hue += maxHueShiftAdjust * weightMap;
			// Using pink/orange, so adjust hue < 0 and saturation > 0
		}
		else if (upperSkinToneColor == 1)
		{
			// We want more orange, so increase saturation
			if (skinToneAdjust > 0.0f)
				satMap[H] = (int)(maxSatShiftAdjust * weightMap);
			// we want more pinks, so decrease hue
			else
				hue += maxHueShiftAdjust * weightMap;
		}
		hueMap[H] = ClampToByte(hue * 255.0f);
	}
	unsigned char H, S, V, _S;
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *pOutput = Output + (Y * Stride);
		unsigned char *pInput = Input + (Y * Stride);
		for (int X = 0; X < Width; X++)
		{
			unsigned char R = pInput[0];
			unsigned char G = pInput[1];
			unsigned char B = pInput[2];
			// Convert color to HSV, extract hue
			rgb2hsv(&R, &G, &B, &H, &S, &V);
			// final color
			_S = (unsigned char)(S + satMap[H]);
			hsv2rgb(&hueMap[H], &_S, &V, &pOutput[0], &pOutput[1], &pOutput[2]);

			pInput += Channels;
			pOutput += Channels;
		}
	}
}

/*
	CPUImageGaussianBlurFilter: A hardware - optimized, variable - radius Gaussian blur
	texelSpacingMultiplier : A multiplier for the spacing between texels, ranging from 0.0 on up, with a default of 1.0.Adjusting this may slightly increase the blur strength, but will introduce artifacts in the result.Highly recommend using other parameters first, before touching this one.
	blurRadiusInPixels : A radius in pixels to use for the blur, with a default of 2.0.This adjusts the sigma variable in the Gaussian distribution function.
	blurRadiusAsFractionOfImageWidth :
	blurRadiusAsFractionOfImageHeight : Setting these properties will allow the blur radius to scale with the size of the image
	blurPasses : The number of times to sequentially blur the incoming image.The more passes, the slower the filter.
	*/

void CalGaussianCoeff(float sigma, float *a0, float *a1, float *a2, float *a3, float *b1, float *b2, float *cprev, float *cnext)
{
	float alpha, lamma, k;

	if (sigma < 0.5f)
		sigma = 0.5f;
	alpha = (float)expf((0.726) * (0.726)) / sigma;
	lamma = (float)expf(-alpha);
	*b2 = (float)expf(-2 * alpha);
	k = (1 - lamma) * (1 - lamma) / (1 + 2 * alpha * lamma - (*b2));
	*a0 = k;
	*a1 = k * (alpha - 1) * lamma;
	*a2 = k * (alpha + 1) * lamma;
	*a3 = -k * (*b2);
	*b1 = -2 * lamma;
	*cprev = (*a0 + *a1) / (1 + *b1 + *b2);
	*cnext = (*a2 + *a3) / (1 + *b1 + *b2);
}

void gaussianHorizontal(unsigned char *bufferPerLine, const unsigned char *lpRowInitial, unsigned char *lpColumn, int width, int height, int Channels, int Nwidth, float a0a1, float a2a3, float b1b2, float cprev, float cnext)
{
	int HeightStep = Channels * height;
	int WidthSubOne = width - 1;
	if (Channels == 3)
	{
		float prevOut[3];
		prevOut[0] = (lpRowInitial[0] * cprev);
		prevOut[1] = (lpRowInitial[1] * cprev);
		prevOut[2] = (lpRowInitial[2] * cprev);
		for (int x = 0; x < width; ++x)
		{
			prevOut[0] = ((lpRowInitial[0] * (a0a1)) - (prevOut[0] * (b1b2)));
			prevOut[1] = ((lpRowInitial[1] * (a0a1)) - (prevOut[1] * (b1b2)));
			prevOut[2] = ((lpRowInitial[2] * (a0a1)) - (prevOut[2] * (b1b2)));
			bufferPerLine[0] = (unsigned char)prevOut[0];
			bufferPerLine[1] = (unsigned char)prevOut[1];
			bufferPerLine[2] = (unsigned char)prevOut[2];
			bufferPerLine += Channels;
			lpRowInitial += Channels;
		}
		lpRowInitial -= Channels;
		lpColumn += HeightStep * WidthSubOne;
		bufferPerLine -= Channels;
		prevOut[0] = (lpRowInitial[0] * cnext);
		prevOut[1] = (lpRowInitial[1] * cnext);
		prevOut[2] = (lpRowInitial[2] * cnext);

		for (int x = WidthSubOne; x >= 0; --x)
		{
			prevOut[0] = ((lpRowInitial[0] * (a2a3)) - (prevOut[0] * (b1b2)));
			prevOut[1] = ((lpRowInitial[1] * (a2a3)) - (prevOut[1] * (b1b2)));
			prevOut[2] = ((lpRowInitial[2] * (a2a3)) - (prevOut[2] * (b1b2)));
			bufferPerLine[0] += (unsigned char)prevOut[0];
			bufferPerLine[1] += (unsigned char)prevOut[1];
			bufferPerLine[2] += (unsigned char)prevOut[2];
			lpColumn[0] = bufferPerLine[0];
			lpColumn[1] = bufferPerLine[1];
			lpColumn[2] = bufferPerLine[2];
			lpRowInitial -= Channels;
			lpColumn -= HeightStep;
			bufferPerLine -= Channels;
		}
	}
	else if (Channels == 4)
	{
		float prevOut[4];

		prevOut[0] = (lpRowInitial[0] * cprev);
		prevOut[1] = (lpRowInitial[1] * cprev);
		prevOut[2] = (lpRowInitial[2] * cprev);
		prevOut[3] = (lpRowInitial[3] * cprev);
		for (int x = 0; x < width; ++x)
		{
			prevOut[0] = ((lpRowInitial[0] * (a0a1)) - (prevOut[0] * (b1b2)));
			prevOut[1] = ((lpRowInitial[1] * (a0a1)) - (prevOut[1] * (b1b2)));
			prevOut[2] = ((lpRowInitial[2] * (a0a1)) - (prevOut[2] * (b1b2)));
			prevOut[3] = ((lpRowInitial[3] * (a0a1)) - (prevOut[3] * (b1b2)));

			bufferPerLine[0] = (unsigned char)prevOut[0];
			bufferPerLine[1] = (unsigned char)prevOut[1];
			bufferPerLine[2] = (unsigned char)prevOut[2];
			bufferPerLine[3] = (unsigned char)prevOut[3];
			bufferPerLine += Channels;
			lpRowInitial += Channels;
		}
		lpRowInitial -= Channels;
		lpColumn += HeightStep * WidthSubOne;
		bufferPerLine -= Channels;

		prevOut[0] = (lpRowInitial[0] * cnext);
		prevOut[1] = (lpRowInitial[1] * cnext);
		prevOut[2] = (lpRowInitial[2] * cnext);
		prevOut[3] = (lpRowInitial[3] * cnext);

		for (int x = WidthSubOne; x >= 0; --x)
		{
			prevOut[0] = ((lpRowInitial[0] * a2a3) - (prevOut[0] * b1b2));
			prevOut[1] = ((lpRowInitial[1] * a2a3) - (prevOut[1] * b1b2));
			prevOut[2] = ((lpRowInitial[2] * a2a3) - (prevOut[2] * b1b2));
			prevOut[3] = ((lpRowInitial[3] * a2a3) - (prevOut[3] * b1b2));
			bufferPerLine[0] += (unsigned char)prevOut[0];
			bufferPerLine[1] += (unsigned char)prevOut[1];
			bufferPerLine[2] += (unsigned char)prevOut[2];
			bufferPerLine[3] += (unsigned char)prevOut[3];
			lpColumn[0] = bufferPerLine[0];
			lpColumn[1] = bufferPerLine[1];
			lpColumn[2] = bufferPerLine[2];
			lpColumn[3] = bufferPerLine[3];
			lpRowInitial -= Channels;
			lpColumn -= HeightStep;
			bufferPerLine -= Channels;
		}
	}
	else if (Channels == 1)
	{
		float prevOut = (lpRowInitial[0] * cprev);

		for (int x = 0; x < width; ++x)
		{
			prevOut = ((lpRowInitial[0] * (a0a1)) - (prevOut * (b1b2)));
			bufferPerLine[0] = (unsigned char)prevOut;
			bufferPerLine += Channels;
			lpRowInitial += Channels;
		}
		lpRowInitial -= Channels;
		lpColumn += HeightStep * WidthSubOne;
		bufferPerLine -= Channels;

		prevOut = (lpRowInitial[0] * cnext);

		for (int x = WidthSubOne; x >= 0; --x)
		{
			prevOut = ((lpRowInitial[0] * a2a3) - (prevOut * b1b2));
			bufferPerLine[0] += (unsigned char)prevOut;
			lpColumn[0] = bufferPerLine[0];
			lpRowInitial -= Channels;
			lpColumn -= HeightStep;
			bufferPerLine -= Channels;
		}
	}
}

void gaussianVertical(unsigned char *bufferPerLine, const unsigned char *lpRowInitial, unsigned char *lpColInitial, int height, int width, int Channels, float a0a1, float a2a3, float b1b2, float cprev, float cnext)
{

	int WidthStep = Channels * width;
	int HeightSubOne = height - 1;
	if (Channels == 3)
	{
		float prevOut[3];
		prevOut[0] = (lpRowInitial[0] * cprev);
		prevOut[1] = (lpRowInitial[1] * cprev);
		prevOut[2] = (lpRowInitial[2] * cprev);

		for (int y = 0; y < height; y++)
		{
			prevOut[0] = ((lpRowInitial[0] * a0a1) - (prevOut[0] * b1b2));
			prevOut[1] = ((lpRowInitial[1] * a0a1) - (prevOut[1] * b1b2));
			prevOut[2] = ((lpRowInitial[2] * a0a1) - (prevOut[2] * b1b2));
			bufferPerLine[0] = (unsigned char)prevOut[0];
			bufferPerLine[1] = (unsigned char)prevOut[1];
			bufferPerLine[2] = (unsigned char)prevOut[2];
			bufferPerLine += Channels;
			lpRowInitial += Channels;
		}
		lpRowInitial -= Channels;
		bufferPerLine -= Channels;
		lpColInitial += WidthStep * HeightSubOne;
		prevOut[0] = (lpRowInitial[0] * cnext);
		prevOut[1] = (lpRowInitial[1] * cnext);
		prevOut[2] = (lpRowInitial[2] * cnext);
		for (int y = HeightSubOne; y >= 0; y--)
		{
			prevOut[0] = ((lpRowInitial[0] * a2a3) - (prevOut[0] * b1b2));
			prevOut[1] = ((lpRowInitial[1] * a2a3) - (prevOut[1] * b1b2));
			prevOut[2] = ((lpRowInitial[2] * a2a3) - (prevOut[2] * b1b2));
			bufferPerLine[0] += (unsigned char)prevOut[0];
			bufferPerLine[1] += (unsigned char)prevOut[1];
			bufferPerLine[2] += (unsigned char)prevOut[2];
			lpColInitial[0] = bufferPerLine[0];
			lpColInitial[1] = bufferPerLine[1];
			lpColInitial[2] = bufferPerLine[2];
			lpRowInitial -= Channels;
			lpColInitial -= WidthStep;
			bufferPerLine -= Channels;
		}
	}
	else if (Channels == 4)
	{
		float prevOut[4];

		prevOut[0] = (lpRowInitial[0] * cprev);
		prevOut[1] = (lpRowInitial[1] * cprev);
		prevOut[2] = (lpRowInitial[2] * cprev);
		prevOut[3] = (lpRowInitial[3] * cprev);

		for (int y = 0; y < height; y++)
		{
			prevOut[0] = ((lpRowInitial[0] * a0a1) - (prevOut[0] * b1b2));
			prevOut[1] = ((lpRowInitial[1] * a0a1) - (prevOut[1] * b1b2));
			prevOut[2] = ((lpRowInitial[2] * a0a1) - (prevOut[2] * b1b2));
			prevOut[3] = ((lpRowInitial[3] * a0a1) - (prevOut[3] * b1b2));
			bufferPerLine[0] = (unsigned char)prevOut[0];
			bufferPerLine[1] = (unsigned char)prevOut[1];
			bufferPerLine[2] = (unsigned char)prevOut[2];
			bufferPerLine[3] = (unsigned char)prevOut[3];
			bufferPerLine += Channels;
			lpRowInitial += Channels;
		}
		lpRowInitial -= Channels;
		bufferPerLine -= Channels;
		lpColInitial += WidthStep * HeightSubOne;
		prevOut[0] = (lpRowInitial[0] * cnext);
		prevOut[1] = (lpRowInitial[1] * cnext);
		prevOut[2] = (lpRowInitial[2] * cnext);
		prevOut[3] = (lpRowInitial[3] * cnext);
		for (int y = HeightSubOne; y >= 0; y--)
		{
			prevOut[0] = ((lpRowInitial[0] * a2a3) - (prevOut[0] * b1b2));
			prevOut[1] = ((lpRowInitial[1] * a2a3) - (prevOut[1] * b1b2));
			prevOut[2] = ((lpRowInitial[2] * a2a3) - (prevOut[2] * b1b2));
			prevOut[3] = ((lpRowInitial[3] * a2a3) - (prevOut[3] * b1b2));
			bufferPerLine[0] += (unsigned char)prevOut[0];
			bufferPerLine[1] += (unsigned char)prevOut[1];
			bufferPerLine[2] += (unsigned char)prevOut[2];
			bufferPerLine[3] += (unsigned char)prevOut[3];
			lpColInitial[0] = bufferPerLine[0];
			lpColInitial[1] = bufferPerLine[1];
			lpColInitial[2] = bufferPerLine[2];
			lpColInitial[3] = bufferPerLine[3];
			lpRowInitial -= Channels;
			lpColInitial -= WidthStep;
			bufferPerLine -= Channels;
		}
	}
	else if (Channels == 1)
	{
		float prevOut = 0;
		prevOut = (lpRowInitial[0] * cprev);
		for (int y = 0; y < height; y++)
		{
			prevOut = ((lpRowInitial[0] * a0a1) - (prevOut * b1b2));
			bufferPerLine[0] = (unsigned char)prevOut;
			bufferPerLine += Channels;
			lpRowInitial += Channels;
		}
		lpRowInitial -= Channels;
		bufferPerLine -= Channels;
		lpColInitial += WidthStep * HeightSubOne;
		prevOut = (lpRowInitial[0] * cnext);
		for (int y = HeightSubOne; y >= 0; y--)
		{
			prevOut = ((lpRowInitial[0] * a2a3) - (prevOut * b1b2));
			bufferPerLine[0] += (unsigned char)prevOut;
			lpColInitial[0] = bufferPerLine[0];
			lpRowInitial -= Channels;
			lpColInitial -= WidthStep;
			bufferPerLine -= Channels;
		}
	}
}

void CPUImageGaussianBlurFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float GaussianSigma)
{

	int Channels = Stride / Width;
	float a0, a1, a2, a3, b1, b2, cprev, cnext;

	CalGaussianCoeff(GaussianSigma, &a0, &a1, &a2, &a3, &b1, &b2, &cprev, &cnext);

	float a0a1 = (a0 + a1);
	float a2a3 = (a2 + a3);
	float b1b2 = (b1 + b2);

	int bufferSizePerThread = (Width > Height ? Width : Height) * Channels;
	unsigned char *bufferPerLine = (unsigned char *)malloc((size_t)bufferSizePerThread);
	unsigned char *tempData = (unsigned char *)malloc((size_t)Height * Stride);
	if (bufferPerLine == NULL || tempData == NULL)
	{
		if (tempData)
		{
			free(tempData);
		}
		if (bufferPerLine)
		{
			free(bufferPerLine);
		}
		return;
	}
	for (int y = 0; y < Height; ++y)
	{
		unsigned char *lpRowInitial = Input + Stride * y;
		unsigned char *lpColInitial = tempData + y * Channels;
		gaussianHorizontal(bufferPerLine, lpRowInitial, lpColInitial, Width, Height, Channels, Width, a0a1, a2a3, b1b2, cprev, cnext);
	}
	int HeightStep = Height * Channels;
	for (int x = 0; x < Width; ++x)
	{
		unsigned char *lpColInitial = Output + x * Channels;
		unsigned char *lpRowInitial = tempData + HeightStep * x;
		gaussianVertical(bufferPerLine, lpRowInitial, lpColInitial, Height, Width, Channels, a0a1, a2a3, b1b2, cprev, cnext);
	}

	free(bufferPerLine);
	free(tempData);
}

	/*
	CPUImageUnsharpMaskFilter: Applies an unsharp mask
	blurRadiusInPixels : The blur radius of the underlying Gaussian blur.The default is 4.0.
	intensity : The strength of the sharpening, from 0.0 on up, with a default of 1.0
	*/
#define float2fixed(x) (((int)((x)*4096.0f + 0.5f)) << 8)

void rgb2ycbcr(unsigned char R, unsigned char G, unsigned char B, unsigned char *y, unsigned char *cb, unsigned char *cr)
{
	*y = (unsigned char)((19595 * R + 38470 * G + 7471 * B) >> 16);
	*cb = (unsigned char)(((36962 * (B - *y)) >> 16) + 128);
	*cr = (unsigned char)(((46727 * (R - *y)) >> 16) + 128);
}

void ycbcr2rgb(unsigned char y, unsigned char Cb, unsigned char Cr, unsigned char *R, unsigned char *G, unsigned char *B)
{
	int y_fixed = (y << 20) + (1 << 19); // rounding
	int r, g, b;
	int cr = Cr - 128;
	int cb = Cb - 128;
	r = y_fixed + cr * float2fixed(1.40200f);
	g = y_fixed + (cr * -float2fixed(0.71414f)) + ((cb * -float2fixed(0.34414f)) * 0xffff0000);
	b = y_fixed + cb * float2fixed(1.77200f);
	r >>= 20;
	g >>= 20;
	b >>= 20;
	if ((unsigned)r > 255)
	{
		if (r < 0)
			r = 0;
		else
			r = 255;
	}
	if ((unsigned)g > 255)
	{
		if (g < 0)
			g = 0;
		else
			g = 255;
	}
	if ((unsigned)b > 255)
	{
		if (b < 0)
			b = 0;
		else
			b = 255;
	}
	*R =(unsigned char) r;
	*G =(unsigned char)g;
	*B =(unsigned char)b;
}

void CPUImageUnsharpMaskFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float GaussianSigma, int intensity)
{
	int Channels = Stride / Width;
	intensity = max(min(intensity, 100), 0);
	int c1 = 256 * (100 - intensity) / 100;
	int c2 = 256 * (100 - (100 - intensity)) / 100;

	unsigned char unsharpMaskMap[256 * 256] = {0};
	for (unsigned int PS = 0; PS < 256; PS++)
	{
		unsigned char *pUnsharpMaskMap = unsharpMaskMap + (PS << 8);
		for (unsigned int PD = 0; PD < 256; PD++)
		{
			unsigned char retPD = ClampToByte((PS - PD) + 128);
			retPD = (unsigned char)((PS <= 128) ? (retPD * PS / 128) : (255 - (255 - retPD) * (255 - PS) / 128));
			//增强边缘法
			//   unsigned char retPD = ClampToByte((PS - PD) + PS);
			pUnsharpMaskMap[0] = ClampToByte((PS * c1 + retPD * c2) >> 8);
			pUnsharpMaskMap++;
		}
	}

	switch (Channels)
	{
	case 4:
	case 3:
	{
		unsigned char *Temp = (unsigned char *)malloc(Width * Height * (sizeof(unsigned char)));
		unsigned char *Blur = (unsigned char *)malloc(Width * Height * (sizeof(unsigned char)));
		if (Blur == NULL || Temp == NULL)
		{
			if (Blur)
			{
				free(Blur);
			}
			if (Temp)
			{
				free(Temp);
			}
			return;
		}
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pInput = Input + (Y * Stride);
			unsigned char *pTemp = Temp + (Y * Width);
			unsigned char *pBlur = Blur + (Y * Width);
			for (int X = 0; X < Width; X++)
			{
				pTemp[0] = (unsigned char)((19595 * pInput[0] + 38470 * pInput[1] + 7471 * pInput[2]) >> 16);

				pBlur[0] = pTemp[0];

				pInput += Channels;
				pTemp++;
				pBlur++;
			}
		}
		CPUImageGaussianBlurFilter(Temp, Blur, Width, Height, Width, GaussianSigma);
		unsigned char cb, cr;
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pInput = Input + (Y * Stride);
			unsigned char *pOutput = Output + (Y * Stride);
			unsigned char *pTemp = Temp + (Y * Width);
			unsigned char *pBlur = Blur + (Y * Width);
			for (int x = 0; x < Width; x++)
			{
				cb = (unsigned char)((36962 * (pInput[2] - (int)(pTemp[0])) >> 16) + 128);
				cr = (unsigned char)((46727 * (pInput[0] - (int)(pTemp[0])) >> 16) + 128);
				//锐化:高反差叠加

				unsigned char *pUnsharpMaskMap = unsharpMaskMap + (pTemp[0] << 8);

				ycbcr2rgb(pUnsharpMaskMap[pBlur[0]], cb, cr, &pOutput[0], &pOutput[1], &pOutput[2]);

				pTemp++;
				pBlur++;
				pOutput += Channels;
				pInput += Channels;
			}
		}
		free(Temp);
		free(Blur);
		break;
	}

	case 1:
	{
		unsigned char *Blur = (unsigned char *)malloc(Width * Height * (sizeof(unsigned char)));
		if (Blur == NULL)
		{
			return;
		}

		CPUImageGaussianBlurFilter(Input, Blur, Width, Height, Width, GaussianSigma);

		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pInput = Input + (Y * Width);
			unsigned char *pBlur = Blur + (Y * Width);
			unsigned char *pOutput = Output + (Y * Width);
			for (int x = 0; x < Width; x++)
			{
				//锐化:高反差叠加
				pOutput[0] =(unsigned char)(pInput[0] - pOutput[0] + 128);
				unsigned char *pUnsharpMaskMap = unsharpMaskMap + (pInput[0] << 8);
				pOutput[0] = pUnsharpMaskMap[pOutput[0]];

				pBlur++;
				pOutput++;
				pInput++;
			}
		}
		free(Blur);
	}

	break;

	default:
		break;
	}
}

static inline void boxfilterRow(const unsigned char *Input, unsigned char *Output, int Width, int Height, int Channels, int Radius)
{
	int iRadius = Radius + 1;
	int iScale = (int)((256.0f * 256.0f) / (2 * Radius + 1));
	int iWidthStep = Width * Channels;
	int iWidthStepDec = (Width - 1) * Channels;
	int iRadChannels = Radius * Channels;
	int iRadChannelsPlus = (iRadChannels + Channels);
	switch (Channels)
	{
	case 1:
	{
		for (int y = 0; y < Height; y++)
		{
			//  处理左边缘
			int iY = y * iWidthStep;
			int sum = Input[iY] * Radius;
			for (int x = 0; x < iRadius; x++)
			{
				int p = (y * Width + x) * Channels;
				sum += Input[p];
			}
			Output[iY] = (unsigned char)((sum * iScale) >> 16);
			for (int x = 1; x < iRadius; x++)
			{
				int pLeft = iY + x * Channels;
				int p0 = pLeft + iRadChannels;

				sum += Input[p0];
				sum -= Input[iY];
				Output[pLeft] =(unsigned char)((sum * iScale) >> 16);
			}

			// 核心区域
			for (int x = iRadius; x < Width - Radius; x++)
			{
				int pKernal = iY + x * Channels;

				int i0 = pKernal + iRadChannels;
				int i1 = pKernal - iRadChannelsPlus;

				sum += Input[i0];
				sum -= Input[i1];

				Output[pKernal] = (unsigned char)((sum * iScale) >> 16);
			}

			// 处理右边缘
			for (int x = Width - Radius; x < Width; x++)
			{
				int iRight = iY + x * Channels;
				int i0 = iY + iWidthStepDec;
				int i1 = iRight - iRadChannelsPlus;

				sum += Input[i0];
				sum -= Input[i1];
				Output[iRight] =(unsigned char)((sum * iScale) >> 16);
			}
		}
		break;
	}
	case 3:
	{
		for (int y = 0; y < Height; y++)
		{
			//  处理左边缘

			int iY = y * iWidthStep;
			int sumR = Input[iY] * Radius;
			int sumG = Input[iY + 1] * Radius;
			int sumB = Input[iY + 2] * Radius;
			for (int x = 0; x < iRadius; x++)
			{
				int i = iY + x * Channels;
				sumR += Input[i];
				sumG += Input[i + 1];
				sumB += Input[i + 2];
			}
			Output[iY] =(unsigned char) ((sumR * iScale) >> 16);
			Output[iY + 1] =(unsigned char) ((sumG * iScale) >> 16);
			Output[iY + 2] =(unsigned char) ((sumB * iScale) >> 16);
			for (int x = 1; x < iRadius; x++)
			{
				int iLeft = iY + x * Channels;
				int i0 = iLeft + iRadChannels;

				sumR += Input[i0];
				sumR -= Input[iY];
				sumG += Input[i0 + 1];
				sumG -= Input[iY + 1];
				sumB += Input[i0 + 2];
				sumB -= Input[iY + 2];
				Output[iLeft] =(unsigned char) ((sumR * iScale) >> 16);
				Output[iLeft + 1] =(unsigned char) ((sumG * iScale) >> 16);
				Output[iLeft + 2] =(unsigned char) ((sumB * iScale) >> 16);
			}

			// 核心区域
			for (int x = iRadius; x < Width - Radius; x++)
			{
				int iKernal = iY + x * Channels;

				int i0 = iKernal + iRadChannels;
				int i1 = iKernal - iRadChannelsPlus;

				sumR += Input[i0];
				sumR -= Input[i1];

				sumG += Input[i0 + 1];
				sumG -= Input[i1 + 1];

				sumB += Input[i0 + 2];
				sumB -= Input[i1 + 2];
				Output[iKernal] = (unsigned char)((sumR * iScale) >> 16);
				Output[iKernal + 1] = (unsigned char)((sumG * iScale) >> 16);
				Output[iKernal + 2] =(unsigned char)((sumB * iScale) >> 16);
			}

			// 处理右边缘
			for (int x = Width - Radius; x < Width; x++)
			{
				int iRight = iY + x * Channels;
				int i0 = iY + iWidthStepDec;
				int i1 = iRight - iRadChannelsPlus;

				sumR += Input[i0];
				sumR -= Input[i1];

				sumG += Input[i0 + 1];
				sumG -= Input[i1 + 1];

				sumB += Input[i0 + 2];
				sumB -= Input[i1 + 2];
				Output[iRight] =(unsigned char) ((sumR * iScale) >> 16);
				Output[iRight + 1] = (unsigned char) ((sumG * iScale) >> 16);
				Output[iRight + 2] = (unsigned char) ((sumB * iScale) >> 16);
			}
		}
		break;
	}
	case 4:
	{
		for (int y = 0; y < Height; y++)
		{
			//  处理左边缘
			int iY = y * iWidthStep;
			int sumR = Input[iY] * Radius;
			int sumG = Input[iY + 1] * Radius;
			int sumB = Input[iY + 2] * Radius;
			for (int x = 0; x < iRadius; x++)
			{
				int i = iY + x * Channels;
				sumR += Input[i];
				sumG += Input[i + 1];
				sumB += Input[i + 2];
			}
			Output[iY] = (unsigned char) ((sumR * iScale) >> 16);
			Output[iY + 1] = (unsigned char) ((sumG * iScale) >> 16);
			Output[iY + 2] =(unsigned char) ((sumB * iScale) >> 16);
			Output[iY + 3] = Input[iY + 3];
			for (int x = 1; x < iRadius; x++)
			{
				int iLeft = iY + x * Channels;
				int i0 = iLeft + iRadChannels;
				sumR += Input[i0];
				sumR -= Input[iLeft];
				sumG += Input[i0 + 1];
				sumG -= Input[iLeft + 1];
				sumB += Input[i0 + 2];
				sumB -= Input[iLeft + 2];
				Output[iLeft] = (unsigned char)((sumR * iScale) >> 16);
				Output[iLeft + 1] =(unsigned char) ((sumG * iScale) >> 16);
				Output[iLeft + 2] =(unsigned char) ((sumB * iScale) >> 16);
				Output[iLeft + 3] = Input[iLeft + 3];
			}

			// 核心区域
			for (int x = iRadius; x < Width - Radius; x++)
			{
				int iKernal = iY + x * Channels;

				int i0 = iKernal + iRadChannels;
				int i1 = iKernal - iRadChannelsPlus;

				sumR += Input[i0];
				sumR -= Input[i1];

				sumG += Input[i0 + 1];
				sumG -= Input[i1 + 1];

				sumB += Input[i0 + 2];
				sumB -= Input[i1 + 2];
				Output[iKernal] =(unsigned char) ( (sumR * iScale) >> 16);
				Output[iKernal + 1] =(unsigned char) ( (sumG * iScale) >> 16);
				Output[iKernal + 2] =(unsigned char) ( (sumB * iScale) >> 16);
				Output[iKernal + 3] = Input[iKernal + 3];
			}

			// 处理右边缘
			for (int x = Width - Radius; x < Width; x++)
			{
				int iRight = iY + x * Channels;
				int i0 = iY + iWidthStepDec;
				int i1 = iRight - iRadChannelsPlus;

				sumR += Input[i0];
				sumR -= Input[i1];

				sumG += Input[i0 + 1];
				sumG -= Input[i1 + 1];

				sumB += Input[i0 + 2];
				sumB -= Input[i1 + 2];
				Output[iRight] = (unsigned char) ((sumR * iScale) >> 16);
				Output[iRight + 1] =(unsigned char) ( (sumG * iScale) >> 16);
				Output[iRight + 2] =(unsigned char) ( (sumB * iScale) >> 16);
				Output[iRight + 3] = Input[iRight + 3];
			}
		}
		break;
	}
	default:
		break;
	}
}

static inline void boxfilterCol(const unsigned char *Input, unsigned char *Output, int Width, int Height, int Channels, int Radius)
{
	int iScale = (int)((256.0f * 256.0f) / (2 * Radius + 1));
	int iWidthStep = Width * Channels;
	int iWidthStepDec = (Height - 1) * iWidthStep;
	int iRadWidthStep = Radius * iWidthStep;
	int iRadWidthStepDec = (iRadWidthStep + iWidthStep);
	int iHeightRadius = Height - Radius;
	int iRadius = Radius + 1;
	switch (Channels)
	{
	case 1:
	{
		for (int x = 0; x < Width; x++)
		{
			//  处理左边缘
			int iX = x * Channels;
			int sum = Input[iX] * Radius;
			for (int y = 0; y < iRadius; y++)
			{
				int i = (y * Width + x) * Channels;
				sum += Input[i];
			}
			Output[x] = (unsigned char) ((sum * iScale) >> 16);

			for (int y = 1; y < iRadius; y++)
			{
				int i = iX + y * iWidthStep;

				int i0 = i + iRadWidthStep;
				int i1 = x * Channels;

				sum += Input[i0];
				sum -= Input[i1];
				Output[i] =(unsigned char) ( (sum * iScale) >> 16);
			}

			// 核心区域
			for (int y = iRadius; y < iHeightRadius; y++)
			{

				int iKernal = iX + y * iWidthStep;
				int i0 = iKernal + iRadWidthStep;
				int i1 = iKernal - iRadWidthStepDec;

				sum += Input[i0];
				sum -= Input[i1];
				Output[iKernal] = (unsigned char) ((sum * iScale) >> 16);
			}

			// 处理右边缘
			for (int y = iHeightRadius; y < Height; y++)
			{
				int iRight = iX + y * iWidthStep;

				int i0 = iWidthStepDec + x * Channels;
				int i1 = iRight - iRadWidthStepDec;

				sum += Input[i0];
				sum -= Input[i1];
				Output[iRight] =(unsigned char) ( (sum * iScale) >> 16);
			}
		}

		break;
	}
	case 3:
	{
		for (int x = 0; x < Width; x++)
		{
			//  处理左边缘
			int iX = x * Channels;
			int sumR = Input[iX] * Radius;
			int sumG = Input[iX + 1] * Radius;
			int sumB = Input[iX + 2] * Radius;
			for (int y = 0; y < iRadius; y++)
			{
				int i = iX + y * iWidthStep;
				sumR += Input[i];
				sumG += Input[i + 1];
				sumB += Input[i + 2];
			}
			Output[iX] = (unsigned char) ((sumR * iScale) >> 16);
			Output[iX + 1] = (unsigned char) ((sumG * iScale) >> 16);
			Output[iX + 2] = (unsigned char) ((sumB * iScale) >> 16);

			for (int y = 1; y < iRadius; y++)
			{
				int i = iX + y * iWidthStep;
				int i0 = i + iRadWidthStep;

				sumR += Input[i0];
				sumR -= Input[iX];
				sumG += Input[i0 + 1];
				sumG -= Input[iX + 1];
				sumB += Input[i0 + 2];
				sumB -= Input[iX + 2];
				Output[i] = (unsigned char) ((sumR * iScale) >> 16);
				Output[i + 1] =(unsigned char) ( (sumG * iScale) >> 16);
				Output[i + 2] =(unsigned char) ( (sumB * iScale) >> 16);
			}

			// 核心区域
			for (int y = iRadius; y < iHeightRadius; y++)
			{

				int iKernal = iX + y * iWidthStep;

				int i0 = iKernal + iRadWidthStep;
				int i1 = iKernal - iRadWidthStepDec;

				sumR += Input[i0];
				sumR -= Input[i1];
				sumG += Input[i0 + 1];
				sumG -= Input[i1 + 1];
				sumB += Input[i0 + 2];
				sumB -= Input[i1 + 2];
				Output[iKernal] = (unsigned char) ((sumR * iScale) >> 16);
				Output[iKernal + 1] =(unsigned char) ( (sumG * iScale) >> 16);
				Output[iKernal + 2] = (unsigned char) ((sumB * iScale) >> 16);
			}

			// 处理右边缘
			for (int y = iHeightRadius; y < Height; y++)
			{
				int iRight = iX + y * iWidthStep;
				int i0 = iWidthStepDec + iX;
				int i1 = iRight - iRadWidthStepDec;

				sumR += Input[i0];
				sumR -= Input[i1];
				sumG += Input[i0 + 1];
				sumG -= Input[i1 + 1];
				sumB += Input[i0 + 2];
				sumB -= Input[i1 + 2];
				Output[iRight] =(unsigned char) ( (sumR * iScale) >> 16);
				Output[iRight + 1] = (unsigned char) ((sumG * iScale) >> 16);
				Output[iRight + 2] = (unsigned char) ((sumB * iScale) >> 16);
			}
		}

		break;
	}
	case 4:
	{
		for (int x = 0; x < Width; x++)
		{
			//  处理左边缘
			int iX = x * Channels;
			int sumR = Input[iX] * Radius;
			int sumG = Input[iX + 1] * Radius;
			int sumB = Input[iX + 2] * Radius;
			for (int y = 0; y < iRadius; y++)
			{
				int i = iX + y * iWidthStep;
				sumR += Input[i];
				sumG += Input[i + 1];
				sumB += Input[i + 2];
			}
			Output[iX] = (unsigned char) ((sumR * iScale) >> 16);
			Output[iX + 1] =(unsigned char) ( (sumG * iScale) >> 16);
			Output[iX + 2] =(unsigned char) ( (sumB * iScale) >> 16);
			Output[iX + 3] = Input[iX + 3];
			for (int y = 1; y < iRadius; y++)
			{
				int i = iX + y * iWidthStep;
				int i0 = i + iRadWidthStep;
				sumR += Input[i0];
				sumR -= Input[iX];
				sumG += Input[i0 + 1];
				sumG -= Input[iX + 1];
				sumB += Input[i0 + 2];
				sumB -= Input[iX + 2];
				Output[i] =(unsigned char) ( (sumR * iScale) >> 16);
				Output[i + 1] = (unsigned char) ((sumG * iScale) >> 16);
				Output[i + 2] = (unsigned char) ((sumB * iScale) >> 16);
				Output[i + 3] = Input[i + 3];
			}

			// 核心区域
			for (int y = iRadius; y < iHeightRadius; y++)
			{

				int iKernal = iX + y * iWidthStep;
				int i0 = iKernal + iRadWidthStep;
				int i1 = iKernal - iRadWidthStepDec;
				sumR += Input[i0];
				sumR -= Input[i1];
				sumG += Input[i0 + 1];
				sumG -= Input[i1 + 1];
				sumB += Input[i0 + 2];
				sumB -= Input[i1 + 2];
				Output[iKernal] = (unsigned char) ((sumR * iScale) >> 16);
				Output[iKernal + 1] =(unsigned char) ( (sumG * iScale) >> 16);
				Output[iKernal + 2] = (unsigned char) ((sumB * iScale) >> 16);
				Output[iKernal + 3] = Input[iKernal + 3];
			}

			// 处理右边缘
			for (int y = iHeightRadius; y < Height; y++)
			{
				int iRight = iX + y * iWidthStep;

				int i0 = iWidthStepDec + iX;
				int i1 = iRight - iRadWidthStepDec;
				sumR += Input[i0];
				sumR -= Input[i1];
				sumG += Input[i0 + 1];
				sumG -= Input[i1 + 1];
				sumB += Input[i0 + 2];
				sumB -= Input[i1 + 2];
				Output[iRight] = (unsigned char) ((sumR * iScale) >> 16);
				Output[iRight + 1] = (unsigned char) ((sumG * iScale) >> 16);
				Output[iRight + 2] = (unsigned char) ((sumB * iScale) >> 16);
				Output[iRight + 3] = Input[iRight + 3];
			}
		}
		break;
	}
	default:
		break;
	}
}
/*
	CPUImageBoxBlurFilter: A hardware - optimized, variable - radius box blur
	texelSpacingMultiplier : A multiplier for the spacing between texels, ranging from 0.0 on up, with a default of 1.0.Adjusting this may slightly increase the blur strength, but will introduce artifacts in the result.Highly recommend using other parameters first, before touching this one.
	blurRadiusInPixels : A radius in pixels to use for the blur, with a default of 2.0.This adjusts the sigma variable in the Gaussian distribution function.
	blurPasses : The number of times to sequentially blur the incoming image.The more passes, the slower the filter.

	*/

void CPUImageBoxBlurFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, int Radius)
{
	int Channels = Stride / Width;
	unsigned char *temp = (unsigned char *)malloc((size_t)Width * Height * Channels);
	if (temp == NULL)
	{
		return;
	}
	boxfilterRow(Input, temp, Width, Height, Channels, Radius);
	boxfilterCol(temp, Output, Width, Height, Channels, Radius);
	free(temp);
}

void CPUImageSharpenFilter(unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float Radius, int sharpness, int intensity)
{
	int Channels = Stride / Width;
	intensity = max(min(intensity, 100), 0);
	int c1 = 256 * (100 - intensity) / 100;
	int c2 = 256 * (100 - (100 - intensity)) / 100;
	//锐化:高反差叠加
	unsigned char sharpnessMap[256 * 256] = {0};
	for (unsigned int PS = 0; PS < 256; PS++)
	{
		unsigned char *pSharpnessMap = sharpnessMap + (PS << 8);
		for (unsigned int PD = 0; PD < 256; PD++)
		{
			unsigned char retPD = ClampToByte((sharpness * (PS - PD)) + 128);
			retPD =(unsigned char) ((PS <= 128) ? (retPD * PS / 128) : (255 - (255 - retPD) * (255 - PS) / 128));
			//增强边缘法
			//   unsigned char retPD = ClampToByte(sharpness*(PS - PD) + PS);

			pSharpnessMap[0] = ClampToByte((PS * c1 + retPD * c2) >> 8);
			pSharpnessMap++;
		}
	}
	switch (Channels)
	{
	case 4:
	case 3:
	{
		unsigned char *temp = (unsigned char *)malloc(Width * Height * (sizeof(unsigned char)));
		unsigned char *blur = (unsigned char *)malloc(Width * Height * (sizeof(unsigned char)));
		if (blur == NULL || temp == NULL)
		{
			if (temp)
			{
				free(temp);
			}
			if (blur)
			{
				free(blur);
			}
			return;
		}
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pInput = Input + (Y * Stride);
			unsigned char *pTemp = temp + (Y * Width);
			unsigned char *pBlur = blur + (Y * Width);
			for (int X = 0; X < Width; X++)
			{
				pTemp[0] = (unsigned char)((19595 * pInput[0] + 38470 * pInput[1] + 7471 * pInput[2]) >> 16);
				pBlur[0] = pTemp[0];
				pInput += Channels;
				pTemp++;
				pBlur++;
			}
		}
		CPUImageBoxBlurFilter(temp, blur, Width, Height, Width, (int)Radius);
		unsigned char cb, cr;
		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pInput = Input + (Y * Stride);
			unsigned char *pOutput = Output + (Y * Stride);
			unsigned char *pTemp = temp + (Y * Width);
			unsigned char *pBlur = blur + (Y * Width);
			for (int x = 0; x < Width; x++)
			{
				cb = (unsigned char)((36962 * (pInput[2] - (int)(pTemp[0])) >> 16) + 128);
				cr = (unsigned char)((46727 * (pInput[0] - (int)(pTemp[0])) >> 16) + 128);
				//锐化:高反差叠加
				unsigned char *pSharpnessMap = sharpnessMap + (pTemp[0] << 8);

				ycbcr2rgb(pSharpnessMap[pBlur[0]], cb, cr, &pOutput[0], &pOutput[1], &pOutput[2]);

				pTemp++;
				pBlur++;
				pOutput += Channels;
				pInput += Channels;
			}
		}
		free(temp);
		free(blur);
		break;
	}

	case 1:
	{

		unsigned char *Blur = (unsigned char *)malloc(Width * Height * (sizeof(unsigned char)));
		if (Blur == NULL)
		{
			return;
		}

		CPUImageBoxBlurFilter(Input, Blur, Width, Height, Width, (int)Radius);

		for (int Y = 0; Y < Height; Y++)
		{
			unsigned char *pInput = Input + (Y * Width);
			unsigned char *pBlur = Blur + (Y * Width);
			unsigned char *pOutput = Output + (Y * Width);
			for (int x = 0; x < Width; x++)
			{
				unsigned char *pSharpnessMap = sharpnessMap + (pInput[0] << 8);
				pOutput[0] = pSharpnessMap[pOutput[0]];

				pBlur++;
				pOutput++;
				pInput++;
			}
		}
		free(Blur);
	}

	break;

	default:
		break;
	}
}

void CPUImageResamplingFilter(unsigned char *Input, unsigned int Width, unsigned int Height, unsigned int Stride, unsigned char *Output, int newWidth, int newHeight, int dstStride)
{
	int Channels = Stride / Width;
	int dstOffset = dstStride - Channels * newWidth;
	float xFactor = (float)Width / newWidth;
	float yFactor = (float)Height / newHeight;

	int ymax = Height - 1;
	int xmax = Width - 1;

	for (int y = 0; y < newHeight; y++)
	{
		float oy = (float)y * yFactor;
		int oy1 = (int)oy;
		int oy2 = (oy1 == ymax) ? oy1 : oy1 + 1;
		float dy1 = oy - (float)oy1;
		float dy2 = 1.0f - dy1;

		unsigned char *tp1 = Input + oy1 * Stride;
		unsigned char *tp2 = Input + oy2 * Stride;

		for (int x = 0; x < newWidth; x++)
		{
			float ox = (float)x * xFactor;
			int ox1 = (int)ox;
			int ox2 = (ox1 == xmax) ? ox1 : ox1 + 1;
			float dx1 = ox - (float)ox1;
			float dx2 = 1.0f - dx1;
			unsigned char *p1 = tp1 + ox1 * Channels;
			unsigned char *p2 = tp1 + ox2 * Channels;
			unsigned char *p3 = tp2 + ox1 * Channels;
			unsigned char *p4 = tp2 + ox2 * Channels;

			for (int i = 0; i < Channels; i++, Output++, p1++, p2++, p3++, p4++)
			{
				*Output = (unsigned char)(dy2 * (dx2 * (*p1) + dx1 * (*p2)) + dy1 * (dx2 * (*p3) + dx1 * (*p4)));
			}
		}
		Output += dstOffset;
	}
}

void CPUImageCropFilter(const unsigned char *Input, int Width, int Height, int srcStride, unsigned char *Output, int cropX, int cropY, int dstWidth, int dstHeight, int dstStride)
{
	int Channels = srcStride / Width;

	const unsigned char *src = Input + cropY * srcStride + cropX * Channels;
	unsigned char *dst = Output;

	for (int y = 0; y < dstHeight; y++)
	{
		memcpy(dst, src, dstStride);
		src += srcStride;
		dst += dstStride;
	}
}

void CPUImageAutoLevel(const unsigned char *Input, unsigned char *Output, int Width, int Height, int Stride, float fraction)
{
	int Channels = Stride / Width;
	switch (Channels)
	{
	case 4:
	case 3:
	{
		unsigned int histoRGB[256 * 3] = {0};
		unsigned int *histoR = &histoRGB[0];
		unsigned int *histoG = &histoRGB[256];
		unsigned int *histoB = &histoRGB[512];
		for (int Y = 0; Y < Height; Y++)
		{
			const unsigned char *pInput = Input + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				histoR[pInput[0]]++;
				histoG[pInput[1]]++;
				histoB[pInput[2]]++;
				pInput += Channels;
			}
		}
		int thresholdRMin = 0;
		int thresholdRMax = 0;
		int thresholdGMin = 0;
		int thresholdGMax = 0;
		int thresholdBMin = 0;
		int thresholdBMax = 0;
		int gap = (int)(fraction * Width * Height);
		int sumR = 0;
		int sumG = 0;
		int sumB = 0;

		for (int i = 0; sumR < gap; ++i)
		{
			sumR += histoR[i];
			thresholdRMin = i;
		}
		sumR = 0;
		for (int i = 255; sumR < gap; --i)
		{
			sumR += histoR[i];
			thresholdRMax = i;
		}
		for (int i = 0; sumG < gap; ++i)
		{
			sumG += histoG[i];
			thresholdGMin = i;
		}
		sumG = 0;
		for (int i = 255; sumG < gap; --i)
		{
			sumG += histoG[i];
			thresholdGMax = i;
		}
		for (int i = 0; sumB < gap; ++i)
		{
			sumB += histoB[i];
			thresholdBMin = i;
		}
		sumB = 0;
		for (int i = 255; sumB < gap; --i)
		{
			sumB += histoB[i];
			thresholdBMax = i;
		}
		unsigned char MapRGB[256 * 3] = {0};
		unsigned char *MapB = &MapRGB[0];
		unsigned char *MapG = &MapRGB[256];
		unsigned char *MapR = &MapRGB[512];
		for (int i = 0; i < 256; i++)
		{
			if (i < thresholdRMin)
				MapR[i] = (i + 0) >> 1;
			else if (i > thresholdRMax)
				MapR[i] = (255);
			else
				MapR[i] = ClampToByte((int)((i - thresholdRMin) * 255.0) / (thresholdRMax - thresholdRMin));
			if (i < thresholdGMin)
				MapG[i] = (i + 0) >> 1;
			else if (i > thresholdGMax)
				MapG[i] = (255);
			else
				MapG[i] = ClampToByte((int)((i - thresholdGMin) * 255.0) / (thresholdGMax - thresholdGMin));
			if (i < thresholdBMin)
				MapB[i] = (0);
			else if (i > thresholdBMax)
				MapB[i] = (255);
			else
				MapB[i] = ClampToByte((int)((i - thresholdBMin) * 255.0) / (thresholdBMax - thresholdBMin));
		}

		for (int Y = 0; Y < Height; Y++)
		{
			const unsigned char *pInput = Input + (Y * Stride);
			unsigned char *pOutput = Output + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				pOutput[0] = MapR[pInput[0]];
				pOutput[1] = MapG[pInput[1]];
				pOutput[2] = MapB[pInput[2]];
				pInput += Channels;
				pOutput += Channels;
			}
		}
		break;
	}
	case 1:
	{
		unsigned int histoGray[256] = {0};
		for (int Y = 0; Y < Height; Y++)
		{
			const unsigned char *pInput = Input + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				histoGray[pInput[0]]++;
				pInput++;
			}
		}
		int thresholdMin = 0;
		int thresholdMax = 0;
		int gap = (int)(fraction * Width * Height);
		int sumGray = 0;
		for (int i = 0; sumGray < gap; ++i)
		{
			sumGray += histoGray[i];
			thresholdMin = i;
		}
		sumGray = 0;
		for (int i = 255; sumGray < gap; --i)
		{
			sumGray += histoGray[i];
			thresholdMax = i;
		}
		unsigned char MapGray[256] = {0};
		if ((thresholdMax - thresholdMin) <= 0)
			return;
		for (int i = 0; i < 256; i++)
		{
			if (i < thresholdMin)
				MapGray[i] = (0);
			else if (i > thresholdMax)
				MapGray[i] = (255);
			else
				MapGray[i] = ClampToByte((int)((i - thresholdMin) * 255.0f) / (thresholdMax - thresholdMin));
		}

		for (int Y = 0; Y < Height; Y++)
		{
			const unsigned char *pInput = Input + (Y * Stride);

			unsigned char *pOutput = Output + (Y * Stride);
			for (int X = 0; X < Width; X++)
			{
				pOutput[0] = MapGray[pInput[0]];
				pInput++;
				pOutput++;
			}
		}
	}
	break;

	default:
		break;
	}
}

void CPUImageSobelEdge(unsigned char *Input, unsigned char *Output, int Width, int Height)
{
	if ((Input == NULL) || (Output == NULL))
		return;
	if ((Width <= 0) || (Height <= 0))
		return;

	unsigned char *SqrLut = (unsigned char *)malloc(65026 * sizeof(unsigned char));
	unsigned char *RowCopy = (unsigned char *)malloc((Width + 2) * 3 * sizeof(unsigned char));
	if ((SqrLut == NULL) || (RowCopy == NULL))
	{
		if (SqrLut != NULL)
			free(SqrLut);
		if (RowCopy != NULL)
			free(RowCopy);
	}

	unsigned char *First = RowCopy, *Second = RowCopy + (Width + 2), *Third = RowCopy + (Width + 2) * 2;

	for (int Y = 0; Y < 65026; Y++)
		SqrLut[Y] = (unsigned char)ClampToByte((sqrtf(Y + 0.0f) + 0.5f));

	memcpy(Second, Input, 1);
	memcpy(Second + 1, Input, Width);
	memcpy(Second + Width + 1, Input + Width - 1, 1);

	memcpy(First, Second, Width + 2);

	memcpy(Third, Input + Width, 1);
	memcpy(Third + 1, Input + Width, Width);
	memcpy(Third + Width + 1, Input + Width + Width - 1, 1);

	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *LinePS = Input + Y * Width;
		unsigned char *LinePD = Output + Y * Width;
		if (Y != 0)
		{
			unsigned char *Temp = First;
			First = Second;
			Second = Third;
			Third = Temp;
		}
		if (Y == Height - 1)
		{
			memcpy(Third, Second, Width + 2);
		}
		else
		{
			memcpy(Third, Input + (Y + 1) * Width, 1);
			memcpy(Third + 1, Input + (Y + 1) * Width, Width);
			memcpy(Third + Width + 1, Input + (Y + 1) * Width + Width - 1, 1);
		}
		for (int X = 0; X < Width; X++)
		{
			int Hori = First[X] + 2 * First[X + 1] + First[X + 3] - (Third[X] + 2 * Third[X + 1] + Third[X + 2]);
			int Vert = First[X] + 2 * Second[X] + Third[X] - (First[X + 2] + 2 * Second[X + 2] + Third[X + 2]);
			int Value = Hori * Hori + Vert * Vert;
			LinePD[X] = SqrLut[min(Value, 65025)];
		}
	}
	if (RowCopy)
	{
		free(RowCopy);
	}
	if (SqrLut)
	{
		free(SqrLut);
	}
}

int CPUImageHoughLines(unsigned char *Input, int Width, int Height, int lineIntensity, int Threshold, float resTheta, int numLine, float *Radius, float *Theta)
{
	int halfHoughWidth = (int)(sqrtf((float)(Width * Width + Height * Height)));
	int houghWidth = halfHoughWidth * 2;
	int maxTheta = (int)(180.0f / resTheta + 0.5f);
	int houghMapSize = houghWidth * maxTheta;
	unsigned short *houghMap = (unsigned short *)calloc((size_t)houghMapSize, sizeof(unsigned short));
	float *sinLUT = (float *)calloc((size_t)maxTheta, sizeof(float));
	float *cosLUT = (float *)calloc((size_t)maxTheta, sizeof(float));
	if (sinLUT == NULL || cosLUT == NULL || houghMap == NULL)
	{
		if (houghMap)
		{
			free(houghMap);
		}
		if (cosLUT)
		{
			free(cosLUT);
		}
		if (sinLUT)
		{
			free(sinLUT);
		}
		return 0;
	}
	float thetaStep = M_PI / maxTheta;
	for (int theta = 0; theta < maxTheta; theta++)
	{
		sinLUT[theta] = (float)fastSin(theta * thetaStep);
		cosLUT[theta] = (float)fastCos(theta * thetaStep);
	}

	for (int y = 0; y < Height; y++)
	{
		unsigned char *pIn = Input + (y * Width);
		for (int x = 0; x < Width; x++)
		{
			if (pIn[x] > Threshold)
			{
				for (int theta = 0; theta < maxTheta; theta++)
				{
					int r = (int)(x * sinLUT[theta] + y * cosLUT[theta] + halfHoughWidth + 0.5f);
					houghMap[r * maxTheta + theta]++;
				}
			}
		}
	}

	int nLine = 0;
	for (int i = 0; i < houghMapSize && nLine < numLine; i++)
	{
		if (houghMap[i] > lineIntensity)
		{
			Radius[nLine] = (float)(i / maxTheta);
			Theta[nLine] = (i - Radius[nLine] * maxTheta) * resTheta;
			Radius[nLine] -= halfHoughWidth;
			nLine++;
		}
	}

	if (houghMap)
	{
		free(houghMap);
	}
	if (cosLUT)
	{
		free(cosLUT);
	}
	if (sinLUT)
	{
		free(sinLUT);
	}

	return nLine;
}

void CPUImageDrawLine(unsigned char *canvas, int width, int height, int stride, int x1, int y1, int x2, int y2, unsigned char R, unsigned char G, unsigned char B)
{
	int channels = stride / width;

	int xs, ys, xe, ye;
	if (x1 == x2)
	{
		if (y1 < y2)
		{
			ys = y1;
			ye = y2;
		}
		else
		{
			ys = y2;
			ye = y1;
		}
		unsigned char *Line = canvas + x1 * channels;
		for (int r = ys; r <= ye; r++)
		{
			unsigned char *curLine = Line + r * stride;

			curLine[0] = R;
			curLine[1] = G;
			curLine[2] = B;
		}
		return;
	}

	float a = (float)(y2 - y1) / (x2 - x1);
	int nHeight = height;

	if ((a > -1) && (a < 1))
	{
		if (x1 < x2)
		{
			xs = x1;
			xe = x2;
			ys = y1;
			ye = y2;
		}
		else
		{
			xs = x2;
			xe = x1;
			ys = y2;
			ye = y1;
		}
		for (int c = xs; c <= xe; c++)
		{
			unsigned char *Line = canvas + c * channels;
			int r = (int)(a * (c - xs) + ys + 0.5f);
			if (r < 0 || r >= nHeight)
				continue;
			unsigned char *curLine = Line + r * stride;
			curLine[0] = R;
			curLine[1] = G;
			curLine[2] = B;
		}
	}
	else
	{
		float invA = 1.0f / a;
		if (y1 < y2)
		{
			ys = y1;
			ye = y2;
			xs = x1;
			xe = x2;
		}
		else
		{
			ys = y2;
			ye = y1;
			xs = x2;
			xe = x1;
		}
		for (int r = ys; r <= ye; r++)
		{
			int c = (int)(invA * (r - ys) + xs + 0.5f);
			unsigned char *Line = canvas + c * channels;
			if (r < 0 || r >= nHeight)
				continue;
			unsigned char *curLine = Line + r * stride;
			curLine[0] = R;
			curLine[1] = G;
			curLine[2] = B;
		}
	}
}

bool CPUImageGetImageSize(const char *file_path, int *width, int *height, int *file_size)
{
	bool has_image_size = false;
	*height = -1;
	*width = -1;
	*file_size = -1;
	FILE *fp = fopen(file_path, "rb");
	if (fp == NULL)
		return has_image_size;
	struct stat st;
	char sigBuf[26];
	if (fstat(fileno(fp), &st) < 0)
	{
		fclose(fp);
		return has_image_size;
	}
	else
	{
		*file_size = (int)st.st_size;
	}
	if (fread(&sigBuf, 26, 1, fp) < 1)
	{
		fclose(fp);
		return has_image_size;
	}
	const char *png_signature = "\211PNG\r\n\032\n";
	const char *ihdr_signature = "IHDR";
	const char *gif87_signature = "GIF87a";
	const char *gif89_signature = "GIF89a";
	const char *jpeg_signature = "\377\330";
	const char *bmp_signature = "BM";
	if ((*file_size >= 10) && (memcmp(sigBuf, gif87_signature, strlen(gif87_signature)) == 0 || memcmp(sigBuf, gif89_signature, strlen(gif89_signature)) == 0))
	{
		// image type: gif
		unsigned short *size_info = (unsigned short *)(sigBuf + 6);
		*width = size_info[0];
		*height = size_info[1];
		has_image_size = true;
	}
	else if ((*file_size >= 24) && (memcmp(sigBuf, png_signature, strlen(png_signature)) == 0 && memcmp(sigBuf + 12, ihdr_signature, strlen(ihdr_signature)) == 0))
	{
		// image type:   png
		unsigned long *size_info = (unsigned long *)(sigBuf + 16);
		*width = (int)byteswap_ulong(size_info[0]);
		*height = (int)byteswap_ulong(size_info[1]);
		has_image_size = true;
	}
	else if ((*file_size >= 16) && (memcmp(sigBuf, png_signature, strlen(png_signature)) == 0))
	{
		// image type: old png
		unsigned long *size_info = (unsigned long *)(sigBuf + 8);
		*width = (int)byteswap_ulong(size_info[0]);
		*height = (int)byteswap_ulong(size_info[1]);
		has_image_size = true;
	}
	else if ((*file_size >= 2) && (memcmp(sigBuf, jpeg_signature, strlen(jpeg_signature)) == 0))
	{
		// image type: jpeg
		printf("Jpeg");
		fseek(fp, 0, SEEK_SET);
		char b = 0;
		fread(&sigBuf, 2, 1, fp);
		fread(&b, 1, 1, fp);
		int w = -1;
		int h = -1;
		while (b && ((unsigned char)b & 0xff) != 0xDA)
		{
			while (((unsigned char)b & 0xff) != 0xFF)
			{
				fread(&b, 1, 1, fp);
			}
			while (((unsigned char)b & 0xff) == 0xFF)
			{
				fread(&b, 1, 1, fp);
			}
			if (((unsigned char)b & 0xff) >= 0xC0 && ((unsigned char)b & 0xff) <= 0xC3)
			{
				fread(&sigBuf, 3, 1, fp);
				fread(&sigBuf, 4, 1, fp);
				unsigned short *size_info = (unsigned short *)(sigBuf);
				h = byteswap_ushort(size_info[0]);
				w = byteswap_ushort(size_info[1]);
				break;
			}
			else
			{
				unsigned short chunk_size = 0;
				fread(&chunk_size, 2, 1, fp);
				if (fseek(fp, byteswap_ushort(chunk_size) - 2, SEEK_CUR) != 0)
					break;
			}
			fread(&b, 1, 1, fp);
		}
		if (w != -1 && h != -1)
		{
			*width = w;
			*height = h;
		}
		has_image_size = true;
	}
	else if ((*file_size >= 26) && (memcmp(sigBuf, bmp_signature, strlen(bmp_signature)) == 0))
	{
		// image type: bmp
		unsigned int header_size = (unsigned int)(*(sigBuf + 14));
		if (header_size == 12)
		{
			unsigned short *size_info = (unsigned short *)(sigBuf + 18);
			*width = size_info[0];
			*height = size_info[1];
		}
		else if (header_size >= 40)
		{
			unsigned int *size_info = (unsigned int *)(sigBuf + 18);
			*width = size_info[0];
			*height = Abs((size_info[1]));
		}
		has_image_size = true;
	}
	else if (*file_size >= 2)
	{
		// image type: ico
		fseek(fp, 0, SEEK_SET);
		unsigned short format = 0;
		unsigned short reserved = 0;
		fread(&reserved, 2, 1, fp);
		fread(&format, 2, 1, fp);
		if (reserved == 0 && format == 1)
		{
			unsigned short num = 0;
			fread(&num, 2, 1, fp);
			if (num > 1)
			{
				printf("this is a muti-ico file.");
			}
			else
			{
				char w = 0, h = 0;
				fread(&w, 1, 1, fp);
				fread(&h, 1, 1, fp);
				*width = (int)((unsigned char)w & 0xff);
				*height = (int)((unsigned char)h & 0xff);
			}
		}
		has_image_size = true;
	}
	if (fp != NULL)
		fclose(fp);
	return has_image_size;
}
#ifdef __cplusplus
}
#endif