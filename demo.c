//如果是Windows的话，调用系统API ShellExecuteA打开图片
#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#define access _access
#else
#include <unistd.h>
#endif
#include "browse.h"
#define USE_SHELL_OPEN
#include "cpuimage.h"
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
//ref:https://github.com/nothings/stb/blob/master/stb_image.h
#define TJE_IMPLEMENTATION
#include "tiny_jpeg.h"
//ref:https://github.com/serge-rgb/TinyJPEG/blob/master/tiny_jpeg.h
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
//计时
#include <stdint.h>
#if defined(__APPLE__)
#include <mach/mach_time.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else // __linux
#include <time.h>
#ifndef CLOCK_MONOTONIC //_RAW
#define CLOCK_MONOTONIC CLOCK_REALTIME
#endif
#endif
static uint64_t nanotimer()
{
	static int ever = 0;
#if defined(__APPLE__)
	static mach_timebase_info_data_t frequency;
	if (!ever)
	{
		if (mach_timebase_info(&frequency) != KERN_SUCCESS)
		{
			return 0;
		}
		ever = 1;
	}
	return mach_absolute_time() * frequency.numer / frequency.denom;
#elif defined(_WIN32)
	static LARGE_INTEGER frequency;
	if (!ever)
	{
		QueryPerformanceFrequency(&frequency);
		ever = 1;
	}
	LARGE_INTEGER t;
	QueryPerformanceCounter(&t);
	return (t.QuadPart * (uint64_t)1e9) / frequency.QuadPart;
#else // __linux
	struct timespec t;
	if (!ever)
	{
		if (clock_gettime(CLOCK_MONOTONIC, &spec) != 0)
		{
			return 0;
		}
		ever = 1;
	}
	clock_gettime(CLOCK_MONOTONIC, &spec);
	return (t.tv_sec * (uint64_t)1e9) + t.tv_nsec;
#endif
}

static double now()
{
	static uint64_t epoch = 0;
	if (!epoch)
	{
		epoch = nanotimer();
	}
	return (nanotimer() - epoch) / 1e9;
};

double calcElapsed(double start, double end)
{
	double took = -start;
	return took + end;
}
#ifndef _MAX_DRIVE
#define _MAX_DRIVE 3
#endif
#ifndef _MAX_FNAME
#define _MAX_FNAME 256
#endif
#ifndef _MAX_EXT
#define _MAX_EXT 256
#endif
#ifndef _MAX_DIR
#define _MAX_DIR 256
#endif
//存储当前传入文件位置的变量
char saveFile[1024];
//加载图片
unsigned char *loadImage(const char *filename, int *Width, int *Height, int *Channels)
{

	return stbi_load(filename, Width, Height, Channels, 0);
}
//保存图片
void saveImage(const char *filename, int Width, int Height, int Channels, unsigned char *Output)
{

	memcpy(saveFile + strlen(saveFile), filename, strlen(filename));
	*(saveFile + strlen(saveFile) + 1) = 0;
	//保存为jpg
	if (!tje_encode_to_file(saveFile, Width, Height, Channels, true, Output))
	{
		fprintf(stderr, "写入 JPEG 文件失败.\n");
		return;
	}

#ifdef USE_SHELL_OPEN
    browse(saveFile);
#endif
}

//分割路径函数
void splitpath(const char *path, char *drv, char *dir, char *name, char *ext)
{
	const char *end;
	const char *p;
	const char *s;
	if (path[0] && path[1] == ':')
	{
		if (drv)
		{
			*drv++ = *path++;
			*drv++ = *path++;
			*drv = '\0';
		}
	}
	else if (drv)
		*drv = '\0';
	for (end = path; *end && *end != ':';)
		end++;
	for (p = end; p > path && *--p != '\\' && *p != '/';)
		if (*p == '.')
		{
			end = p;
			break;
		}
	if (ext)
		for (s = end; (*ext = *s++);)
			ext++;
	for (p = end; p > path;)
		if (*--p == '\\' || *p == '/')
		{
			p++;
			break;
		}
	if (name)
	{
		for (s = p; s < end;)
			*name++ = *s++;
		*name = '\0';
	}
	if (dir)
	{
		for (s = path; s < p;)
			*dir++ = *s++;
		*dir = '\0';
	}
}

//取当前传入的文件位置
void getCurrentFilePath(const char *filePath, char *saveFile)
{
	char drive[_MAX_DRIVE];
	char dir[_MAX_DIR];
	char fname[_MAX_FNAME];
	char ext[_MAX_EXT];
	splitpath(filePath, drive, dir, fname, ext);
	int n = strlen(filePath);
	memcpy(saveFile, filePath, n);
	char *cur_saveFile = saveFile + (n - strlen(ext));
	cur_saveFile[0] = '_';
	cur_saveFile[1] = 0;
}

int main(int argc, char **argv)
{
	printf("Image Processing \n ");
	printf("博客:http://tntmonks.cnblogs.com/ \n ");
	printf("支持解析如下图片格式: \n ");
	printf("JPG, PNG, TGA, BMP, PSD, GIF, HDR, PIC \n ");

	//检查参数是否正确
	if (argc < 2)
	{
		printf("参数错误。 \n ");
		printf("请拖放文件到可执行文件上，或使用命令行：demo.exe 图片 \n ");
		printf("请拖放文件例如: demo.exe d:\\image.jpg \n ");

		return 0;
	}

	char *szfile = argv[1];
	//检查输入的文件是否存在
	if (access(szfile, 0) == -1)
	{
		printf("输入的文件不存在，参数错误！ \n ");
	}

	getCurrentFilePath(szfile, saveFile);

	int Width = 0;					  //图片宽度
	int Height = 0;					  //图片高度
	int Channels = 0;				  //图片通道数
	unsigned char *inputImage = NULL; //输入图片指针

	int filesize = 0;
	CPUImageGetImageSize(szfile, &Width, &Height, &filesize);
	printf("file:%s\nfilesize:%d\nwidth:%d\nheight:%d\n", szfile, filesize, Width, Height);

	double startTime = now();
	//加载图片
	inputImage = loadImage(szfile, &Width, &Height, &Channels);

	double nLoadTime = calcElapsed(startTime, now());
	printf("加载耗时: %d 毫秒!\n ", (int)(nLoadTime * 1000));
	if ((Channels != 0) && (Width != 0) && (Height != 0))
	{
		//分配与载入同等内存用于处理后输出结果
		unsigned char *outputImg = (unsigned char *)stbi__malloc(Width * Channels * Height * sizeof(unsigned char));
		if (inputImage)
		{
			//如果图片加载成功，则将内容复制给输出内存，方便处理
			memcpy(outputImg, inputImage, Width * Channels * Height);
		}
		else
		{
			printf("加载文件: %s 失败!\n ", szfile);
		}
		startTime = now();
		float arrRho[100];
		float arrTheta[100];
		int nTNum = 200;
		int nTVal = 100;
		float Theta = 1.0f;
		CPUImageGrayscaleFilter(inputImage, outputImg, Width, Height, Width * Channels);
		CPUImageSobelEdge(outputImg, outputImg, Width, Height);
		int nLine = CPUImageHoughLines(outputImg, Width, Height, nTNum, nTVal, Theta, 100, arrRho, arrTheta);
		memcpy(outputImg, inputImage, Width * Channels * Height);
		for (int i = 0; i < nLine; i++)
		{
			if (arrTheta[i] == 90)
			{
				CPUImageDrawLine(outputImg, Width, Height, Width * Channels, (int)arrRho[i], 0, (int)arrRho[i], Height - 1, 255, 0, 0);
			}
			else
			{
				int x1 = 0;
				int y1 = (int)(arrRho[i] / fastCos(arrTheta[i] * M_PI / 180.0f) + 0.5f);
				int x2 = Width - 1;
				int y2 = (int)((arrRho[i] - x2 * fastSin(arrTheta[i] * M_PI / 180.0f)) / fastCos(arrTheta[i] * M_PI / 180.0f) + 0.5f);
				CPUImageDrawLine(outputImg, Width, Height, Width * Channels, x1, y1, x2, y2, 255, 0, 0);
			}
		}
		//处理算法
		double nProcessTime = now();
		printf("处理耗时: %d 毫秒!\n ", (int)(nProcessTime * 1000));
		//保存处理后的图片
		startTime = now();

		saveImage("_done.jpg", Width, Height, Channels, outputImg);
		double nSaveTime = calcElapsed(startTime, now());

		printf("保存耗时: %d 毫秒!\n ", (int)(nSaveTime * 1000));
		//释放占用的内存
		if (outputImg)
		{
			stbi_image_free(outputImg);
			outputImg = NULL;
		}

		if (inputImage)
		{
			stbi_image_free(inputImage);
			inputImage = NULL;
		}
	}
	else
	{
		printf("加载文件: %s 失败!\n", szfile);
	}

	getchar();
	printf("按任意键退出程序 \n");

	return EXIT_SUCCESS;
}