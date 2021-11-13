/**
 * @file BatchMandelCalculator.cc
 * @author Patrik Nemeth <xnemet04@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date 11.11.2021
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>

#include "BatchMandelCalculator.h"

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	constexpr size_t alignment = 64;

	size_t size = height * width * sizeof(int);
	size_t misalignment = (size % alignment);
	int padding = misalignment > 0 ? alignment - misalignment : 0;
	data = (int *)(aligned_alloc(alignment, size + padding));
	int *pdata = data;
	// Pre-fill with `limit`
	#pragma omp simd aligned(pdata : 64) simdlen(64)
	for (size_t i = 0; i < width * height; i++) {
		pdata[i] = limit;
	}

	size = width * sizeof(bool);
	misalignment = (size % alignment);
	padding = misalignment > 0 ? alignment - misalignment : 0;
	limitLock = (bool *)(aligned_alloc(alignment, size + padding));

	size = width * sizeof(float);
	misalignment = (size % alignment);
	padding = misalignment > 0 ? alignment - misalignment : 0;
	zImags = (float *)(aligned_alloc(alignment, size + padding));
	zReals = (float *)(aligned_alloc(alignment, size + padding));
}

BatchMandelCalculator::~BatchMandelCalculator() {
	free(limitLock);
	free(zImags);
	free(zReals);
	free(data);
	limitLock = NULL;
	zImags = NULL;
	zReals = NULL;
	data = NULL;
}

int * BatchMandelCalculator::calculateMandelbrot () {
	// Create pointers in local scope for the alignment pragma
	bool *plimitLock = limitLock;
	float *pzImags = zImags;
	float *pzReals = zReals;

	// Reduce precision in favor of performance
	float f_x_start = x_start;
	float f_y_start = y_start;
	float f_dx = dx;
	float f_dy = dy;

	constexpr int blockSize = 64;

	for (int blockI = 0; blockI < height / blockSize; blockI++)
	{
		for (int blockJ = 0; blockJ < width / blockSize; blockJ++)
		{
			for (int i = 0, acc = 0; i < blockSize; i++, acc = 0)
			{
				// Fixes alignment issue in the vectorized loop
				int iProper = blockI * blockSize + i;
				int *pdata = data + iProper * width + (blockJ * blockSize);

				for (int k = 0; (k < limit) && (acc < blockSize); k++)
				{
					#pragma omp simd simdlen(64) \
							aligned(plimitLock, pdata, pzImags, pzReals : 64) \
							reduction(+ : acc)
					for (int j = 0; j < blockSize; j++)
					{
						float jProper = blockJ * blockSize + j;
						// The first `limit` iteration is unique
						pzReals[j] = k == 0 ? f_x_start + jProper * f_dx : pzReals[j];
						pzImags[j] = k == 0 ? f_y_start + iProper * f_dy : pzImags[j];
						plimitLock[j] = k == 0 ? true : plimitLock[j];

						float r2 = pzReals[j] * pzReals[j];
						float i2 = pzImags[j] * pzImags[j];

						acc += r2 + i2 > 4.0f && plimitLock[j];

						pdata[j] = r2 + i2 > 4.0f && plimitLock[j]
							? k
							: pdata[j];
						plimitLock[j] = r2 + i2 > 4.0f && plimitLock[j]
							? false
							: plimitLock[j];

						pzImags[j] =
							2.0f * pzReals[j] * pzImags[j] + (f_y_start + iProper * f_dy);
						pzReals[j] = r2 - i2 + (f_x_start + jProper * f_dx);
					}
				}
			}
		}
	}

	// If width and height are not divisible by blockSize,
	// then the remaining incomplete blocks at the ends of rows and columns
	// need to be calculated separately.

	const int heightRemainder = (height / blockSize) * blockSize;
	const int widthRemainder = (width / blockSize) * blockSize;

	for (int i = 0, acc = 0; i < heightRemainder; i++, acc = 0)
	{
		// Fixes alignment issue in the vectorized loop
		int *pdata = data + i * width;

		for (int k = 0; (k < limit) && (acc < widthRemainder); k++)
		{
			#pragma omp simd \
					aligned(plimitLock, pdata, pzImags, pzReals : 64) \
					reduction(+ : acc)
			for (int j = widthRemainder; j < width; j++)
			{
				// The first `limit` iteration is unique
				pzReals[j] = k == 0 ? f_x_start + j * f_dx : pzReals[j];
				pzImags[j] = k == 0 ? f_y_start + i * f_dy : pzImags[j];
				plimitLock[j] = k == 0 ? true : plimitLock[j];

				float r2 = pzReals[j] * pzReals[j];
				float i2 = pzImags[j] * pzImags[j];

				acc += r2 + i2 > 4.0f && plimitLock[j];

				pdata[j] = r2 + i2 > 4.0f && plimitLock[j]
					? k
					: pdata[j];
				plimitLock[j] = r2 + i2 > 4.0f && plimitLock[j]
					? false
					: plimitLock[j];

				pzImags[j] =
					2.0f * pzReals[j] * pzImags[j] + (f_y_start + i * f_dy);
				pzReals[j] = r2 - i2 + (f_x_start + j * f_dx);
			}
		}
	}

	for (int i = heightRemainder, acc = 0; i < height; i++, acc = 0)
	{
		// Fixes alignment issue in the vectorized loop
		int *pdata = data + i * width;

		for (int k = 0; (k < limit) && (acc < width); k++)
		{
			#pragma omp simd \
					aligned(plimitLock, pdata, pzImags, pzReals : 64) \
					reduction(+ : acc)
			for (int j = 0; j < width; j++)
			{
				// The first `limit` iteration is unique
				pzReals[j] = k == 0 ? f_x_start + j * f_dx : pzReals[j];
				pzImags[j] = k == 0 ? f_y_start + i * f_dy : pzImags[j];
				plimitLock[j] = k == 0 ? true : plimitLock[j];

				float r2 = pzReals[j] * pzReals[j];
				float i2 = pzImags[j] * pzImags[j];

				acc += r2 + i2 > 4.0f && plimitLock[j];

				pdata[j] = r2 + i2 > 4.0f && plimitLock[j]
					? k
					: pdata[j];
				plimitLock[j] = r2 + i2 > 4.0f && plimitLock[j]
					? false
					: plimitLock[j];

				pzImags[j] =
					2.0f * pzReals[j] * pzImags[j] + (f_y_start + i * f_dy);
				pzReals[j] = r2 - i2 + (f_x_start + j * f_dx);
			}
		}
	}

	return data;
}