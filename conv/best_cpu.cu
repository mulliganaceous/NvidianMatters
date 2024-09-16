/* ------------
 * This code is provided solely for the personal and private use of
 * students taking the CSC367H5 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited.
 * All forms of distribution of this code, whether as given or with
 * any changes, are expressly prohibited.
 *
 * Authors: Bogdan Simion, Felipe de Azevedo Piovezan
 *
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2019 Bogdan Simion
 * -------------
 */

#include "kernels.h"
#include <pthread.h>
#include <stdio.h>

#define STEPPRINTF(...)   ;//fprintf(stdout,__VA_ARGS__)
#define THREADPRINTF(...) ;//fprintf(stdout,__VA_ARGS__)
#define NORMALIZATION 1

typedef struct filter_t {
  int32_t dimension;
  const int8_t *matrix;
} our_filter;

typedef struct common_work_t {
	const our_filter *f;
	const int32_t *original_image;
	int32_t *output_image;
	int32_t width;
	int32_t height;
	int32_t max_threads;
	char ran;
	int32_t largest;
	int32_t smallest;
	void *common_info;
	pthread_barrier_t *barrier;
} common_work;

typedef struct work_t {
	common_work *common;
	int32_t id;
	int32_t method;
	char ran;
	int32_t largest;
	int32_t smallest;
	void *thread_info;
	work_t *otherwork;
} work;

typedef struct queue_work_t {
	int32_t work_chunk;
	int32_t chunks;
	int32_t *tilerow;
	int32_t *tilecol;
	int32_t nextchunk;
	pthread_mutex_t mutex;
} queuework;


/*************** COMMON WORK ***********************/
/* Normalizes a pixel given the smallest and largest integer values
 * in the image */
void normalize_pixel(int32_t *target, int64_t pixel_idx, int32_t smallest, int32_t largest) {
	if (smallest == largest) {
		return;
	}
	target[pixel_idx] = ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}

/* Processes a single pixel and returns the value of processed pixel */
int64_t apply2d(const our_filter *f, 
	const int32_t *original, int32_t *target,
	int32_t width, int32_t height, int row, int column) {
	int32_t pbounds = (f->dimension-1)/2;
	int64_t pix = 0;
	for (int ip = 0; ip < f->dimension; ip++) {
		for (int jp = 0; jp < f->dimension; jp++) {
			int row_cur = row + ip - pbounds;
			int col_cur = column + jp - pbounds;
			if (row_cur >= 0 && row_cur < height && col_cur >= 0 && col_cur < width) {
				pix += f->matrix[f->dimension*ip + jp]*original[width*row_cur + col_cur];
				STEPPRINTF("%d %d %d %d (%d) (%d) = %d\n", row, column, row_cur, col_cur, f->matrix[f->dimension*ip + jp], f->matrix[f->dimension*ip + jp]*original[width*row_cur + col_cur], pix);
			}
		}
	}
	target[width*row + column] = pix;
	return pix;
}

void determine_largest_smallest_pixel(work *pargs) {
	THREADPRINTF("\t[%d,] tries to determine largest/smallest pixel.\n", pargs->id);
	if (pargs->common->ran) return;
	for (int k = 0; k < pargs->common->max_threads; k++) {
		if (pargs->otherwork[k].ran) {
			if (!pargs->common->ran|| pargs->otherwork[k].largest > pargs->common->largest)
				pargs->common->largest = pargs->otherwork[k].largest;
			if (!pargs->common->ran|| pargs->otherwork[k].smallest < pargs->common->smallest)
				pargs->common->smallest = pargs->otherwork[k].smallest;
		}
		pargs->common->ran = 1;
		THREADPRINTF("\t[%d,%d] Min: %d, Max: %d (%d %d).\n", pargs->id, k, pargs->otherwork[k].smallest, pargs->otherwork[k].largest, pargs->common->smallest, pargs->common->largest);
	}
}

/***************** WORK QUEUE *******************/
/* You don't have to implement this. It is just a suggestion for the
 * organization of the code.
 */
void *queue_work(void *args) {
	work *pargs; 
	pargs = (work *) args;
	queuework *queueinfo = (queuework *) pargs->common->common_info;
	int32_t tilecur = pargs->id;
	int64_t pix;
	THREADPRINTF("Thread %d running workqueue on a %d×%d image with %d threads starting from tile %d.\n", pargs->id, pargs->common->height, pargs->common->width, pargs->common->max_threads, tilecur);
	
	// Continually process until queue is empty
	while (tilecur < queueinfo->chunks) {
		THREADPRINTF("\t[%d] Running at tile %ds...\n", pargs->id, tilecur);
		for (int i_tile = 0; i_tile < queueinfo->work_chunk && i_tile + queueinfo->tilerow[tilecur] < pargs->common->height; i_tile++) {
			for (int j_tile = 0; j_tile < queueinfo->work_chunk && j_tile + queueinfo->tilecol[tilecur] < pargs->common->width; j_tile++) {
				pix = apply2d(pargs->common->f, 
					pargs->common->original_image, 
					pargs->common->output_image, 
					pargs->common->width, 
					pargs->common->height,
					i_tile + queueinfo->tilerow[tilecur], j_tile + queueinfo->tilecol[tilecur]);
					STEPPRINTF("\t[%d,%d] (%4d %4d) %d\n", pargs->id, tilecur, i_tile + queueinfo->tilerow[tilecur], j_tile + queueinfo->tilecol[tilecur], pix);
				if (!pargs->ran || pix < pargs->smallest) pargs->smallest = pix;
				if (!pargs->ran || pix > pargs->largest) pargs->largest = pix;
				pargs->ran = true;
			}
		}
		// Perform mutex once a thread finishes its chunk so no conflicts occur upon updating next chunk.
		pthread_mutex_lock(&queueinfo->mutex);
		tilecur = queueinfo->nextchunk;
		queueinfo->nextchunk++;
		pthread_mutex_unlock(&queueinfo->mutex);
	}
	THREADPRINTF("Thread %d finished phase 1 at tile %ds...\n", pargs->id, tilecur);

	// Reset current tile to the id, and reset next chunk to thread count. Perform mutex until we can determine largest/smallest pixel. Only one thread gets to determine largest/smallest pixel.
	pthread_barrier_wait(pargs->common->barrier);
	tilecur = pargs->id;
	queueinfo->nextchunk = pargs->common->max_threads;
	pthread_mutex_lock(&queueinfo->mutex);
	determine_largest_smallest_pixel(pargs);
	pthread_mutex_unlock(&queueinfo->mutex);

	// Normalization phase
	pthread_barrier_wait(pargs->common->barrier);
	while (NORMALIZATION && tilecur < queueinfo->chunks) {
		THREADPRINTF("\t[%d] Normalizing at tile %ds...\n", pargs->id, tilecur);
		for (int i_tile = 0; i_tile < queueinfo->work_chunk && i_tile + queueinfo->tilerow[tilecur] < pargs->common->height; i_tile++) {
			for (int j_tile = 0; j_tile < queueinfo->work_chunk && j_tile + queueinfo->tilecol[tilecur] < pargs->common->width; j_tile++) {
				normalize_pixel(pargs->common->output_image, 
					pargs->common->width*(i_tile + queueinfo->tilerow[tilecur]) + (j_tile + queueinfo->tilecol[tilecur]), 
					pargs->common->smallest, pargs->common->largest);
			}
		}
		// Perform mutex once a thread finishes its chunk so no conflicts occur upon updating next chunk.
		pthread_mutex_lock(&queueinfo->mutex);
		tilecur = queueinfo->nextchunk;
		queueinfo->nextchunk++;
		pthread_mutex_unlock(&queueinfo->mutex);
	}
	THREADPRINTF("Thread %d Finished normalization at tile %ds with min %d and max %d.\n", pargs->id, tilecur, pargs->common->smallest, pargs->common->largest);
	
	pthread_exit(0);
	return NULL; 
}


void run_best_cpu(const int8_t *filter, int32_t dimension, const int32_t *input, int32_t *output, 
	int32_t width, int32_t height, int32_t *smallest, int32_t *largest) 
{
	// Set up filter as in f
    our_filter *f = (our_filter *)malloc(sizeof(our_filter));
    f->dimension = dimension;
    f->matrix = filter;

    // Parameters for work_chunk size and number of threads               
    int work_chunk = 16;
    int num_threads = 8;

    // Set up thread-creation variables
	pthread_t th[num_threads];
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	// Set up barrier
	pthread_barrier_t barrier;
    int rc;

	// Set up common work info
	common_work *workinfo = (common_work *)malloc(sizeof(common_work));
	workinfo->f = f;
	workinfo->original_image = input;
	workinfo->output_image = output;
	workinfo->width = width;
	workinfo->height = height;
	workinfo->max_threads = num_threads;
	workinfo->ran = 0;
	workinfo->largest = -1;
	workinfo->smallest = -1;
	workinfo->barrier = &barrier;

	work_t *pargs = (work_t *)malloc(num_threads*sizeof(work_t));
    int32_t tiles = (1+(width-1)/work_chunk)*(1+(height-1)/work_chunk);
    int32_t *tilerow = (int32_t *)malloc(tiles*sizeof(int32_t));
    int32_t *tilecol = (int32_t *)malloc(tiles*sizeof(int32_t));
    int32_t row_cur = 0;
    int32_t col_cur = 0;
    for (int k = 0; k < tiles; k++) { // On current tile row
        if (col_cur < width) {
            tilerow[k] = row_cur;
            tilecol[k] = col_cur;
            col_cur += work_chunk;
        }
        else if (col_cur >= width) { // Next tile row
            row_cur += work_chunk;
            col_cur = 0;
            tilerow[k] = row_cur;
            tilecol[k] = col_cur;
            col_cur += work_chunk;
        }
    }

    // Perform queued work
    if (num_threads >= tiles) workinfo->max_threads = tiles;
    queuework *queueinfo = (queuework *)malloc(sizeof(queuework));
    queueinfo->work_chunk = work_chunk;
    queueinfo->chunks = tiles;
    queueinfo->tilerow = tilerow;
    queueinfo->tilecol = tilecol;
    queueinfo->nextchunk = workinfo->max_threads;

    // Set up mutex and barrier
    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);
    queueinfo->mutex = mutex;
    pthread_barrier_init(&barrier, NULL, workinfo->max_threads);
    workinfo->common_info = queueinfo;
    
    // Perform loop. Handle edge case where more threads than pixels.
    if (num_threads >= queueinfo->chunks) workinfo->max_threads = queueinfo->chunks;
    for (int tid = 0; tid < num_threads && tid < workinfo->max_threads; tid++) {
        // printf("%s\n", "create");
        pargs[tid].common = workinfo;
        pargs[tid].id = tid;
        pargs[tid].ran = 0;
        pargs[tid].largest = -1;
        pargs[tid].smallest = -1;
        pargs[tid].otherwork = pargs;
        // Create thread, then check for abnormal exit or incomplete execution
        rc = pthread_create(&th[tid], &attr, queue_work, (void*)(&pargs[tid]));
        if (rc) {
            printf("%s\n", "pthread");
            exit(-1);
        }
    }
    // Join threads
    pthread_attr_destroy(&attr);
    for (int tid = 0; tid < num_threads && tid < workinfo->max_threads; tid++) {
        rc = pthread_join(th[tid], NULL);
        if (rc) {
            printf("%s\n", "pthread");
            exit(-1);
        }
    }

    // Declare smallest and largest, which is used for normalization
    *smallest = workinfo->smallest;
    *largest = workinfo->largest;

    // Free queue information, work information, and the filter
    free(queueinfo);
    free(tilerow);
    free(tilecol);

    free(pargs);
    free(workinfo);
    free(f);

}

/***** GPU REDUCTION *****/
/** Given an array of int32_t values, obtain the minimum (sgn==1) or maximum (sgn==-1) value
 */
__global__ void minmaxreduce(int32_t *d_input, int32_t *d_output, char sgn, unsigned n) {
    /* Partial result are placed in shared memory. */
    extern __shared__ int32_t s_mm[];

    /* First run; given block size of s1, it can support an array of size 2*s1 */
    const unsigned s1 = blockDim.x;
    const unsigned bId = blockIdx.x;
    const unsigned tId = threadIdx.x;
    const unsigned index = 2*s1*bId + tId;
    if (index < n)
        s_mm[tId] = d_input[index];
    if (index + s1 < n)
        if (sgn*d_input[index + s1] < sgn*s_mm[tId]) s_mm[tId] = d_input[index + s1];
    __syncthreads();

    /* Subsequent runs */
    for (unsigned s = s1/2; s > 32; s>>=1) {
        if (index + s < n && tId < s)
            if (sgn*d_input[index + s] < sgn*s_mm[tId]) s_mm[tId] = d_input[index + s];
        __syncthreads();
    }
    if (tId < 32) {
        if (s1 >= 64 && index + 32 < n)
            if (sgn*s_mm[tId + 32] < sgn*s_mm[tId]) s_mm[tId] = s_mm[tId + 32];
        if (s1 >= 32 && index + 16 < n)
            if (sgn*s_mm[tId + 16] < sgn*s_mm[tId]) s_mm[tId] = s_mm[tId + 16];
        if (s1 >= 16 && index + 8 < n)
            if (sgn*s_mm[tId + 8] < sgn*s_mm[tId]) s_mm[tId] = s_mm[tId + 8];
        if (s1 >= 8 && index + 4 < n)
            if (sgn*s_mm[tId + 4] < sgn*s_mm[tId]) s_mm[tId] = s_mm[tId + 4];
        if (s1 >= 4 && index + 2 < n)
            if (sgn*s_mm[tId + 2] < sgn*s_mm[tId]) s_mm[tId] = s_mm[tId + 2];
        if (s1 >= 2 && index + 1 < n)
            if (sgn*s_mm[tId + 1] < sgn*s_mm[tId]) s_mm[tId] = s_mm[tId + 1];
        }

        /* Copy partial result representing a block */
        if (tId == 0) {
            d_output[bId] = s_mm[tId];
    }
}

/** Perform normalization
 */
__global__ void normalize(int32_t *d_output, int32_t *d_min, int32_t *d_max, unsigned n) {
    const unsigned index = blockDim.x*blockIdx.x + threadIdx.x;
    if (d_min[0] < d_max[0])
    	d_output[index] = ((d_output[index] - d_min[0])*255)/(d_max[0] - d_min[0]);
}

