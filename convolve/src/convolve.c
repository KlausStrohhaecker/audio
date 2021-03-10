#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <mem.h>
#include <math.h>
#include <stdint.h>
#include <sys/time.h>

#include "convolve.h"

// ---------- LOCAL VARIABLES -----------

static unsigned                  inS_size;
static unsigned                  inK_size;
static CONV_readBuffer_Callback  inS_readCallback;
static CONV_readBuffer_Callback  inK_readCallback;
static CONV_writeBuffer_Callback out_writeCallback;

static double const* inS;
static double const* inK;
static double*       out;
static double*       tmp;
static double*       tail;

static unsigned allocSize;

// ---------- LOCAL PROTOTYPES -----------

static void     freeBuffers(void);
static int      allocBuffers(void);
static uint64_t getTimeUSec(void);

void     TDConvolve(double result[], double const a[], double const b[], unsigned const size, double tmp[]);
void     setEffort(unsigned const count);
unsigned getTDConvolveBufferSize(unsigned const requestedSize);

// ---------- PUBLIC FUNCTIONS -----------

unsigned verbose;

// when a read callback are NULL, all zeros is assumed for input data
// when write callback is NULL, output is simply ignored
int CONV_setupConvolution(unsigned inA_Size, unsigned inB_Size,
                          CONV_readBuffer_Callback inA_ReadCallback, CONV_readBuffer_Callback inB_ReadCallback,
                          CONV_writeBuffer_Callback const out_WriteCallback)
{
  if (inA_Size < 2 || inB_Size < 2)
    return -1;

  CONV_endConvolution();

  // if B is larger then let A be the kernel and B the signal
  int const swap    = (inB_Size > inA_Size);
  inS_size          = swap ? inB_Size : inA_Size;
  inK_size          = swap ? inA_Size : inB_Size;
  inS_readCallback  = swap ? inB_ReadCallback : inA_ReadCallback;
  inK_readCallback  = swap ? inA_ReadCallback : inB_ReadCallback;
  out_writeCallback = out_WriteCallback;

  if (verbose)
    printf(" signal=%u kernel=%u%s, output=%u", inS_size, inK_size, swap ? " (files swapped)" : "", inS_size + inK_size - 1);
  allocSize      = getTDConvolveBufferSize(inK_size);
  unsigned count = ((inS_size) + (allocSize - 1)) / allocSize;
  if (verbose)
    printf(", blocks=%u, RAM buffers=%.1lfMB\n", count, (double) allocSize * 7.0 * sizeof(double) / 1000000.0);
  setEffort(count);

  if (!allocBuffers())
    return -2;

  return 1;
}

int CONV_endConvolution(void)
{
  freeBuffers();
  return 1;
}

int CONV_doConvolution(void)
{
  uint64_t time = getTimeUSec();
  // fill kernel buffer and pad to allocSize with zeros
  unsigned elems = 0;
  if (inK_readCallback)
    elems = (*inK_readCallback)((double*) inK, inK_size);                // returns actual # of elements read
  memset((void*) &inK[elems], 0, (allocSize - elems) * sizeof(inK[0]));  // clear remaining values

  if (verbose)
    printf("   0%%");
  int first = 1;
  while (1)
  {
    // read in an allocSize'd chunk of signal
    elems = 0;
    if (inS_readCallback)
      elems = (*inS_readCallback)((double*) inS, allocSize);               // returns actual # of elements read
    memset((void*) &inS[elems], 0, (allocSize - elems) * sizeof(inS[0]));  // clear remaining values

    // convolve the chunks
    TDConvolve(out, inS, inK, allocSize, tmp);

    // add in previous tail
    if (!first)
      for (unsigned i = 0; i < inK_size - 1; i++)  // tail is one sample shorter than the kernel
        out[i] += tail[i];                         // buffer 1st half contains the signal
    first = 0;

    // write out data (only the signal part of the buffer)
    unsigned wrElems = elems;
    if (out_writeCallback)
      wrElems = (*out_writeCallback)((double*) out, elems);  // returns actual # of elements written
    if (wrElems != elems)
      return -1;

    inS_size -= elems;

    if (inS_size > 0)  // signal NOT finished ?
    {
      // backup new tail
      for (unsigned i = 0; i < inK_size - 1; i++)  // tail is one sample shorter than the kernel
        tail[i] = out[i + allocSize];              // buffer 2nd half contains the tail
      continue;                                    // next chunk
    }

    // write out tail and terminate
    if (out_writeCallback)
      (*out_writeCallback)((double*) &out[elems], inK_size - 1);  // tail is one sample shorter than the kernel
    time = getTimeUSec() - time;
    if (verbose)
      printf(" done in %.2lfsec\n", (double) time / 1e6);
    return 1;
  }  // while (1);
}

// ---------- LOCAL FUNCTIONS -----------

static void freeBuffers(void)
{
  tail = tail ? free((void*) tail), NULL : NULL;
  tmp  = tmp ? free((void*) tmp), NULL : NULL;
  out  = out ? free((void*) out), NULL : NULL;
  inK  = inK ? free((void*) inK), NULL : NULL;
  inS  = inS ? free((void*) inS), NULL : NULL;
}

static int allocBuffers(void)
{
  // determine alloc size from kernel (which is <= signal)
  inS  = (double const*) calloc(allocSize, sizeof(*inS));
  inK  = (double const*) calloc(allocSize, sizeof(*inK));
  out  = (double*) calloc(allocSize * 2, sizeof(*out));
  tmp  = (double*) calloc(allocSize * 2, sizeof(*tmp));
  tail = (double*) calloc(allocSize, sizeof(*tail));
  if (!inS || !inK || !out || !tmp || !tail)
  {
    freeBuffers();
    return 0;
  }
  return 1;
}

static uint64_t getTimeUSec(void)
{
#ifdef __MINGW32__

  struct timeval  tv;
  struct timezone tz;
  if (gettimeofday(&tv, &tz))
    return 0;
  else
    return 1000000ul * (uint64_t)(tv.tv_sec) + (uint64_t)(tv.tv_usec);

#else

  struct timespec tp;

  if (clock_gettime(CLOCK_MONOTONIC, &tp))
    return 0;
  else
    return 1000000ul * (uint64_t)(tp.tv_sec) + (uint64_t)(tp.tv_nsec) / 1000ul;

#endif
}

// --------------------------------------------------------------------------------------------
//    Time-Domain Convolver Core
// --------------------------------------------------------------------------------------------

#define CONV_DIRECT_MUL_SIZE (16)

// local prototypes
static inline void _mul_knuth(double r[], double const a[], double const b[], unsigned const w, double tmp[]);

static inline void mul_brute(double r[], double const a[], double const b[], unsigned const w);

static uint64_t effort;
static uint64_t directCntr;
static uint64_t directCntrThreshold;

#define log2_of_3 1.5849625007211561814537389439478

//--------------------------- PUBLIC FUNCTIONS -------------------------

void setEffort(unsigned const count)
{
  directCntr          = 0;
  directCntrThreshold = (double) effort * (double) count / 100.1;
}

unsigned getTDConvolveBufferSize(unsigned const requestedSize)
{  // Find the smallest possible partition size >= requesteSize
  //  Any valid partition size is of the form K * 2^N, with K being <= direct mul size, and N > 0
  unsigned directMulSize = requestedSize;
  unsigned powerOfTwo    = 0;
  while (directMulSize > CONV_DIRECT_MUL_SIZE)
  {
    if (directMulSize & 1)  // odd ?
      directMulSize++;      // +1 to obtain next multiple of 2
    directMulSize /= 2;
    powerOfTwo++;
  }

  unsigned const result = directMulSize * (1 << powerOfTwo);
  if (verbose)
    printf(", atom size=%u (of %u..%u), nesting=%u\n ==> partition size=%u",
           directMulSize, CONV_DIRECT_MUL_SIZE / 2 + 1, CONV_DIRECT_MUL_SIZE, powerOfTwo, result);

  effort = pow((1 << powerOfTwo), log2_of_3);

  return result;
}

void TDConvolve(double result[], double const a[], double const b[], unsigned const size, double tmp[])
{  // tmp and r must be of length 2*w.
  // w must be a value obtained by 'getConvolveBufferSize(requestedSize)'
  // note the last value of r is zero by definition
  _mul_knuth(result, a, b, size, tmp);
}

//--------------------------- LOCAL FUNCTIONS -------------------------

// source: https://www.musicdsp.org/en/latest/Filters/66-time-domain-convolution-with-o-n-log2-3.html
static void _mul_knuth(double r[], double const a[], double const b[], unsigned const w, double tmp[])
{
  if (w > CONV_DIRECT_MUL_SIZE)
  {
    unsigned const m = w >> 1;

    for (unsigned i = 0; i < m; i++)
      r[i] = a[m + i] - a[i];
    for (unsigned i = 0; i < m; i++)
      r[i + m] = b[i] - b[m + i];

    _mul_knuth(tmp, r, r + m, m, tmp + w);
    _mul_knuth(r, a, b, m, tmp + w);
    _mul_knuth(r + w, a + m, b + m, m, tmp + w);

    for (unsigned i = 0; i < m; i++)
    {
      double s = r[m + i] + r[w + i];
      r[m + i] = s + r[i] + tmp[i];
      r[w + i] = s + r[w + m + i] + tmp[i + m];
    }
    return;
  }
  memset(r + w, 0, w * sizeof(r[0]));
  mul_brute(r, a, b, w);
}

static inline void mul_brute(double r[], double const a[], double const b[], unsigned const w)
{
  static int   percent;
  double const ai = a[0];
  for (unsigned j = 0; j < w; j++)
    r[j] = ai * b[j];

  for (unsigned i = 1; i < w; i++)
  {
    double* const rr = r + i;
    double const  ai = a[i];
    for (unsigned j = 0; j < w; j++)
      rr[j] += ai * b[j];
  }

  if (verbose)
  {
    if (++directCntr < directCntrThreshold)
      return;
    directCntr -= directCntrThreshold;
    printf("\x08\x08\x08\x08%3i%%", ++percent);
  }
}
