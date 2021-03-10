#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <unistd.h>
#include <malloc.h>

#include "convolve.h"

#define VERSION " -- convolve V1.0 -- \n"

static void error(char const *const format, ...)
{
  fprintf(stderr, VERSION);
  va_list ap;
  fflush(stdout);
  va_start(ap, format);
  vfprintf(stderr, format, ap);
  va_end(ap);
  putc('\n', stderr);
  fflush(stderr);
}

static void usage(void)
{
  fflush(stderr);
  puts(" Usage: convolve  [-q]  <signal-file-1>  <signal-file-2>  <output-file>\n"
       "  file format for all is 'plain array of double' (8bytes per value)"
       "  -q : quiet operation");
}

FILE *signalFile;
FILE *kernelFile;
FILE *outputFile;

static unsigned ReadSignal(double buffer[], unsigned nElem)
{
  return fread(buffer, sizeof(buffer[0]), nElem, signalFile);
}

static unsigned ReadKernel(double buffer[], unsigned nElem)
{
  return fread(buffer, sizeof(buffer[0]), nElem, kernelFile);
}

static unsigned WriteOutput(double const buffer[], unsigned nElem)
{
  return fwrite(buffer, sizeof(buffer[0]), nElem, outputFile);
}

int main(int argc, char *argv[])
{
  if (argc != 4 && argc != 5)
  {
    error(" Wrong number of arguments!");
    usage();
    return 3;
  }

  verbose = 1;
  if (argc == 5)
  {
    if (strcmp(argv[1], "-q") == 0)
      verbose = 0;
    argv++;
    argc--;
  }

  char const *const signalFileName = argv[1];
  char const *const kernelFileName = argv[2];
  char const *const outputFileName = argv[3];
  if (verbose)
    printf(VERSION);

  // Open Signal File
  signalFile = fopen(signalFileName, "rb");
  if (NULL == signalFile)
  {
    error(" Could not open signal-1 file '%s'", signalFileName);
    return 3;
  }
  fseek(signalFile, 0L, SEEK_END);
  long signalFileSize = ftell(signalFile);
  rewind(signalFile);
  if (signalFileSize <= 0 || ((signalFileSize & 0b111) != 0))
  {
    printf("%li\n", signalFileSize);
    error(" Wrong signal-1 file size %li Bytes (must be > 0 and a multiple of 8)", signalFileSize);
    return 3;
  }
  signalFileSize /= sizeof(double);

  // Open Kernel File
  kernelFile = fopen(kernelFileName, "rb");
  if (NULL == signalFile)
  {
    error(" Could not open signal-2 file '%s'", kernelFileName);
    return 3;
  }
  fseek(kernelFile, 0L, SEEK_END);
  long kernelFileSize = ftell(kernelFile);
  rewind(kernelFile);
  if (kernelFileSize <= 0 || ((kernelFileSize & 0b111) != 0))
  {
    error(" Wrong signal-2 file size %l Bytes (must be > 0 and a multiple of 8)", signalFileSize);
    fclose(signalFile);
    fclose(kernelFile);
    return 3;
  }
  kernelFileSize /= sizeof(double);

  // Open Output File
  outputFile = fopen(outputFileName, "wb");
  if (NULL == signalFile)
  {
    error(" Could not open output file for writing '%s'", outputFileName);
    fclose(signalFile);
    return 3;
  }

  // convolve
  if (CONV_setupConvolution(signalFileSize, kernelFileSize, ReadSignal, ReadKernel, WriteOutput) <= 0)
  {
    error(" Could not set up convolution (out of memory)");
    goto _exit;
  }

  if (CONV_doConvolution() <= 0)
  {
    error(" Could not process convolution (file write error)");
    goto _exit;
  }

_exit:
  CONV_endConvolution();
  fclose(signalFile);
  fclose(kernelFile);
  fclose(outputFile);

  return 0;
}
