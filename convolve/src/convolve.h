#pragma once

extern unsigned verbose;

// return number of elements, not bytes
typedef unsigned (*CONV_readBuffer_Callback)(double buffer[], unsigned const nElem);
typedef unsigned (*CONV_writeBuffer_Callback)(double const buffer[], unsigned const nElem);

int CONV_setupConvolution(unsigned inA_Size, unsigned inB_Size,
                          CONV_readBuffer_Callback inA_ReadCallback, CONV_readBuffer_Callback inB_ReadCallback,
                          CONV_writeBuffer_Callback out_WriteCallback);

int CONV_endConvolution(void);
int CONV_doConvolution(void);
