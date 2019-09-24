#pragma once
#ifndef DATASET_H_INCLUDED
#define DATASET_H_INCLUDED
#include <fstream>
#include <vector>
#include <conio.h>
#include <thread>
#include "Matrix.h"
#include "NN_Tools.h"
//////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////  GET DATASET FROM HARD DISK   ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void Prepare_TrainSet(Arguments& Arg, DatasetParam& DP);     // Reads the TrainSet from the hard disk and puts them in matrix X and matrix Y
void Prepare_TestSet(Arguments& Arg, DatasetParam& DP);      // Reads the TestSet from the hard disk and puts them in matrix X and matrix Y
void Prepare_TrainSet1(Arguments& Arg, DatasetParam& DP);
void Prepare_TestSet1(Arguments& Arg, DatasetParam& DP);
void SWAP(Matrix* MAT, int i, int k);                        // Swaps the ith column with the kth column in MAT
void SWAP(BoolMatrix* MAT, int i, int k);
Matrix* Normalize(Matrix* X);                                // Normalize DataSet
Matrix* Normalize_01(Matrix* X);
#endif // DATASET_H_INCLUDED
