#pragma once
#ifndef Image_Classifier_HEADER
#define Image_Classifier_HEADER
#include <thread>
#include "Dictionary.h"
#include "Matrix.h"
#include "NN_Tools.h"
#include "Activations.h"
#include "DataSet.h"


typedef Dictionary<string, Matrix*> Mat_Dictionary;

class Image_Classifier
{
private:
	Arguments * Arg;
	DatasetParam* DP;
	bool momentum;				  // Indicates whether momentum is used or not
	bool isLastepoch;			  // Label for the last epoch
	int  Cores;                   // The number of allowed threads in the current hardware for maximum efficiency
	Matrix*** D;		          // Dropout Matrices in fully connected layers
	matrix<bool>**** D2;          // DropConnect Matrices in fully connected layers

	//Fully connected Dictionaries
	Mat_Dictionary      FC_Parameters;		  // Dictionary containing weights and biases of fully connected layers
	Mat_Dictionary*     FC_Cache;		      // Dictionary containing temporaral values of internal activations of fully connected layers
	Mat_Dictionary*     FC_Grades;			  // Dictionary containing gradients of weights and biases of fully connected layers
	Mat_Dictionary      FC_ADAM;              // Dictionary containing ADAM gradients

public:
	// Interface functions
	Image_Classifier(Arguments* A, DatasetParam* D);         // Initialize the network with arguments A
	void train();									    // Begin training
	void test(Mode devOrtest);							// Test the network
	void StoreParameters();							    // Store the network's parameters
	void RetrieveParameters();							// Retrieve stored parametrs
	void TestParameters();								// Test the network's parameters
	void Print();										// Print the network's parameters


private:
	// Fully connected main functions
	void init_FC();                           // Initialize the fully connected network
	void train_FC();                          // Train the fully connected network in a single thread
	void test_FC(Mode mode);				  // Test the fully connected network on either dev or test sets

	// Fully connected feed forward and back propagation functions
	Matrix* FC_FeedForward(Mode mode, int ThreadNum);
	void    FC_CalGrads(BoolMatrix* cur_Y, Matrix* Y_hat, int ThreadNum);
	void    FC_UpdateParameters(int iteration, int ThreadNum);

};
#endif // !Image_Classifier_HEADER

