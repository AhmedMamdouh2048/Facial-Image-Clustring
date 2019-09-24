#ifndef NN_TOOLS_H_INCLUDED
#define NN_TOOLS_H_INCLUDED
#include <iostream>
#include "Matrix.h"
#include "Activations.h"
#include <thread>
//////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////  Matrix types   //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
typedef matrix<float> Matrix;
typedef matrix<bool> BoolMatrix;
typedef matrix<unsigned char> U_IntMatrix;
//////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////  Enumerations   //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
enum TypeOfNet { FC, LENET1, CUSTOM };
enum Optimizer { ADAM, GRADIENT_DESCENT };
enum ErrorType { SQAURE_ERROR, CROSS_ENTROPY };
enum Mode { TRAIN, DEV, TEST, MAX, AVG };
//////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////  Structures   ///////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
struct layer
{
	float neurons;
	ActivationType activation;
	float numOfLinearNodes;

	void put(float n, ActivationType activ)
	{
		neurons = n;
		activation = activ;
	}
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////
struct DatasetParam
{
	uint32_t Train_Examples;
	uint32_t Test_Examples;
	uint32_t ImageDim;
	uint32_t ImagesOfPerson_test;
	int CompressedImageSize;
	int ImagesOfPerson;
	int ImagesOfOthers;
	uint32_t BIG_FILE;
	uint32_t KDEF;
	uint32_t CFEED;
	uint32_t AR;
	uint32_t TEST_FILE;
	int numFiles;
	int curFile;
	const char* CompressedImages_dir;
	const char** X_dir;
	const char** Y_dir;
	const char* Xtest_img_dir;
	const char* Xtest_activ_dir;
	const char* ClusteredImagesPath;
	const char* ParametersPath;
	bool Get_dataSet;
	bool AllPossibilities;
	bool shuffle;
	bool normalize_01;
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////
struct Arguments
{
	TypeOfNet NetType;                  // Type of neural network. Either FC (fully connected) or LENET1 (convolutional)
	Optimizer optimizer;                // The type of optimizer. Either Gradient Descent or ADAM
	ErrorType ErrType;                  // The type of error used. Either square error or cross entropy
	layer* layers;                      // The activations and number of neurons in each layer
	int numOfLayers;                    // The number of layers
	int numOfEpochs;                    // The total number of epochs required for training
	int batchSize;                      // The batch size
	int numPrint;                       // The rate of printing the accuracy on the screen
	int Test_Batch_Size;				// The size of the test batch for cost calculation
	float learingRate;                  // The learning rate alpha
	float decayRate;                    // The decay of learining rate
	float curLearningRate;              // The current value of learning rate
	float regularizationParameter;      // The regularization parameter lambda
	float curCost;                      // Current value of cost function
	float prevCost;                     // Previous value of cost function
	float threshold;                    // A threshold in error calculation
	float numErrors;                    // Number of patterns that have error larger than threshold
	float* keep_prob;                   // The probabilty distribution of drop-out and drop-connection across the layers
	bool batchNorm;                     // Is batch normalization activated or not
	bool dropout;                       // Is drop-out activated or not
	bool dropConnect;                   // Is drop-connection activated or not
	bool SaveParameters;                // Save the parameters or not
	bool RetrieveParameters;            // Retrieve the parameters or not
	bool TestParameters;				// Test the retrieved parameters
	string ActivationsPath;             // The path of the file at which the actications are saved at
	Matrix* X;                          // The input dataset
	BoolMatrix* Y;                      // The labels of the input dataset
	Matrix* X_dev;                      // The development set
	BoolMatrix* Y_dev;                  // The labels of the development dataset
	Matrix* X_test;                     // The test set
	BoolMatrix* Y_test;                 // The labels of the test set
	U_IntMatrix* X_test_img;
	bool negative;
	Arguments() : curCost(9999), prevCost(9999)
	{}
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////  String generation   ///////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string CharGen(std::string name, int i);
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////  Testing and evaluation   /////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
float AccuracyTest(BoolMatrix* Y, Matrix* Y_hat, Arguments& Arg);
void cluster(Arguments& Arg, DatasetParam& DP, BoolMatrix* Y_hat);
BoolMatrix* MatToBool(Matrix* MAT, float threshold);
void to_JPG(U_IntMatrix* X, string PATH);
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////  Threadded dot product   //////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* DOT(Matrix* X, Matrix* Y);
void DotPart(int part, Matrix* result, Matrix* X, Matrix* Y);
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////    Printing utilities    //////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void PrintLayout(Arguments  &Arg, DatasetParam &DP);

#endif // NN_TOOLS_H_INCLUDED
