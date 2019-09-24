#include "NN_Tools.h"
#include "DataSet.h"
#include <conio.h>
using namespace std;

string CharGen(string name, int i)
{
	int temp = i;
	int counter1;   //number of decimal digits in i

	if (temp == 0)
		counter1 = 1;
	else
	{
		for (counter1 = 0; temp != 0; counter1++)
			temp = temp / 10;
	}


	int counter2 = name.size();   //number of chars in name

	string result;
	if (counter2 == 1) { result = "W0"; }
	if (counter2 == 2) { result = "dW0"; }
	if (counter2 == 3) { result = "Sdw0"; }
	if (counter2 == 4) { result = "dACP0"; }
	if (counter2 == 5) { result = "dACP01"; }
	if (counter2 == 6) { result = "dACP012"; }
	if (counter2 == 7) { result = "dACP0123"; }
	if (counter2 == 8) { result = "dACP01234"; }
	if (counter2 == 9) { result = "dACP012345"; }
	if (counter2 == 10) { result = "dACP0123456"; }
	if (counter2 == 11) { result = "dACP01234567"; }
	if (counter2 == 12) { result = "dACP012345678"; }


	for (unsigned int j = 0; j < name.size(); j++) //copy the name into result
		result[j] = name[j];

	int j = counter1 + counter2 - 1;      //copy the number into result
	temp = i;
	do
	{
		result[j] = '0' + (temp % 10);
		temp = temp / 10;
		j--;
	} while (temp != 0);

	return result;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float AccuracyTest(BoolMatrix* Y, Matrix* Y_hat, Arguments& Arg)
{
   float numOferr=0;
   if (Arg.ErrType == SQAURE_ERROR)
    {
        for(int i=0; i<Y_hat->Columns(); i++)                  //making the predictions either 1 or 0
        {
            if(Y->access(0,i)==0)
            {
                if(Y_hat->access(0,i)>-.8)
                   numOferr++;
            }
            if(Y->access(0,i)==1)
            {
                if(Y_hat->access(0,i)<.8)
                   numOferr++;
            }
        }
    }
    else if (Arg.ErrType== CROSS_ENTROPY)
    {
        for(int i=0; i<Y_hat->Columns(); i++)                  //making the predictions either 1 or 0
        {
            if(Y->access(0,i)==0)
            {
                if(Y_hat->access(0,i)>0.3)
                   numOferr++;
            }
            if(Y->access(0,i)==1)
            {
                if(Y_hat->access(0,i)<0.7)
                   numOferr++;
            }
        }
    }
   float errorPercent = numOferr / Y->Columns();
    cout<<"numOferr = "<< numOferr<<endl;
	cout << "Percentage of error = " << errorPercent * 100 << " %" << endl;
    return errorPercent;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cluster(Arguments& Arg, DatasetParam& DP, BoolMatrix* Y_hat)
{

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
BoolMatrix* MatToBool(Matrix* MAT, float threshold)
{
    BoolMatrix* Result = new BoolMatrix(MAT->Rows(), MAT->Columns());
    for (int i = 0; i < MAT->Rows(); i++)
    {
        for (int j = 0; j < MAT->Columns(); j++)
        {
            if (MAT->access(i, j) < threshold)
                Result->access(i, j) = 0;
            else
                Result->access(i, j) = 1;
        }
    }
    return Result;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void to_JPG(U_IntMatrix* X, string PATH)
{

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* DOT(Matrix* X, Matrix* Y)
{
	Matrix* result = new Matrix(X->Rows(), Y->Columns());
	int CORES = thread::hardware_concurrency();
	thread** Threads = new  thread*[CORES];
	Y = Y->TRANSPOSE();
	float** Y_data = Y->ptr();
	float** X_data = X->ptr();
	float** result_data = result->ptr();
	for (int i = 0; i < CORES; i++)
	{
		Threads[i] = new thread(DotPart, i + 1, result, X, Y);
	}

	for (int i = 0; i < CORES; i++)
	{
		Threads[i]->join();
		delete Threads[i];
	}
	delete Threads;

	if (X->Rows() % CORES != 0)
	{
		int numOfRows = X->Rows() % CORES;
		int limit = X->Rows();
		int start = limit - numOfRows;
		for (int i = start; i < limit; i++)
			for (int j = 0; j < Y->Rows(); j++)
				for (int k = 0; k < X->Columns(); k++)
					result_data[i][j] += X_data[i][k] * Y_data[j][k];
	}
	delete Y;
	return result;
}
///////////////////////////////////////////////////////////////////////
void DotPart(int part, Matrix* result, Matrix* X, Matrix* Y)
{
	float** Y_data = Y->ptr();
	float** X_data = X->ptr();
	float** result_data = result->ptr();
	int numOfRows = X->Rows() / thread::hardware_concurrency();
	int limit = part * numOfRows;
	int start = limit - numOfRows;
	for (int i = start; i < limit; i++)
		for (int j = 0; j < Y->Rows(); j++)
			for (int k = 0; k < X->Columns(); k++)
				result_data[i][j] += X_data[i][k] * Y_data[j][k];
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void MIX(Matrix*& X, Matrix*& Y, Matrix* X_, Matrix* Y_)
{
	Matrix* XX = new Matrix(X->Rows(), X->Columns() + X_->Columns());
	Matrix* YY = new Matrix(Y->Rows(), Y->Columns() + Y_->Columns());
	int i = 0;
	int j = 0;
	int k = 0;
	int m = 0;

	for (i = 0; i < X->Rows(); i++)
		for (j = 0; j < X->Columns(); j++)
			XX->access(i, j) = X->access(i, j);


	for (i = 0; i < XX->Rows(); i++)
		for (k = j, m = 0; k < XX->Columns(); k++, m++)
			XX->access(i, k) = X_->access(i, m);




	for (i = 0; i < Y->Rows(); i++)
		for (j = 0; j < Y->Columns(); j++)
			YY->access(i, j) = Y->access(i, j);


	for (i = 0; i < YY->Rows(); i++)
		for (k = j, m = 0; k < YY->Columns(); k++, m++)
			YY->access(i, k) = Y_->access(i, m);


	delete X;
	delete Y;
	delete X_;
	delete Y_;
	X = XX;
	Y = YY;
}
//////////////////////////////////////////////////////////////////////////////////////////////
void PrintLayout(Arguments & Arg, DatasetParam & DP)
{
	cout << ">> DataSet Information: " << endl;
	cout << "CompressedImageSize = " << DP.CompressedImageSize << " neurons" << endl;
	cout << "Total No. Training Files = " << DP.numFiles << endl;
	cout << "NO. Training Images = " << DP.Train_Examples << endl;
	cout << "NO. Compressed Test Images = " << DP.Test_Examples << endl << endl;



	//------------------------------------------------------------------//
	//---------------------- Print Network Layout ----------------------//
	//------------------------------------------------------------------//
	cout << ">> Training Information: " << endl;
	cout << "Type Of Network: ";
	switch (Arg.NetType)
	{
	case FC: cout << "Fully Connected" << endl; break;
	case LENET1: cout << "LENET1" << endl; break;
	case CUSTOM: cout << "Convolution";
	}
	cout << "Optimization Algorithm: ";
	switch (Arg.optimizer)
	{
	case ADAM: cout << "ADAM" << endl; break;
	case GRADIENT_DESCENT: cout << "Gradient Descent" << endl;
	}
	cout << "Cost Function: ";
	switch (Arg.ErrType)
	{
	case CROSS_ENTROPY: cout << "Cross Entropy" << endl; break;
	case SQAURE_ERROR: cout << "Square Error" << endl;
	}
	cout << "Learining Rate = " << Arg.learingRate << endl;
	cout << "Batch Size = " << Arg.batchSize << endl;
	cout << "No. Layers = " << Arg.numOfLayers << endl;
	for (int i = 0; i < Arg.numOfLayers; i++)
	{
		cout << "Layer " << i + 1 << " = " << Arg.layers[i].neurons << " Neurons" << endl;
	}
	cout << endl;

	cout << ">> Parameters Retrieving and Saving:" << endl;
	cout << "Retrieve Parameters (y/n)? ";
	char ans = _getche();
	if(ans = 'y')
        Arg.RetrieveParameters = true;
    else
        Arg.RetrieveParameters = false;
    cout << endl ;

    cout << "Save Parameters (y/n)? ";
    ans = _getche();
	if(ans = 'y')
        Arg.SaveParameters = true;
    else
        Arg.SaveParameters= false;
    cout << endl << endl;
}
