#include "DataSet.h"
//////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////  GET DATASET FROM HARD DISK   ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void Prepare_TrainSet(Arguments& Arg, DatasetParam& DP)
{
	//Retrieving stored CompressedImages from hard disk
	cout << "Preparing DataSet ..." << endl;
	Matrix* CompressedImages = new Matrix(DP.CompressedImageSize, DP.Train_Examples, Random);
	CompressedImages->Read(DP.CompressedImages_dir);
	cout << "CompressImages dim = " << CompressedImages->Rows() << " X " << CompressedImages->Columns() << endl;

	Matrix* X = nullptr;
	BoolMatrix* Y = nullptr;
	if (!DP.AllPossibilities)
	{
		//Preparing X & Y
		int numOfPatterns = DP.Train_Examples*(DP.ImagesOfPerson + DP.ImagesOfOthers);
		X = new Matrix(DP.CompressedImageSize * 2, numOfPatterns);
		Y = new BoolMatrix(1, numOfPatterns);
		cout << "X dim = " << X->Rows() << " X " << X->Columns() << endl;
		//Getting Y (1 for each images is repeated ImagesOfPerson number of times & 0 is repeated ImagesOfOthers number of times)
		//Getting Upper half of X (where each pattern is repeated ImagesOfPerson+ImagesOfOthers number of times)
		for (int i = 0; i<DP.CompressedImageSize; i++)
		{
			for (int j = 0; j<X->Columns(); j++)
			{
				int index = j / (DP.ImagesOfPerson + DP.ImagesOfOthers);
				X->access(i, j) = CompressedImages->access(i, index);
				if (i == 0)
				{
					if ((j % (DP.ImagesOfPerson + DP.ImagesOfOthers))<DP.ImagesOfPerson)
						Y->access(0, j) = 1;
					else
						Y->access(0, j) = 0;
				}
			}
		}

		//Getting Lower half of X
		int groupIndex = 0; int c1 = 0; int c2 = 0; int index = 0;
		for (int i = 0; i<X->Columns(); i++)
		{
			if (DP.ImagesOfPerson != 0 && i != 0 && i % ((DP.ImagesOfPerson + DP.ImagesOfOthers) * DP.ImagesOfPerson) == 0)
			{
				groupIndex++;
			}
			if (c1 < DP.ImagesOfPerson)
			{
				index = DP.ImagesOfPerson*groupIndex + i % (DP.ImagesOfPerson + DP.ImagesOfOthers);
				if (index >= DP.Train_Examples)
					index -= DP.Train_Examples;
				c1++;
				if (c1 >= DP.ImagesOfPerson && DP.ImagesOfOthers == 0)
				{
					c1 = 0;
				}
			}
			else
			{
				if (c2 < DP.ImagesOfOthers)
				{
					index = rand() % (DP.Train_Examples - DP.ImagesOfPerson) + DP.ImagesOfPerson*(groupIndex + 1);
					if (index >= DP.Train_Examples)
						index -= DP.Train_Examples;
				}
				c2++;
				if (c2 >= DP.ImagesOfOthers)
				{
					c2 = 0; c1 = 0;
				}
			}
			for (int j = 0; j<DP.CompressedImageSize; j++)
			{
				X->access(j + DP.CompressedImageSize, i) = CompressedImages->access(j, index);
			}
		}
	}
	else
	{
		int numOfPatterns = DP.Train_Examples*(DP.Train_Examples - 1) / 2;
		X = new Matrix(DP.CompressedImageSize * 2, numOfPatterns);
		Y = new BoolMatrix(1, numOfPatterns);
		cout << "X dim = " << X->Rows() << " X " << X->Columns() << endl;
		int X_Column_index = 0;
		for (int FirstImage = 0; FirstImage < CompressedImages->Columns() - 1; FirstImage++)
		{
			int numOfOnes = DP.ImagesOfPerson - FirstImage % DP.ImagesOfPerson - 1;
			int counter = 0;
			for (int SecondImage = FirstImage + 1; SecondImage < CompressedImages->Columns(); SecondImage++)
			{
				if (counter<numOfOnes)
				{
					Y->access(0, X_Column_index) = 1;
					counter++;
				}
				else
				{
					Y->access(0, X_Column_index) = 0;
				}

				for (int i = 0; i < X->Rows() / 2; i++)
				{
					X->access(i, X_Column_index) = CompressedImages->access(i, FirstImage);
				}
				int j = 0;
				for (int i = X->Rows() / 2; i < X->Rows(); i++)
				{
					X->access(i, X_Column_index) = CompressedImages->access(j, SecondImage);
					j++;
				}
				X_Column_index++;
			}
		}
	}

	if (Arg.ErrType == SQAURE_ERROR)
	{
		Arg.X = Normalize(X);
		Arg.negative = true;
	}
	else if (DP.normalize_01)
		Arg.X = Normalize_01(X);
	else
		Arg.X = X;

	//Shuffling X and Y
	if (DP.shuffle)
	{
		for (int i = 0; i< Arg.X->Columns(); i++)
		{
			int s = rand() % Arg.X->Columns();
			SWAP(Arg.X, i, s);
			SWAP(Arg.Y, i, s);
		}
	}
	//Writing X & Y in files
	int fileSize = Arg.X->Columns() / DP.numFiles;
	for (int i = 0; i < DP.numFiles; i++)
	{
		Matrix* temp1 = Arg.X->SubMat(0, i*fileSize, -1, (i + 1)*fileSize - 1);
		BoolMatrix* temp2 = Arg.Y->SubMat(0, i*fileSize, -1, (i + 1)*fileSize - 1);
		temp1->Write(DP.X_dir[i]);
		temp2->Write(DP.Y_dir[i]);
		delete temp1;
		delete temp2;
	}
	delete CompressedImages;

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Prepare_TestSet(Arguments& Arg, DatasetParam& DP)
{
	//Retrieving stored CompressedImages from hard disk
	cout << "Preparing Testset ..." << endl;
	Matrix* CompressedImages = new Matrix(DP.CompressedImageSize, DP.Test_Examples);
	CompressedImages->Read(DP.Xtest_activ_dir);
	cout << "Compressed Test Images dim = " << CompressedImages->Rows() << " X " << CompressedImages->Columns() << endl;
	U_IntMatrix* temp = new U_IntMatrix(DP.Test_Examples, DP.ImageDim * DP.ImageDim);
	temp->Read(DP.Xtest_img_dir);
	Arg.X_test_img = temp->TRANSPOSE();
	delete temp;

	int numOfPatterns = DP.Test_Examples*(DP.Test_Examples - 1) / 2;
	Matrix* X_test = new Matrix(DP.CompressedImageSize * 2, numOfPatterns);
	BoolMatrix* Y_test = new BoolMatrix(1, numOfPatterns);
	cout << "X dim = " << X_test->Rows() << " X " << X_test->Columns() << endl;

	int X_test_Column_index = 0;
	for (int FirstImage = 0; FirstImage < CompressedImages->Columns() - 1; FirstImage++)
	{
		int numOfOnes = 10 - FirstImage % 10 - 1;
		int counter = 0;
		for (int SecondImage = FirstImage + 1; SecondImage < CompressedImages->Columns(); SecondImage++)
		{
			if (counter<numOfOnes)
			{
				Y_test->access(0, X_test_Column_index) = 1;
				counter++;
			}
			else
			{
				Y_test->access(0, X_test_Column_index) = 0;
			}
			for (int i = 0; i < X_test->Rows() / 2; i++)
			{
				X_test->access(i, X_test_Column_index) = CompressedImages->access(i, FirstImage);
			}
			int j = 0;
			for (int i = X_test->Rows() / 2; i < X_test->Rows(); i++)
			{
				X_test->access(i, X_test_Column_index) = CompressedImages->access(j, SecondImage);
				j++;
			}
			X_test_Column_index++;
		}
	}

	Arg.X_test = X_test;
	Arg.Y_test = Y_test;

	delete CompressedImages;
	if (DP.shuffle)
	{
		for (int i = 0; i< Arg.X_test->Columns(); i++)
		{
			int s = rand() % Arg.X_test->Columns();
			SWAP(Arg.X_test, i, s);
			SWAP(Arg.Y_test, i, s);
		}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Prepare_TrainSet1(Arguments& Arg, DatasetParam& DP)
{
	//Retrieving stored CompressedImages from hard disk
	cout << "Preparing DataSet ..." << endl;
	Matrix* CompressedImages = new Matrix(DP.CompressedImageSize, DP.Train_Examples, 0);
	CompressedImages->Read(DP.CompressedImages_dir);
	cout << "CompressImages dim = " << CompressedImages->Rows() << " X " << CompressedImages->Columns() << endl;
	Matrix* X = nullptr;
	BoolMatrix* Y = nullptr;
	if (!DP.AllPossibilities)
	{
		//Preparing X & Y
		int numOfPatterns = DP.Train_Examples*(DP.ImagesOfPerson + DP.ImagesOfOthers);
		X = new Matrix(DP.CompressedImageSize, numOfPatterns);
		Y = new BoolMatrix(1, numOfPatterns);
		cout << "X dim = " << X->Rows() << " X " << X->Columns() << endl;
		//Getting Y (1 for each images is repeated ImagesOfPerson number of times & 0 is repeated ImagesOfOthers number of times)
		for (int j = 0; j < X->Columns(); j++)
		{
			if ((j % (DP.ImagesOfPerson + DP.ImagesOfOthers)) < DP.ImagesOfPerson)
				Y->access(0, j) = 1;
			else
				Y->access(0, j) = 0;
		}

		//Getting X
		int groupIndex = 0; int c1 = 0; int c2 = 0; int index = 0; int image_index = 0;
		for (int i = 0; i<X->Columns(); i++)
		{
			if (DP.ImagesOfPerson != 0 && i != 0 && i % ((DP.ImagesOfPerson + DP.ImagesOfOthers) * DP.ImagesOfPerson) == 0)
			{
				groupIndex++;
			}
			if (c1 < DP.ImagesOfPerson)
			{
				if (c1 == 0 && i != 0)
				{
					image_index++;
					cout << image_index << "    " << i << endl;
				}
				index = DP.ImagesOfPerson*groupIndex + i % (DP.ImagesOfPerson + DP.ImagesOfOthers);
				if (index >= DP.Train_Examples)
					index -= DP.Train_Examples;
				c1++;
				if (c1 >= DP.ImagesOfPerson && DP.ImagesOfOthers == 0)
				{
					c1 = 0;
				}
			}
			else
			{
				if (c2 < DP.ImagesOfOthers)
				{
					index = rand() % (DP.Train_Examples - DP.ImagesOfPerson) + DP.ImagesOfPerson*(groupIndex + 1);
					if (index >= DP.Train_Examples)
						index -= DP.Train_Examples;
				}
				c2++;
				if (c2 >= DP.ImagesOfOthers)
				{
					c2 = 0; c1 = 0;
				}
			}
			for (int j = 0; j<DP.CompressedImageSize; j++)
			{
				float temp = CompressedImages->access(j, image_index) - CompressedImages->access(j, index);
				X->access(j, i) = temp * temp;
			}
		}
	}
	else
	{
		int numOfPatterns = DP.Train_Examples*(DP.Train_Examples - 1) / 2;
		X = new Matrix(DP.CompressedImageSize, numOfPatterns);
		Y = new BoolMatrix(1, numOfPatterns);
		cout << "X dim = " << X->Rows() << " X " << X->Columns() << endl;
		int X_Column_index = 0;
		for (int FirstImage = 0; FirstImage < CompressedImages->Columns() - 1; FirstImage++)
		{
			int numOfOnes = DP.ImagesOfPerson - FirstImage % DP.ImagesOfPerson - 1;
			int counter = 0;
			for (int SecondImage = FirstImage + 1; SecondImage < CompressedImages->Columns(); SecondImage++)
			{
				if (counter < numOfOnes)
				{
					Y->access(0, X_Column_index) = 1;
					counter++;
				}
				else
				{
					Y->access(0, X_Column_index) = 0;
				}

				for (int i = 0; i < X->Rows(); i++)
				{
					float temp = CompressedImages->access(i, FirstImage) - CompressedImages->access(i, SecondImage);
					X->access(i, X_Column_index) = temp * temp;
				}
				X_Column_index++;
			}
		}
	}
	if (Arg.ErrType == SQAURE_ERROR)
	{
		Arg.X = Normalize(X);
		Arg.negative = true;
	}
	else if (DP.normalize_01)
		Arg.X = Normalize_01(X);
	else
		Arg.X = X;

	Arg.Y = Y;


	//Shuffling X and Y
	if (DP.shuffle)
	{
		for (int i = 0; i< Arg.X->Columns(); i++)
		{
			int s = rand() % Arg.X->Columns();
			SWAP(Arg.X, i, s);
			SWAP(Arg.Y, i, s);
		}
	}


	//Writing X & Y in files
	int fileSize = Arg.X->Columns() / DP.numFiles;
	for (int i = 0; i < DP.numFiles; i++)
	{
		Matrix* temp1 = Arg.X->SubMat(0, i*fileSize, -1, (i + 1)*fileSize - 1);
		BoolMatrix* temp2 = Arg.Y->SubMat(0, i*fileSize, -1, (i + 1)*fileSize - 1);
		temp1->Write(DP.X_dir[i]);
		temp2->Write(DP.Y_dir[i]);
		delete temp1;
		delete temp2;
	}
	delete CompressedImages;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Prepare_TestSet1(Arguments& Arg, DatasetParam& DP)
{
	//Retrieving stored CompressedImages from hard disk
	cout << "Preparing Testset ..." << endl;
	Matrix* CompressedImages = new Matrix(DP.CompressedImageSize, DP.Test_Examples);
	CompressedImages->Read(DP.Xtest_activ_dir);
	cout << "Compressed Test Images dim = " << CompressedImages->Rows() << " X " << CompressedImages->Columns() << endl;
	U_IntMatrix* temp = new U_IntMatrix(DP.Test_Examples, DP.ImageDim * DP.ImageDim);
	temp->Read(DP.Xtest_img_dir);
	Arg.X_test_img = temp;
	delete temp;

	int numOfPatterns = DP.Test_Examples*(DP.Test_Examples - 1) / 2;
	Matrix* X_test = new Matrix(DP.CompressedImageSize, numOfPatterns);
	BoolMatrix* Y_test = new BoolMatrix(1, numOfPatterns);
	cout << "X_test dim = " << X_test->Rows() << " x " << X_test->Columns() << endl;

	int X_test_Column_index = 0;
	for (int FirstImage = 0; FirstImage < CompressedImages->Columns() - 1; FirstImage++)
	{
		int numOfOnes = DP.ImagesOfPerson_test - FirstImage % DP.ImagesOfPerson_test - 1;
		int counter = 0;
		for (int SecondImage = FirstImage + 1; SecondImage < CompressedImages->Columns(); SecondImage++)
		{
			if (counter<numOfOnes)
			{
				Y_test->access(0, X_test_Column_index) = 1;
				counter++;
			}
			else
			{
				Y_test->access(0, X_test_Column_index) = 0;
			}
			for (int i = 0; i < X_test->Rows(); i++)
			{
				float temp = CompressedImages->access(i, FirstImage) - CompressedImages->access(i, SecondImage);
				X_test->access(i, X_test_Column_index) = temp * temp;
			}
			X_test_Column_index++;
		}
	}

	if (Arg.ErrType == SQAURE_ERROR)
	{
		Arg.X_test = Normalize(X_test);
		Arg.negative = true;
	}
	else if (DP.normalize_01)
		Arg.X_test = Normalize_01(X_test);
	else
		Arg.X_test = X_test;

	Arg.Y_test = Y_test;


	delete CompressedImages;
	if (DP.shuffle)
	{
		for (int i = 0; i< Arg.X_test->Columns(); i++)
		{
			int s = rand() % Arg.X_test->Columns();
			SWAP(Arg.X_test, i, s);
			SWAP(Arg.Y_test, i, s);
		}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SWAP(Matrix* MAT, int i, int k)
{
	Matrix* temp = new Matrix(MAT->Rows(), 1);
	for (int j = 0; j < MAT->Rows(); j++)
	{
		temp->access(j, 0) = MAT->access(j, i);
		MAT->access(j, i) = MAT->access(j, k);
		MAT->access(j, k) = temp->access(j, 0);
	}
	delete temp;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SWAP(BoolMatrix* MAT, int i, int k)
{
	Matrix* temp = new Matrix(MAT->Rows(), 1);
	for (int j = 0; j < MAT->Rows(); j++)
	{
		temp->access(j, 0) = MAT->access(j, i);
		MAT->access(j, i) = MAT->access(j, k);
		MAT->access(j, k) = temp->access(j, 0);
	}
	delete temp;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* Normalize(Matrix* X)
{
	Matrix* X_SumCol = X->SUM("column");
	Matrix* X_Mean = X_SumCol->div(X->Columns());
	Matrix* X_Mue = X->sub(X_Mean);
	Matrix* X_Mue_SQUARE = X_Mue->SQUARE();
	Matrix* X_Mue_SQUARE_SumCol = X_Mue_SQUARE->SUM("column");
	Matrix* X_Var = X_Mue_SQUARE_SumCol->div(X->Columns());
	delete X_SumCol;
	delete X_Mean;
	delete X_Mue_SQUARE;
	delete X_Mue_SQUARE_SumCol;



	Matrix* X_Var_Eps = X_Var->add(1e-7);
	Matrix* X_Var_Eps_SQUARE = X_Var_Eps->SQRT();
	Matrix* X_Telda = X_Mue->div(X_Var_Eps_SQUARE);
	delete X_Mue;
	delete X_Var;
	delete X_Var_Eps;
	delete X_Var_Eps_SQUARE;

	Matrix* X_Telda_Min = X_Telda->sub(X_Telda->MinElement());
	Matrix* X_Normalized = X_Telda_Min->div(X_Telda->MaxElement() - X_Telda->MinElement());
	delete X_Telda;
	delete X_Telda_Min;

	Matrix* X1 = X_Normalized->mul(2);
	delete X_Normalized;
	Matrix* X_Norm = X1->add(-1);
	delete X1;
	return X_Norm;
}
//////////////////////////////////////////////////////////////////////////////////////////////
Matrix* Normalize_01(Matrix* X)
{
	Matrix* X_SumCol = X->SUM("column");
	Matrix* X_Mean = X_SumCol->div(X->Columns());
	Matrix* X_Mue = X->sub(X_Mean);
	Matrix* X_Mue_SQUARE = X_Mue->SQUARE();
	Matrix* X_Mue_SQUARE_SumCol = X_Mue_SQUARE->SUM("column");
	Matrix* X_Var = X_Mue_SQUARE_SumCol->div(X->Columns());
	delete X_SumCol;
	delete X_Mean;
	delete X_Mue_SQUARE;
	delete X_Mue_SQUARE_SumCol;



	Matrix* X_Var_Eps = X_Var->add(1e-7);
	Matrix* X_Var_Eps_SQUARE = X_Var_Eps->SQRT();
	Matrix* X_Telda = X_Mue->div(X_Var_Eps_SQUARE);
	delete X_Mue;
	delete X_Var;
	delete X_Var_Eps;
	delete X_Var_Eps_SQUARE;

	Matrix* X_Telda_Min = X_Telda->sub(X_Telda->MinElement());
	Matrix* X_Normalized = X_Telda_Min->div(X_Telda->MaxElement() - X_Telda->MinElement());
	delete X_Telda;
	delete X_Telda_Min;

	return X_Normalized;
}
