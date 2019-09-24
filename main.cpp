#include "DataSet.h"
#include "Image_Classifier.h"
#define ever ;;
#define Yes 1
#define No 0
int main()
{
	//------------------------------------------------------------------//
	//-------------------------- DataSet -------------------------------//
	//------------------------------------------------------------------//

	DatasetParam DP;
	DP.numFiles = 1;
	DP.curFile = 0;
	DP.Train_Examples = 1500;   // for random 3904 ... for allposs 720
	DP.Test_Examples = 100;
	DP.ImageDim = 100;
	DP.ImagesOfPerson = 8;
	DP.ImagesOfOthers = 24;
	DP.ImagesOfPerson_test = 10;
	//----------------------------------------//
	DP.CompressedImageSize = 200;
	//----------------------------------------//
	DP.CompressedImages_dir = "F:\\Persons_200";
	//-----------------------------------------//
	DP.Get_dataSet = true;
	DP.shuffle = true;              //for train only..no shuffle in test
	DP.AllPossibilities = true;
	DP.normalize_01 = false;        //for both train and test ..
	//-----------------------------------------//
	DP.X_dir = new const char*[DP.numFiles];
	DP.Y_dir = new const char*[DP.numFiles];
	//-----------------------------------------//
	//X_TRAIN_AllPossibilities_200
	//Y_TRAIN_AllPossibilities_200

	//X_TRAIN_AllPossibilities_200_shuffiled
	//Y_TRAIN_AllPossibilities_200_shuffiled

	//X_TRAIN_AllPossibilities_200_shuffiled_Normalized(1,0)
	//Y_TRAIN_AllPossibilities_200_shuffiled_Normalized(1,0)

	//X_TRAIN_AllPossibilities_200_shuffiled_Normalized(-1,1)
	//Y_TRAIN_AllPossibilities_200_shuffiled_Normalized(-1,1)
	/*******************************************************************************************/
	//X_TRAIN_Random_200_shuffiled
	//Y_TRAIN_Random_200_shuffiled

	//X_TRAIN_Random_200_shuffiled_Normalized(1,0)
	//Y_TRAIN_Random_200_shuffiled_Normalized(1,0)

	//X_TRAIN_Random_200_shuffiled_Normalized(-1,1)
	//Y_TRAIN_Random_200_shuffiled_Normalized(-1,1)

	DP.X_dir[0] = "F:\\X & Y train for 200 latent layer-random(diff square)\\X & Y train for 200 latent layer-random(diff square)\\X_TRAIN_Random_200_shuffiled_Normalized(1,0)";
	DP.Y_dir[0] = "F:\\X & Y train for 200 latent layer-random(diff square)\\X & Y train for 200 latent layer-random(diff square)\\Y_TRAIN_Random_200_shuffiled_Normalized(1,0)";
	//-----------------------------------------//
	//DP.Xtest_img_dir = "F:\\4th computer and control\\project\\Final project\\DataSet\\Image Classifier\\Persons_binary_file_10000X3912\\Persons";
	//DP.Xtest_activ_dir = "F:\\4th computer and control\\project\\Final project\\Persons_Activations\\Persons_200";

	DP.Xtest_img_dir = "F:\\UsefullDataSets\\Perosns_Test\\New folder\\OrderedTestPersons_100";
	//DP.Xtest_activ_dir = "F:\\4th computer and control\\project\\Final project\\Datasets_N\\Binary Files\\Newclass\\OrderedTestPersonsActivations_100";
	DP.Xtest_activ_dir = "F:\\UsefullDataSets\\Perosns_Test\\New folder\\OrderedTestPersonsActivations_100";

	//DP.Xtest_img_dir = "F:\\4th computer and control\\project\\Final project\\Datasets_N\\Binary Files\\Newclass\\Persons_Test_390";
	//DP.Xtest_activ_dir = "F:\\4th computer and control\\project\\Final project\\Datasets_N\\Binary Files\\Newclass\\Persons_TestActivations500_390";
	//DP.ClusteredImagesPath = "F:\\4th computer and control\\project\\Final project\\Datasets_N\\Images Grey\\Clustered Images\\";
	//-----------------------------------------//
	DP.ParametersPath = "F:\\CurImgClassFierParas\\";
	//------------------------------------------------------------------------//


	//------------------------------------------------------------------//
	//------------------- Network Architecture -------------------------//
	//------------------------------------------------------------------//


	int numOfLayers = 5;
	layer*  layers = new layer[numOfLayers];
	layers[0].put(DP.CompressedImageSize, NONE);
	layers[1].put(100, LEAKYRELU);
	layers[2].put(50, LEAKYRELU);
	layers[3].put(10, LEAKYRELU);
	layers[4].put(1, SIGMOID);


	float*  keep_prob = new float[numOfLayers];
	keep_prob[0] = 1;
	keep_prob[1] = 0.6;
	keep_prob[2] = 1;
	//keep_prob[3] = 1;
	//keep_prob[4] = 1;

	Arguments Arg;
	Arg.NetType = FC;
	Arg.optimizer = ADAM;
	Arg.ErrType = CROSS_ENTROPY;
	Arg.layers = layers;
	Arg.numOfLayers = numOfLayers;
	Arg.keep_prob = keep_prob;
	//---------------------------//
	Arg.numPrint = 1;
	Arg.numOfEpochs = 1;
	Arg.batchSize = 1024;
	Arg.Test_Batch_Size = 3904*(8+24);  //To test on trainset :(720*719/2) 258840 for all paterns in allposs , 3904*(8+24) for all paterns in random 
	Arg.threshold = 0.5;
	//---------------------------//
	Arg.learingRate = 0.005;
	Arg.decayRate = 1;
	Arg.regularizationParameter = 0;
	//---------------------------//
	Arg.batchNorm = true;
	Arg.dropout = false;
	Arg.dropConnect = false;
	Arg.negative = false;
	//----------------------------//
	Arg.SaveParameters = true;
	Arg.RetrieveParameters = false;
	Arg.TestParameters = false;
	//----------------------------//

	//------------------------------------------------------------------//
	//-------------------------- Training ------------------------------//
	//------------------------------------------------------------------//

	Prepare_TestSet1(Arg, DP);    //if square error noramalize (-1,1) for both train and test ..
	if (DP.Get_dataSet)
		Prepare_TrainSet1(Arg, DP);
	else
	{
		int numOfPatterns;
		if (!DP.AllPossibilities)
			numOfPatterns = DP.Train_Examples*(DP.ImagesOfPerson + DP.ImagesOfOthers);
		else
			numOfPatterns = DP.Train_Examples*(DP.Train_Examples - 1) / 2;

		//Arg.X = new Matrix(DP.CompressedImageSize * 2, numOfPatterns);
		Arg.X = new Matrix(DP.CompressedImageSize , numOfPatterns);
		Arg.Y = new BoolMatrix(1, numOfPatterns);
		if(Arg.ErrType == SQAURE_ERROR)
			Arg.negative = true;
	}


	Image_Classifier IC(&Arg, &DP);
	PrintLayout(Arg, DP);
	IC.RetrieveParameters();
	IC.TestParameters();
	int i = 0;
	for (ever)
	{
		clock_t start = clock();
		cout << endl << ">> Epoch no. " << ++i << ":" << endl;
		for (DP.curFile = 0; DP.curFile < DP.numFiles; DP.curFile++)
		{
			if (i == 1 || DP.numFiles > 1)
			{
				Arg.X->Read(DP.X_dir[DP.curFile]);
				Arg.Y->Read(DP.Y_dir[DP.curFile]);
				Arg.X_dev = Arg.X;
				Arg.Y_dev = Arg.Y;
			}
			IC.train();
			//IC.test(DEV);
			IC.test(TEST);

			Arg.learingRate = Arg.learingRate * Arg.decayRate;
			
		}

		clock_t end = clock();
		double duration_sec = double(end - start) / CLOCKS_PER_SEC;
		cout << "Time = " << duration_sec << endl;
		//_getche();
	}
	_getche();
	return 0;
}
