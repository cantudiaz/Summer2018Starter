#!/usr/bin/env wolframscript
(* ::Package:: *)

(*
The label data is read from the IAM database webpage. After reading the text file, some string processing
is done, since some of the labels have some errors and inconsistencies. 
*)
data = Import["http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/words.txt"] // StringDrop[#,802]&;
dataTidy = StringReplace[data, "n log n"-> "nlogn"] // StringReplace[#, "B B C"-> "BBC"]& // StringReplace[#,"M P"-> "MP"]& // StringReplace[#, "A T V"-> "ATV"]&//StringReplace[#, "I T V"-> "ITV"]&//StringReplace[#,x:"NN "~~ "T V" :> x<>"TV"]& // StringReplace[#,"T C O" :> "T CO"]&;
dataList = (StringSplit[dataTidy]//Partition[#,9]& );
colNames = StringSplit@"word_id segmentation_result graylevel x_bounding y_bounding w_bounding h_bounding gramm_tag transcription"; 
dataSet = Association@@@Table[colNames[[i]]-> (dataList)[[j,i]], {j,1,Length[dataList]},{i,1,Length[colNames]}] // Dataset;

fileNames = FileNames["*.png",{"~/words"}, Infinity];

(* Some files are corrupt, so they are deleted. *)
newCol = Association["image"-> File@#]&/@fileNames;
dataSet = Join[dataSet, newCol,2 ] // Dataset;
dataSet = Delete[dataSet, {{4153},{113622}}];
fileNames =Delete[fileNames,{{4153},{113622}}];

(* This rectangle will be a white background for the images. *)
rectangle = Import["~/rectangle.png"];

(* Function used to fit an image to either width 128 or height 32, keeping aspect ratio. *)
reshape = Function[{image0, widthTarget0, heigthTarget0},
	Module[{image= image0,widthTarget = widthTarget0, heigthTarget=heigthTarget0, width, heigth ,fx,fy,f,newSize, resizedImage},
	{width,heigth} = ImageDimensions[image];
	fx=width/widthTarget;
	fy=heigth/heigthTarget;
	f=Max[fx,fy];
	newSize={Max[Min[widthTarget, Round[width/f]],1], Max[Min[heigthTarget,Round[heigth/f]],1]} ;
	resizedImage=ImageResize[image, newSize];
	resizedImage
	]
];

(* Function used to superimpose the image over the rectangle. *)
imageInRectangle = Function[{image0,rectangle0, widthTarget0, heigthTarget0},
	Module[{image=image0, rectangle=rectangle0,widthTarget = widthTarget0, heigthTarget=heigthTarget0, composedImage},
	composedImage=ImageCompose[rectangle, reshape[image,widthTarget, heigthTarget], {Center}];
	composedImage
	]
];

(*The next line has to be modified if running the script in Windows.*)
newDirectories5 = StringReplace[#,"/"~~LetterCharacter~~DigitCharacter..~~"-"~~DigitCharacter..~~LetterCharacter|""~~"-"~~DigitCharacter..~~"-"~~DigitCharacter..~~ ".png"-> ""]&/@fileNames // StringReplace[#,"/words"-> "/words_resized5"]&//DeleteDuplicates;

(*Uncomment the line below if you want to create the directories in your system.*)
(*CreateDirectory[File@#]&/@newDirectories5;*)

exportFileNames5 =StringReplace[#,"/words"-> "/words_resized5"]&/@fileNames;

(*
Uncomment and modify path if you want to export the files into a local directory.

ParallelMap[Export[exportFileNames5[[#]],ImageCompose[Import["~/rectangle.png"], reshape[Import@fileNames[[#]],128, 32], {Left,Center}, {Left, Center}]]&,Range[Length[fileNames]]]
*)

(* Further data preprocessing *)
imageAndLabels= dataSet[[1;;, {"image", "transcription"}]];
modifiedLabels=imageAndLabels[[1;;,2]]  // Normal;
modifiedImages=File[#]&/@exportFileNames5;
modifiedData = Table[modifiedImages[[n]]->  modifiedLabels[[n]], {n,1,Length[modifiedLabels],1}];
modifiedData = Select[modifiedData // Association, StringFreeQ[_?UpperCaseQ]] // Select[#, StringFreeQ[PunctuationCharacter]] & // Select[#, StringFreeQ[DigitCharacter]]& //Normal;

(* String with all characters (26 in total) *)
chars = Keys@KeySort@Counts@Flatten@Characters@modifiedData [[1;;,2]];

{testData, trainData} =  TakeDrop[modifiedData ,Ceiling[Length[dataSet]/10]];

encoder = NetEncoder[{"Image",{128,32}, "ColorSpace"-> "Grayscale"}];
decoder=NetDecoder[{"CTCBeamSearch",chars,"BeamSize"->50}];
ctcLoss = CTCLossLayer["Target"-> NetEncoder[{"Characters",chars}]];

(* One of the models *)
laNet2 = NetInitialize@NetChain[{
	ConvolutionLayer[64,{3,3}],
	Ramp,
	ConvolutionLayer[64,3],
	Ramp,
	PoolingLayer[2],
	ConvolutionLayer[128,3],
	Ramp,
	ConvolutionLayer[128,3],
	Ramp,
	PoolingLayer[2],
	ConvolutionLayer[256,3],
	Ramp,
	ConvolutionLayer[256,3],
	Ramp,
	ConvolutionLayer[512,3],
	Ramp,
	PoolingLayer[2],
	NetMapOperator@LinearLayer[Length[chars]+1],
	Ramp,
	DropoutLayer[0.5],
	NetMapOperator@LinearLayer[Length[chars]+1],
	DropoutLayer[0.5],
	SoftmaxLayer[]
	},
"Input"->encoder,
"Output"->decoder]

(* Training the model *)
laNet2Train = NetTrain[laNet2, trainData,LossFunction->ctcLoss,MaxTrainingRounds->50,ValidationSet->testData, TargetDevice->"GPU", TrainingProgressReporting-> "Print", TrainingProgressCheckpointing->{"Directory", "~/progress1", "Interval"-> Quantity[2,"Rounds"]}]

Export["~/net1.wlnet",laNet2Train]