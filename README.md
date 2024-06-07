## Requirements
- python>=3.9
- SimpleITK==2.3.1
- numpy==1.26.4
- matplotlib==3.5.1
- torch==1.12.1+cu116
- torchvision==0.13.1+cu116
- torchmetrics==1.3.1
- tqdm==4.66.1
- scikit-learn==1.4.1
- transformers==4.38.2

## Data Pretreatment
In clinical pretreatment, we first generate a text report in Text_Builder.py and then attempt to feature it through Clinicalbert.

In image pretreatment, we implement a size change and window adjustment, make skull stripping and registration and get information on voxel size in CT_Preprocessing.py by the following command:
```
python Utils/CT_Preprocessing.py
```
In the process of data preprocessing, we perform window size setting, skull removal, template registration, and normalization operations on the original image by turn.
Among them, the original image path and image template need to be provided by the user themselves.


## Train
You could train the model through the following command:
```
python train.py
```
In the process of training, we sequentially initialized the model, read in the data, trained and verified it with a 5-fold crossover, validated the evaluation mode, and saved the model.

## Usages
For more detailed usage, you can refer to our code and annotation.

