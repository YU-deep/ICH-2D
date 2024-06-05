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

In image pretreatment, we implement a size change and window adjustment, make skull stripping and registration and get information on voxel size in CT_Preprocessing.py

## Train
You could train the model through the following command:
```
python train.py
```

## Usages
For more detailed usage, you can refer to our code and annotation.

