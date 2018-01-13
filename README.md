# Skin Cancer Classification
ISIC 2017: Skin Lesion Analysis Towards Melanoma Detection (Lesion Classification)

This study proposes the use of deep learning algorithms to detect the presence of skin cancer, specifically melanoma, from images of skin lesions taken by a standard camera. Skin cancer is the most prevalent form of cancer in the US where 3.3 million people get treated each year. The 5-year survival rate of melanoma is 98% when detected and treated early yet over 10,000 people are lost each year due mostly to late-stage diagnoses. Thus, there is a need to make melanoma screening and diagnoses methods cheaper, quicker, simpler, and more accessible. This study aims to produce an inexpensive and fast computer-vision based machine learning tool that can be used by doctors and patients to track and classify suspicious skin lesions as benign or malignant with adequate accuracy using only a cell phone camera.

Required install:

```
h5py==2.7.1
Keras==2.1.1
matplotlib==2.1.0
numpy==1.13.3
opencv-python==3.3.0.10
scikit-learn==0.19.1
tensorflow==1.3.0
tensorflow-tensorboard==0.1.8
urllib3==1.22
virtualenv==15.1.0
```

# Dataset
https://challenge.kitware.com/#challenge/n/ISIC_2017%3A_Skin_Lesion_Analysis_Towards_Melanoma_Detection

To create dataset :
```
python main.py create
```

To train the classifier
```
python main.py train
```

To test the classifier
```
python main.py test
```

# People
- Veda Deepta Pal - https://github.com/vedadeepta
- Tanmoy Bhowmik - https://github.com/tanmoyopenroot
- Arkaprabha Das - https://github.com/arkaprabhadas
