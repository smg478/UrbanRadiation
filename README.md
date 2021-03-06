[//]: # (Image References)
[image1]: ./res/img1.png
[image2]: ./res/img2.png
[image3]: ./res/img3.png
[image4]: ./res/img4.png



# Detecting Radiological Threats in Urban Areas
The goal of this challenge was to find the presence of radioactive sources in urban setting using gamma ray detectors. Particularly, presence of 6 different types of radioactive sources need to be identified while travelling with the detector-NaI(Tl) at various speeds. Detailes of this challenge can be found in https://www.topcoder.com/challenges/30085346. Code in this repo complies with the requirements from here: https://docs.google.com/document/d/17QuWdnKX0KZpP-7h0a186Sx6HpsQssT-QnsnQTYTIkk/edit.

![alt text][image1]
![alt text][image2]



## Code docomentation

Data is assumed to be in 'data/' folder. Another empty folder 'wdata/' is needed for temporary file writing.

```
data/
    trainingAnswers.csv
    training/
      100001.csv
      100002.csv
      ...
    testing/
      200001.csv
      200002.csv
      ...
   
wdata/
```      
The folder consists of 7 scripts that do data preparation, training and prediction. Scripts are written in python language.

Training and testing both can be done on a CPU based machine. However, training in a GPU is much faster. Testing using GPU doesn’t concern much.


## Script description

### Data preparation
```
01_make_slice_data.py
02_make_features.py
```
- Script 01 makes approximately 81,000 segmented data from 9,700 training data available and saves newly generated data on “wdata/training_slice” folder and corresponding answer file as 'wdata/trainingAnswers_slice.csv'.
- Script 02  takes the files generated by script 01, creates 151 features from each file and finally save everything as 'wdata/train_feature_bin_30_slice.csv'

### Training
```
03_train_ANN_CNN.py
```

![alt text][image4]

- This script trains a hybrid model of convolutional neural network (CNN) and multi-layer perceptron (MLP) neural network using training features generated in script 02.
- Model weights will be saved in ‘weights/’ folder as well as in ‘wdata/weights’ folder

### Inference
```
06_predict_25.py
07_predict_3000.py
08_predict_6000.py
```

- 3 prediction files are identical except they predict on different segment (window) sizes. The prediction was carried out on 200 anchor points. These scripts use weights produced from script 03.
-- Script 06: window size = total counts in test file / 12
-- Script 07: window size = 3000 counts
-- Script 08: window size = 6000 counts
- 3 different thresholds (e.g. 3, 5 and 7 out of 200) were used to judge a test file as source positive or negative.
- The reason I have used 3 scripts for prediction instead of 1 because I found it little complicated to run parallel jobs with tensorflow models. So I have saved time by using 3 scripts running in parallel.
Output files will be saved under ‘wdata/submits’ folder

### Ensemble predictions
```
09_vote_ensemble.py
```
- This script ensemble the prediction of the source type and location by voting style from the 9 output files produced from inference (script 06, 07, 08).
- The output file will be saved under ‘wdata/submits’ folder

### Finetune source location
```
10_timeProcess.py
```

- Outputs the final predictions and saves it to the current directory.

![alt text][image3]

## How to run the code

The code is expected to run in Docker container. Docker is assumed to be installed in the host computer. This code doesn’t  need a GPU for training or inference. Dockerfile file is sufficient for a cpu only machine. It installs necessary python dependancies on a Ubuntu 16.04 OS.

- start docker
```
sudo service docker start
```
- build solution from the folder that contains Dockerfile
```
docker build -t smg478 .
```
- Strat container
```
docker run -v <local_data_path>:/data:ro -v <local_writable_area_path>:/wdata -it <id>
```
- Inference using pre-built model
```
bash test.sh /data/testing/ solution.csv
```
- Train
```
bash train.sh /data/training/ /data/trainingAnswers.csv
```
- Inference on newly trained model - produces solution file on current directory
```
bash test.sh /data/testing/ solution.csv
```

### Expected running time:
- Local PC config: Ubuntu 14.04, Intel i7 (8-core), 32 GB RAM, SSD
- Disc space required: 7 GB for processed data file + 5 MB for model weights
- Training : bash train.sh /data/training/ /data/trainingAnswers.csv
 (2.0 / 1.0) hr in a (CPU / GPU) based machine
- Testing: bash test.sh /data/testing/ solution.csv
6 hr in a CPU based machine (+ 30 min, if processed data file needs to be generated again. Usually this file will be generated during the training phase (200MB))

For detailes, please refer to summery-documentation file.
