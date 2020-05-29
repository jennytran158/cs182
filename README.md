I did this project in a group of 3 people.  To read my part in the report click [here](https://github.com/jennytran158/cs182/blob/master/182bert.pdf)!

To see the whole report click [here](https://github.com/jennytran158/cs182/blob/master/CS182_Writeup.pdf) and click [github](https://github.com/vsrin1/CS182-Final-Project) for other model files in the report. 

### Inference using initial model:
1. Download model [here](https://drive.google.com/drive/u/0/folders/11jn8qVLVRMjwmv-REYTimGLYdsn76nAs)
2. Run model:
python test_submission.py <input.jsonl>


### Inference using model with ordinal outputs:
1. Download model [here](https://drive.google.com/drive/u/0/folders/199iZfuYO4Z2dRFFb0jFRtzqxlagwUCdF)
2. Run model:
python test_submission.py <input.jsonl>




### Training:
1. Rename the training file to train.jsonl 
2. Run:
python train_<classifier_name>.py


### Testing:
python test_submission.py <test_file>.jsonl <model_dir_name>
