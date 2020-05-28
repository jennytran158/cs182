I did this project in a group of 3 people.  To read my part in the report click [here](https://github.com/jennytran158/cs182/blob/master/182bert.pdf)!

To see the whole report click [here](https://github.com/jennytran158/cs182/blob/master/CS182_Writeup.pdf) and click [github](https://github.com/vsrin1/CS182-Final-Project) for other model files in the report. 

Inference using initial model:
1. Download model at:
https://drive.google.com/drive/u/0/folders/11jn8qVLVRMjwmv-REYTimGLYdsn76nAs
2. Run model:
python test_submission.py <input.jsonl>


Inference using model with ordinal outputs:
1. Download model at:
https://drive.google.com/drive/u/0/folders/199iZfuYO4Z2dRFFb0jFRtzqxlagwUCdF
2. Run model:
python test_submission.py <input.jsonl>




Training:

For all models but Binary Classifier Model:
1. Modify 
train_df = pd.read_json('train1.json')
to
train_df = pd.read_json('<file_name>.json')
2. run:
python train_<model_name>.py


For Binary Classifier Model:
1. Modify 
train_df1 = pd.read_json('train1.json')
to
train_df1 = pd.read_json('<file_name>.json')
2. Modify
train_df2 = pd.read_json('train2.json')
to 
train_df2 = pd.read_json('<file_name>.json')
3. Modify
c = 5
to 
c = <assigned_category>
4. Run for each label but 1
python train_binary.py

5. Run to train binary class for label 1:
python train_binary1.py


