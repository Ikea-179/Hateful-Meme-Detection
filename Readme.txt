##Hateful Meme Detection
This zip file includes all the notebooks for our project submission. 
The work was done by 2 people:
*Yijia Xue
*Ruitian Wu

The following texts introduce the purpose of each notebook.
*TF_IDF.ipynb This code implemented the TF-IDF based model(text unimodal) with different ML-based classifier
*TextCNN.ipynb This notebook implemented the Text-CNN based(text unimodal), with some exploratory data analysis 
*ResNet.ipynb implemented Resnet50 based(image unimodal) model
*BERT_Model.ipynb implemented the Bert-based(text unimodal) classification model
*Face Detection.ipynb implemented Haar Cascade Classifier model to detect the most important face in the image
*Object Detection.ipynb implemented the YOLOv5 model to detect the object in the image
*VisualBERT.ipynb implemented the final classification model based on Visual BERT with three channels input

Essential Guidance before running the notebooks:
1. Change the data loading path to your own data loading path
2. Follow the code to install necessary python library
3. For text-based model, you need to read the json file under the hateful_memes folder
   For image-based model, you need to download the meme image from the link to google drive
   For the final classification model, there are three channels' input, first you will use only text       	       	        
   And Image input to get the baseline model result. 
   Then, you need use 'new_merged.csv' to get three channel's input and implemented the final classification    
   model to get our final model's result.
4. If you use ImageFolder to read the image， you need first to create two folders to store all the image. Then you can run it.
5.### Link to google drive which contains the meme image
https://drive.google.com/file/d/1ohViVHFKOI3b8VAEt018aPER6V0coEhH/view?usp=sharing