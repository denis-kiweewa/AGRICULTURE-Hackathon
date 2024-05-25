# AgricHackathon_Mak_Ai_Lab

STEPS TAKEN

IMPORT THE DATASET

**EDA (Exploratory Data Analysis)**

The dataset needs to be analysed ,after which it will be prepared forthe model. The following things have been done to analyse the data.

1.   Finding out the number of images in each folder ie. Train and Validation
2.   Checking for imbalance in the dataset
3.   Visualisation of a few random images in the dataset. This was done manually.

NB: To run this code cell , please import helper_functions.py into google colab. helper_functions.py can be obtained from the requirements file

**MODEL**

Now that the data is ready to be fed to the model. The question is which is the best model to use!. Three different experiments were run using 10% of the data to select the model that gives the best performance . These included;
1.   EfficientNetB0
2.   ResnetV250
3.   VGG19

Ultimately EfficientNetBO was chosen because of two reasons ;
1.   Higher Accuracy
2.   Small size (Almost 5x smaller than ResnetV250 and almost 10x smaller than VGG19) for deployment in a Mobile App which is one of the objectives of this Hackathon

This section has two parts.Firstly training a feature extraction model (EffiientnetB0) and secondly fine-tuning the model. Due to colabs limits, these parts were done sequentially but separately. Ie. Use first 3hrs limit for the first part --> Save model --> Load saved model and use it for second part(Another 3hr Limit)

**EFFICIENTNETB0**

Here, a pretrained transfer learning model trained on imagenet was used. Efficientnet performed very well on imagenet with fewer parameters. More information can be found on https://arxiv.org/pdf/1905.11946 

Only the top layer was removed and replaced with a dense layer with softmax activation. Before we get started with the modelling we need to set up 4 things;
**Mixed Precision Training:** Increases training speed and reduces memory usage by using lower precision (float16) while maintaining model accuracy with selective higher precision (float32).This works very well with GPUs with compute capability 7.0 or higher such as T4
**Tensorboard Callback:** Enables real-time visualization of metrics, training progress, and model graphs within TensorBoard, aiding in debugging and optimization.
**Early Stopping Callback**: Halts training when the monitored metric (e.g., validation loss) stops improving after a specified number of epochs, preventing overfitting and saving resources.
**ReduceLROnPlateau Callback**: Reduces the learning rate when the monitored metric (e.g., validation loss) stops improving, helping the model to converge more effectively.

**FINE-TUNING THE EFFICIENTNETB0 MODEL**
During fine-tuning, a number of top layers are unfrozen and trained on the custom dataset

Time to finetune the model!. Due to Serial Experimentation (over 25 experiments) and Research, the following model and dataset parameter values are the best proven values that yielded the best results. These parameters are used across the entire notebook.

**Parameter : Value**
*   Dropout : 0.2
*   Image size : 448,448
*   Batch size : 32
*   Number of layers finetuned : 118
*   Augmentation layers : None (Basically no augmentation was done)
*   Patience value for early stopping callback : 5
*   Patience value for ReduceLROnPlateau callback : 3

**SUBMISSION**
The finetuned model that was saved above is now loaded to be used for submission. At the end of this notebook the csv file was downloaded and uploaded to the leaderboard

**NB 1:** There was one issue encountered with the test data . Particularly with image 492 having a colour error being labelled as corrupted and errored out the code for submission. Therefore the image was manually cropped to remove the colour error. Therefore when running this notebook , The test set was downloaded (about 1.85gb) --> Image 492 corrected --> The dataset uploaded to a google drive and the link copied. 

**NB:2:** The predictions made in this part of the code were rounded off to 4dp as was shown in an image that was used in the "making a submission" notebook provided under the Hackathon Github repository. As a team we feel that better f1 scores might have been obtained had the rounding been to like 2dp . However we have to adhere to the stict guidelines.

**WHAT TO DO NEXT**

The following are the potential next steps for this notebook;
1. **Finding ways to work around the issue of class imbalance**. Smote and Upsampling were experimented however, attempts were futile as the ram kept crashing.  
2. **Creating a more efficient data pipeline**. This will improve dataset loading and model training speeds. As well as possibly answering the challenge in 1 above.
