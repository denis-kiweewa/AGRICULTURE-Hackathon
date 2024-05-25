# AGRICULTURE-Hackathon
**Documentation: Running the YOLOv8 Cocoa Detection Notebook**

### Introduction
This documentation provides instructions on how to run the YOLOv8 Cocoa Detection Notebook. The notebook is designed to train a YOLOv8 object detection model to detect cocoa readiness in images. Below are the steps to run the notebook successfully.

### Prerequisites
Before running the notebook, ensure you have the following:

1. Access to a computing environment with GPU support, such as Google Colab or a local machine with a GPU.
2. Necessary data files:
   - Cocoa image dataset (`cocoa_new.zip`)
   - Label mapping CSV file (`label_map.csv`)

### Steps to Run the Notebook

#### 1. Setup Environment
- Ensure all required libraries are installed. This can be done by running the first cell of the notebook, which installs the necessary dependencies using pip.

#### 2. Download and Extract Data
- Execute the code in Cell Three to download the dataset (`cocoa_new.zip`) and extract it. This step will create a directory named `cocoa_new` containing the training images and label mapping CSV file.

#### 3. Data Exploration and Preprocessing
- Explore the dataset and inspect the label mapping CSV file using Cells Four to Nine.

#### 4. Generate YOLO Labels
- Use Cells Ten to Sixteen to generate YOLO format labels for the dataset. These labels are required for training the YOLOv8 model.

#### 5. Train the YOLOv8 Model
- Train the YOLOv8 model using Cells Twenty-Two to Twenty-Six. Specify the desired model version (`yolov8s.pt`, `yolov8x.pt`, etc.), training epochs, batch size, and other parameters.

#### 6. Evaluate the Model
- After training, evaluate the model's performance using Cells Twenty-Seven to Twenty-Eight.

#### 7. Fine-Tune the Model (Optional)
- Optionally, fine-tune the trained model for further improvements. Adjust parameters such as learning rate and batch size as needed.

### Conclusion
By following the above steps, you can successfully run the YOLOv8 Cocoa Detection Notebook and train a model to detect cocoa readiness in images. Adjust parameters and experiment with different model versions to optimize performance further.
