# CNN Based Image Classification for Returned Products

This project automates the classification of returned products in an online shopping platform using a Convolutional Neural Network (CNN) model. The system processes images of returned items, predicts their categories, and stores the results for further analysis.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Data](#data)
- [Model](#model)
- [System Architecture](#system-architecture)
- [Usage](#usage)
- [Results](#results)

## Project Overview
As e-commerce platforms expand, the volume of returned items increases, making efficient categorization essential for managing inventory, processing refunds, and maintaining customer satisfaction. Traditional manual sorting is labor-intensive, error-prone, and lacks scalability. To address this, the project leverages machine learning (ML) techniques, specifically Convolutional Neural Networks (CNNs), to automate the classification of returned items in an online fashion store.
CNNs are highly effective for image processing tasks due to their ability to capture spatial hierarchies in visual data. They have been widely adopted in various applications, including inventory management and quality control, due to their high accuracy and efficiency. This project integrates a CNN-based image classification model into a batch processing system, automating the categorization of returned items and reducing the need for manual intervention.
The primary objective of this project is to develop an automated system for classifying images of returned items in an online fashion store, thereby reducing the manual effort and cost associated with the categorization process. Specific objectives include:

1. **Model Development:** Create a CNN-based model capable of accurately classifying images of returned items into predefined categories.
2. **System Integration:** Seamlessly connect the model with the API for batch processing.
3. **System Automation:** Automate the classification process using cron jobs.

## Installation

To set up the project on your local machine, follow these steps:

### Prerequisites

Ensure that you have the following software installed:

- **Python 3.x**: The project is developed using Python 3.x. Make sure Python is installed on your machine.
- **Pip**: Python's package installer. You can check if you have it by running `pip --version` in your terminal.
- **Git**: For cloning the repository.

### Step-by-Step Installation

1. **Clone the Repository**

    First, clone the repository from GitHub:

    ```bash
    git clone https://github.com/Achshah-RM/CNN-Based-Image-Classification-for-Returned-Products.git

2. **Navigate to the Project Directory**

    Move into the project directory:
  
    ```bash
    cd repository-name

3. **Install the Required Python Packages**

    Install the necessary Python packages manually using pip. The following are the main libraries required:
    
    ```bash
    pip install tensorflow
    pip install flask
    pip install pandas
    pip install numpy
    pip install requests
    pip install pillow

4. **Download the Dataset**

    Ensure the dataset files (fashion-mnist_train.csv and fashion-mnist_test.csv) are placed in the Data directory within the project. You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist) or directly use the ones provided in the repository.

5. **Generate Images from the CSV Dataset**

    Run the script to generate images from the CSV files:
    
    ```bash
    python generate_image.py

6. **Run the Flask Application**

    Start the Flask API by running the following command:
    
    ```bash
    python app.py

7. **Test the Batch Processing**

    Run the batch processing script to send requests to the API and generate predictions:
    
    ```bash
    python batch_request.py

8. **Setting Up the Cron Job (Optional)**
   
    If you want to automate the batch processing, set up a cron job (on Unix-based systems) as detailed in the documentation. Ensure you adjust the cron job's timing as per your requirements.
    Add the following line to schedule the batch processing at 2:00 AM daily:
    
    ```bash
    crontab -e
    0 2 * * * /opt/anaconda3/bin/python3 /path/to/your/batch_request.py

This should be everything you need to set up the project and run the system on your local machine. Ensure you follow each step carefully to avoid any issues during installation.
## Data

### Dataset Overview

The project utilizes the **Fashion-MNIST** dataset, which is a collection of Zalando's article images. This dataset is designed to be a drop-in replacement for the original MNIST dataset, widely used for benchmarking machine learning algorithms. The Fashion-MNIST dataset provides a more challenging alternative by featuring images of fashion items instead of handwritten digits.

### Dataset Details

- **Training Set**: 60,000 examples
- **Test Set**: 10,000 examples
- **Image Size**: 28x28 pixels, grayscale
- **Total Pixels per Image**: 784 (28x28)
- **Pixel Value Range**: 0 (white) to 255 (black)

Each image is associated with a label from one of the following 10 categories:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

### Acknowledgements

The dataset was originally downloaded from the [Fashion-MNIST GitHub repository](https://github.com/zalandoresearch/fashion-mnist).

### License

The dataset is provided under the MIT License © 2017 Zalando SE. The full license can be viewed in the [Fashion-MNIST GitHub repository](https://github.com/zalandoresearch/fashion-mnist).

## Model

The Convolutional Neural Network (CNN) architecture designed for this project is tailored to process 28x28 grayscale images from the Fashion MNIST dataset and classify them into one of ten categories. The model is composed of several layers, each playing a critical role in feature extraction and classification.

- **Convolutional Layers:** These layers progressively detect features at different levels of abstraction, starting with basic edges and progressing to more complex patterns like textures and shapes.
- **Pooling Layers:** MaxPooling layers follow each convolutional layer to reduce the spatial dimensions of the feature maps, which helps in reducing computational complexity while retaining essential features.
- **Flatten Layer:** Converts the 3D feature maps into a 1D vector, preparing the data for the fully connected layers.
- **Fully Connected Layers:** Integrate the features learned in the convolutional layers and make a classification decision. A dropout layer is included to prevent overfitting.
- **Output Layer:** This layer uses the softmax activation function to produce a probability distribution over the 10 categories, allowing the model to classify each input image accurately.

For a detailed description of the model architecture, including the layer breakdown and training process, please refer to the `Documentation (Project Report).pdf` file.

## System Architecture

The system architecture is designed to facilitate efficient and accurate processing of image data, ensuring seamless interaction between components:

- **Image Directory:** Stores the image files that serve as the input for batch processing.
- **Batch Processing:** A scheduled script reads images from the directory and sends them to the API for processing.
- **Data Ingestion (API Module):** The API receives images, normalizes, resizes, and reshapes them to prepare for model processing.
- **Model Serving (API Module):** The processed images are classified by the CNN model, which generates predictions.
- **Prediction Return:** The predictions are sent back to the Batch Processing script.
- **Output:** The Batch Processing script saves the predictions into a CSV file for further analysis or reporting.

For a more detailed explanation of the system components and data flow, please refer to the `Documentation (Project Report).pdf` file.

## Usage

Once the batch_process.py file is triggered by the cron job, the system will process the images and generate predictions, which will be saved in a file called predictions.csv in your local directory.

To evaluate the accuracy of the predictions, you can run the model_accuracy.py script.

    ```bash
    python model_accuracy.py
    
This script will compare the predicted labels with the actual labels and display the accuracy of the model's predictions.

## Results

The model's performance was evaluated based on accuracy and loss metrics. During the training process, both training and validation accuracy increased steadily, while the corresponding losses decreased, indicating effective learning. The final evaluation on the test dataset yielded an accuracy of 90.61% and a loss of 0.27, suggesting that the model generalizes well to unseen data, making it suitable for real-world deployment. To further analyze the model's performance across different categories, a confusion matrix was generated. The confusion matrix and the accompanying classification report highlight the areas where the model excelled and where it struggled, offering insight into potential improvements for future iterations.

For a more detailed breakdown of the results, including precision, recall, and F1-score for each category, please refer to the `Documentation (Project Report).pdf` file.
