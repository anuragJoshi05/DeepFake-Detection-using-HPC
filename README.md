## **DeepFake Detection using HPC**

### **Overview**
This project focuses on detecting deepfake images using a **Convolutional Neural Network (CNN)** combined with **High-Performance Computing (HPC)** techniques. By leveraging **parallel computing**, the model is trained using both sequential and distributed strategies, significantly improving performance. The detection pipeline includes **image preprocessing, feature extraction, and classification** using deep learning.

### **Key Features**
- **Deepfake Image Classification** using a CNN-based model
- **Parallel Training** with multi-GPU support for faster performance
- **Error Level Analysis (ELA)** to detect tampering artifacts
- **Optimized Performance** using data-level and model-level parallelism
- **Efficient Preprocessing Pipeline** including compression, grayscale conversion, and pixel analysis

### **Architecture**
The deepfake detection pipeline consists of the following steps:

1. **Input Image:** The model takes an image (or video frame) as input.
2. **Image Compression:** Reveals hidden tampering artifacts.
3. **Difference Calculation (ELA):** Generates an error level analysis image to highlight discrepancies.
4. **Feature Extraction:** Identifies key patterns for deepfake detection.
5. **Grayscale Conversion & Bit Computation:** Converts the image to grayscale and refines pixel values.
6. **Pixel Value Extraction:** Extracts essential pixel features.
7. **Reshape Input Image:** Ensures compatibility with the CNN.
8. **Dataset Creation:** Processed images are stored for training.
9. **CNN Model:** Classifies images as real or fake.
10. **Parallel Training:** Uses multi-GPU training for improved efficiency.

### **CNN Model Architecture**

| Layer Type             | Output Shape      | Parameters |
|------------------------|------------------|------------|
| Conv2D (32 filters, 3x3) | (126, 126, 32) | 896 |
| BatchNormalization     | (126, 126, 32)  | 128 |
| MaxPooling2D (2x2)     | (63, 63, 32)   | 0 |
| Conv2D (64 filters, 3x3) | (61, 61, 64) | 18,496 |
| BatchNormalization     | (61, 61, 64)  | 256 |
| MaxPooling2D (2x2)     | (30, 30, 64)   | 0 |
| Flatten               | (57600)         | 0 |
| Dense (128 neurons)   | (128)          | 7,372,928 |
| Dropout (0.5)        | (128)          | 0 |
| Dense (2 neurons)    | (2)            | 258 |

**Total Parameters:** 7,392,962

### **High-Performance Computing (HPC) Setup**

#### **1️⃣ Data-Level Parallelism:**
- Splits the dataset across multiple GPUs
- Compares **synchronous vs. asynchronous training**
- Uses **gradient aggregation** to update the global model

#### **2️⃣ Model-Level Parallelism:**
- Distributes different CNN layers across GPUs
- Implements **tensor and pipeline parallelism**

### **Experimental Configuration**
- **Dataset:** CASIA1 (real images) & CASIA2 (tampered images)
- **Image Size:** 128x128 pixels
- **Epochs:** 30 (with early stopping)
- **Batch Size:** 32
- **Learning Rate:** 0.0001 (decay: 0.000001)
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Data Split:** 80% training, 20% validation

### **Results and Analysis**

#### **Training Time Comparison**

| Training Method | Time Taken (seconds) |
|----------------|----------------------|
| Sequential (5 epochs) | 2436.73 |
| Parallel (30 epochs) | 196.47 |

#### **Accuracy Comparison**

| Training Method | Accuracy |
|----------------|----------|
| HPC Training | **92.7%** |
| Non-HPC Training | 87.0% |

#### **Impact of Error Level Analysis (ELA)**
- **Without ELA:** Accuracy = **78%**
- **With ELA:** Accuracy = **92.7%**

### **Conclusion**
This project demonstrates the effectiveness of deepfake detection using CNNs and **HPC techniques**. By leveraging **distributed training and parallel computing**, the model achieves:
- **Significantly faster training times**
- **Higher accuracy in deepfake detection**
- **Efficient handling of large datasets using multi-GPU setups**
