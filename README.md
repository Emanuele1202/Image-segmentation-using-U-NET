# REMY Robotics Entry Project

## Introduction
This project focuses on image segmentation using the U-Net architecture for unsupervised image segmentation. Image segmentation involves dividing an image into meaningful segments, which helps in identifying and processing relevant parts without analyzing the entire image.

## U-Net Architecture
U-Net is a fully convolutional neural network designed for semantic segmentation tasks. It consists of an encoder that extracts features and a decoder that up-samples these features to generate detailed segmentation masks. The architecture is symmetrical, with skip connections between the encoder and decoder to combine spatial and semantic information effectively.

## Unsupervised Image Segmentation
Unsupervised segmentation aims to categorize pixels based on visual characteristics without labeled data. This approach is useful when labeled datasets are difficult or expensive to obtain. The project utilizes clustering methods and graph-based techniques to generate pseudo-labels, which are then used to train the U-Net model.

## Parameters Overview
Key parameters used in training include:
- **Epochs**: Number of complete passes through the training dataset.
- **Loss Function**: Binary crossentropy to measure prediction accuracy.
- **Metrics**: Dice coefficient to evaluate segmentation performance.
- **Optimizer**: Adam for adaptive learning rates.
- **Learning Rate**: Controls the step size in parameter updates.
- **Callbacks**: Functions like EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau to manage training.
- **ImageDataGenerator**: Augments training data to improve model generalization.
- **Batch Size**: Number of training examples per iteration.
- **Validation Split**: Portion of data used for validation during training.

## Results
The model was trained for 34 epochs, achieving a final validation Dice Coefficient of 0.5998, indicating reasonable segmentation performance. The predicted masks showed good alignment with ground truth, although improvements are possible with more training and parameter tuning.

## Example Results

## Conclusion
The U-Net model demonstrated the capability to perform unsupervised image segmentation effectively. Future work includes increasing epochs, experimenting with different architectures, and enhancing data augmentation to further improve performance.
