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
The table below, shows the values of the metrics used, during each Epoch, and the training
process lasted about 4 hours:
<img width="358" alt="table" src="https://github.com/Emanuele1202/Image-segmentation-using-U-NET/assets/100868959/110414fa-86a8-42d1-a243-0d23c4fec138">


This training results show a gradual improvement in both the training and validation
metrics over the epochs, in fact:
• Loss and Dice Coefficient Trends: The training loss is decreasing, and the dice coefficient is gradually increasing, indicating that the model is learning to better fit the
training data. The validation loss and dice coefficient on the validation set also show
improvement, which is a positive sign.
• Early Stopping was triggered after 34 epochs, as there was no significant improvement
in the validation loss. This is mainly used to prevent overfitting.
• The final validation Dice Coefficient is 0.5998, which suggests that the model is performing reasonably well, indeed it means that we have an overlap of approximately
60% between the train image and the train mask.
Regarding the predicted mask, which basically is our main focus, I am reporting here
just few outputs results:

<img width="353" alt="result" src="https://github.com/Emanuele1202/Image-segmentation-using-U-NET/assets/100868959/f54bfc3d-442e-46cf-8327-a0ed88ed9f26">

Where we can see that, even with the 60% of overlapping, it is possible to understand
that the Net is learning well, but still struggle in more complicated scenario like the following:
<img width="346" alt="failed" src="https://github.com/Emanuele1202/Image-segmentation-using-U-NET/assets/100868959/15e72ca4-cbf1-40bf-8925-b56f2947e911">


## Conclusion
The U-Net model demonstrated the capability to perform unsupervised image segmentation effectively. Future work includes increasing epochs, experimenting with different architectures, and enhancing data augmentation to further improve performance.
