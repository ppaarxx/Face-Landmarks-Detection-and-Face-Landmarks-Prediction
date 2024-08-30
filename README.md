<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Facial Landmark Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            background-color: #f4f4f9;
            color: #333;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        p {
            margin-bottom: 15px;
        }
        code {
            background-color: #eee;
            padding: 2px 4px;
            border-radius: 4px;
            font-size: 90%;
        }
        .code-block {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            padding: 10px;
            overflow-x: auto;
            margin-bottom: 20px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <h1>Facial Landmark Detection</h1>

    <p>
        Facial landmark detection is a critical task in computer vision, focusing on identifying key facial features like the eyes, nose, mouth, and jawline. This is essential for various applications, including head pose estimation, facial gesture recognition, and face swapping. This project leverages deep learning models and PyTorch to detect facial landmarks in images. Below is a detailed breakdown of the components and processes involved.
    </p>

    <h2>1. Dataset Preparation</h2>
    <p>
        The <code>ibug_300W_large_face_landmark_dataset</code> dataset is utilized for training and testing the model. This dataset contains a large collection of face images, each annotated with 68 landmark points. These points represent the key facial features that the model aims to predict. The dataset is initially downloaded and extracted to Google Drive for ease of access and storage.
    </p>

    <div class="code-block">
        <code>
            # Code to download and extract dataset<br>
            # (Note: This part is commented out to ensure Python compatibility)<br>
            # if not os.path.exists('/content/ibug_300W_large_face_landmark_dataset'): ... 
        </code>
    </div>

    <h2>2. Data Augmentation</h2>
    <p>
        Data augmentation techniques are implemented to enhance the model's ability to generalize across different scenarios. The <code>Transforms</code> class defines several augmentation methods such as rotation, resizing, color jittering, and face cropping. These augmentations simulate real-world variations in facial images, improving the robustness of the model.
    </p>

    <div class="code-block">
        <code>
            class Transforms():<br>
            &nbsp;&nbsp;&nbsp;&nbsp;def rotate(self, image, landmarks, angle):<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... # rotation logic<br>
            &nbsp;&nbsp;&nbsp;&nbsp;def resize(self, image, landmarks, img_size):<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... # resizing logic<br>
            &nbsp;&nbsp;&nbsp;&nbsp;...
        </code>
    </div>

    <h2>3. Data Loader</h2>
    <p>
        A custom PyTorch <code>Dataset</code> class named <code>FaceLandmarksDataset</code> is created to load images and their corresponding landmarks. This class handles the parsing of XML files for landmark annotations and ensures that each image is paired with the correct set of landmarks. Additionally, it applies the defined transformations to each image-landmark pair.
    </p>

    <div class="code-block">
        <code>
            class FaceLandmarksDataset(Dataset):<br>
            &nbsp;&nbsp;&nbsp;&nbsp;def __init__(self, transform=None):<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... # Initialization logic<br>
            &nbsp;&nbsp;&nbsp;&nbsp;def __len__(self):<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return len(self.image_filenames)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;def __getitem__(self, index):<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... # Image and landmark loading logic<br>
        </code>
    </div>

    <h2>4. Model Architecture</h2>
    <p>
        The model architecture is based on ResNet-18, a popular convolutional neural network (CNN) known for its efficiency and performance in image-related tasks. The network is modified to output 136 values, corresponding to the x and y coordinates of the 68 landmark points. This architecture enables the model to effectively learn the spatial relationships between different facial features.
    </p>

    <div class="code-block">
        <code>
            class Network(nn.Module):<br>
            &nbsp;&nbsp;&nbsp;&nbsp;def __init__(self, num_classes=136):<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;super().__init__()<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.model = models.resnet18(pretrained=True)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... # Model adjustment logic<br>
        </code>
    </div>

    <h2>5. Training and Validation</h2>
    <p>
        The model is trained using the Mean Squared Error (MSE) loss function, which is well-suited for regression tasks like landmark detection. An Adam optimizer is employed to update the model weights iteratively based on the computed gradients. The training loop also includes validation after each epoch to monitor the model's performance on unseen data and to implement early stopping when the validation loss improves.
    </p>

    <div class="code-block">
        <code>
            for epoch in range(1, num_epochs + 1):<br>
            &nbsp;&nbsp;&nbsp;&nbsp;... # Training logic<br>
            &nbsp;&nbsp;&nbsp;&nbsp;network.eval()<br>
            &nbsp;&nbsp;&nbsp;&nbsp;with torch.no_grad():<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... # Validation logic<br>
        </code>
    </div>

    <h2>6. Inference</h2>
    <p>
        After training, the model is evaluated on the test set to gauge its ability to predict landmarks on new, unseen images. The predicted landmarks are plotted alongside the ground truth landmarks to visually assess the model's accuracy. This step is crucial for validating the model's generalization capability.
    </p>

    <div class="code-block">
        <code>
            with torch.no_grad():<br>
            &nbsp;&nbsp;&nbsp;&nbsp;... # Load the best model<br>
            &nbsp;&nbsp;&nbsp;&nbsp;... # Predict landmarks on test images<br>
            &nbsp;&nbsp;&nbsp;&nbsp;... # Plot the predictions vs. ground truth<br>
        </code>
    </div>

    <h2>7. Conclusion</h2>
    <p>
        This project demonstrates the end-to-end pipeline for facial landmark detection using deep learning techniques. From dataset preparation to model training and validation, each step is crucial in building a robust model capable of accurately detecting facial landmarks. Future work could involve experimenting with more advanced architectures or incorporating additional datasets to further improve the model's performance.
    </p>
</body>
</html>
