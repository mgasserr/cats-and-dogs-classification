# Cats and Dogs Classification with VGG16

This project demonstrates binary image classification for cats and dogs using transfer learning with **VGG16** in TensorFlow/Keras. The main implementation lives in the Jupyter notebook [cat_and_dog_vgg16.ipynb](cat_and_dog_vgg16.ipynb), which downloads the dataset, builds a preprocessing pipeline, trains a model, evaluates it, and visualizes predictions.

## Project Overview

The notebook follows this workflow:

1. Install the required Python packages.
2. Download the cats-vs-dogs dataset using KaggleHub.
3. Load the images with TensorFlow’s `image_dataset_from_directory` API.
4. Build a transfer learning model on top of pretrained VGG16 features.
5. Train the classifier on the training split and validate it on a held-out subset.
6. Evaluate the final model on the test set.
7. Plot training curves and show sample predictions.

## Repository Contents

- [cat_and_dog_vgg16.ipynb](cat_and_dog_vgg16.ipynb): end-to-end notebook for dataset download, model training, evaluation, and visualization.
- [README.md](README.md): project documentation and usage guide.

## Model Summary

The notebook uses a pretrained **VGG16** base model with the convolutional layers frozen. A custom classification head is added on top for binary prediction:

- input resizing to `224 x 224`
- VGG16 preprocessing
- pretrained VGG16 feature extractor
- flatten layer
- dense layer with ReLU activation
- dropout for regularization
- final sigmoid output for binary classification

This setup is a standard transfer learning approach that works well when you have a moderate dataset and want to reuse strong visual features learned from ImageNet.

## Dataset

The notebook downloads the dataset from Kaggle using:

```python
kagglehub.dataset_download("tongpython/cat-and-dog")
```

It expects the following directory layout after download:

- `training_set/training_set`
- `test_set/test_set`

The notebook uses the training directory to create an internal validation split and uses the test directory for final evaluation.

## Requirements

The notebook installs the main runtime dependencies automatically:

- `tensorflow`
- `kagglehub`
- `matplotlib`
- `numpy`

If you are running locally, make sure you also have access to KaggleHub and an environment that can install TensorFlow successfully.

## How to Run

### Option 1: Open in Google Colab

The notebook includes an Open in Colab badge at the top. You can open it in Colab and run the cells there.

### Option 2: Run Locally

1. Clone or download the repository.
2. Open [cat_and_dog_vgg16.ipynb](cat_and_dog_vgg16.ipynb) in Jupyter Notebook, JupyterLab, or VS Code.
3. Run the notebook cells in order.
4. Wait for the dataset to download, then train and evaluate the model.

## Expected Output

When the notebook runs successfully, you should see:

- dataset download confirmation
- training and validation accuracy/loss for each epoch
- final test accuracy
- plots of training history
- sample predictions on test images with the predicted class and confidence

## Notebook Structure

The notebook is organized into two main code sections:

1. Data preparation, model creation, training, and evaluation.
2. Training curve visualization and prediction inspection.

## Notes

- The notebook is set to use a small number of epochs for quick validation. You can increase `EPOCHS` for better performance.
- The base VGG16 layers are frozen, so the notebook trains only the custom classification head.
- Because the dataset is loaded with a validation split, the training directory must contain class subfolders for cats and dogs.

## Troubleshooting

- If dataset download fails, check your internet connection and confirm that KaggleHub can access the dataset.
- If TensorFlow installation fails locally, use a Python environment that supports your platform and version.
- If you get directory errors, verify that the downloaded dataset still contains the nested `training_set/training_set` and `test_set/test_set` paths.
- If training is slow, reduce the batch size or use a GPU-enabled environment.

## License

No license file is currently included in this repository. Add one if you plan to share or reuse the project publicly.