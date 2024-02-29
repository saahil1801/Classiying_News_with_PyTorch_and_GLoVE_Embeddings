# Classiying_News_with_PyTorch_and_GLoVE_Embeddings


# PyTorch Text Classification using Embeddings

This repository contains code for a simple text classification task using PyTorch and pre-trained word embeddings. We train a neural network classifier to classify news articles into one of four categories: World, Sports, Business, and Sci/Tech.

## Overview

In this project, we use PyTorch along with TorchText and pre-trained GloVe word embeddings to build and train a text classifier. Here's a brief overview of the main components:

- **Data**: We use the AG News dataset available in TorchText, which consists of news articles categorized into four classes: World, Sports, Business, and Sci/Tech.
  
- **Preprocessing**: Text data is tokenized using a basic English tokenizer provided by PyTorch. We then convert tokens into GloVe word embeddings.
  
- **Model**: We define a simple neural network classifier consisting of linear layers with ReLU activations.
  
- **Training**: The model is trained using the Adam optimizer and Cross-Entropy Loss over several epochs.
  
- **Evaluation**: We evaluate the trained model on a separate test set and report accuracy, classification report, and confusion matrix.

## Requirements

- Python 3.x
- PyTorch
- TorchText
- tqdm
- scikit-learn
- matplotlib

## Results

After training, the model's performance is evaluated on the test set, yielding accuracy, a classification report, and a confusion matrix visualized using matplotlib.

## Conclusion

This project demonstrates how to build a simple text classifier using PyTorch and pre-trained word embeddings. By leveraging TorchText and GloVe embeddings, we achieve decent performance on the text classification task.

Feel free to explore the code and experiment with different architectures or datasets! If you have any questions or suggestions, please open an issue or pull request. Happy coding!
