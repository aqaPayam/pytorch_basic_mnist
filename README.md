# PyTorch Basics: MNIST Classification

This repository contains a Jupyter notebook and a Python script that demonstrate the basics of building, training, and evaluating a neural network using PyTorch on the MNIST dataset. The notebook is intended as a practical introduction to PyTorch for beginners in deep learning.

## Repository Contents

- `pytorch_basic.ipynb`: Jupyter notebook for MNIST classification using PyTorch.
- `pytorch_basic.py`: Python script with additional PyTorch functions.

## Overview

The notebook covers the following topics:
- **Data Handling with PyTorch**: Loading and preprocessing the MNIST dataset.
- **Building a Neural Network**: Defining a simple feedforward neural network.
- **Training the Network**: Implementing a training loop and optimizing the model.
- **Evaluating the Model**: Testing the model on the test dataset and calculating performance metrics.

The Python script (`pytorch_basic.py`) includes utility functions for tensor operations and additional PyTorch practice.

## Prerequisites

To run the notebook and script, you need to have the following installed:
- Python 3.6 or later
- Jupyter Notebook
- PyTorch
- torchvision

You can install the required packages using the following commands:

```bash
pip install torch torchvision
pip install notebook
```

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/aqaPayam/pytorch_basic_mnist.git
    cd pytorch_basic_mnist
    ```

2. Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

3. Open the `pytorch_basic.ipynb` notebook and run the cells sequentially.

## Notebook Structure

### Imports and Setup
- Importing necessary libraries such as `torch`, `torchvision`, and `torch.nn`.
- Setting up data transformations for preprocessing the MNIST dataset.

### Data Transformation and Loading
- Loading the MNIST dataset using `torchvision.datasets`.
- Applying transformations using `torchvision.transforms`.
- Creating data loaders for batching and shuffling data.

### Model Definition
- Defining a neural network class using `torch.nn.Module`.
- Adding layers: fully connected layers, dropout, and ReLU activation.

### Training Function
- Implementing a training function to iterate over batches.
- Calculating loss and performing backpropagation.
- Updating model weights using an optimizer.

### Testing Function
- Implementing a function to evaluate the model on the test dataset.
- Calculating and printing the average loss and accuracy.

### Training Loop
- Running the training process for a specified number of epochs.
- Evaluating the model after each epoch.

### Results
- Displaying the test average loss and accuracy.

## Python Script (`pytorch_basic.py`)

The Python script contains the following functions:
- `hello()`: Prints a hello message.
- `create_sample_tensor()`: Creates a sample tensor.
- `mutate_tensor()`: Mutates specific elements in a tensor.
- `count_tensor_elements()`: Counts the number of elements in a tensor.
- `create_tensor_of_pi()`: Creates a tensor filled with the value 3.14.
- `multiples_of_ten()`: Generates multiples of ten within a range.
- `slice_indexing_practice()`: Demonstrates various slicing operations.
- `slice_assignment_practice()`: Demonstrates slice assignment operations.
- `shuffle_cols()`: Re-orders the columns of a tensor.
- `reverse_rows()`: Reverses the rows of a tensor.
- `take_one_elem_per_col()`: Extracts specific elements from each column.
- `count_negative_entries()`: Counts the number of negative values in a tensor.
- `make_one_hot()`: Creates one-hot encoded vectors.
- `reshape_practice()`: Reshapes a tensor using specific operations.
- `zero_row_min()`: Sets the minimum value in each row to zero.
- `batched_matrix_multiply()`: Performs batched matrix multiplication.
- `normalize_columns()`: Normalizes the columns of a matrix.
- `mm_on_cpu()`: Performs matrix multiplication on CPU.
- `mm_on_gpu()`: Performs matrix multiplication on GPU.

## Key PyTorch Functions and Methods

The notebook and script demonstrate the use of several important PyTorch functions and methods, including:

### Data Handling
- `torchvision.datasets.MNIST`
- `torchvision.transforms.Compose`
- `torchvision.transforms.ToTensor`
- `torchvision.transforms.Normalize`
- `torch.utils.data.DataLoader`

### Tensor Operations
- `torch.device`
- `.to()`

### Model Building
- `torch.nn.Module`
- `torch.nn.Linear`
- `torch.nn.Dropout`
- `torch.nn.ReLU`
- `torch.nn.CrossEntropyLoss`

### Training
- `torch.optim.Adam`
- `optimizer.zero_grad()`
- `optimizer.step()`

### Evaluation
- `torch.no_grad()`
- `.argmax()`
- `.item()`

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This notebook is inspired by the official PyTorch tutorials and various deep learning resources available online.

Feel free to customize this README to better fit your needs before uploading it to GitHub.
```

### Repository Setup Instructions

1. Create a new repository on GitHub named `pytorch_basic_mnist`.
2. Clone the repository to your local machine.
3. Copy the `pytorch_basic.ipynb` and `pytorch_basic.py` files to the cloned repository.
4. Add the `README.md` file to the repository.
5. Commit the changes and push them to GitHub.

Here are the steps in code form:

```bash
# Clone the repository (replace <your-username> with your GitHub username)
git clone https://github.com/aqaPayam/pytorch_basic_mnist.git
cd pytorch_basic_mnist

# Copy the files to the repository directory
cp /path/to/pytorch_basic.ipynb .
cp /path/to/pytorch_basic.py .

# Create the README.md file and add the content provided above

# Add, commit, and push the changes
git add pytorch_basic.ipynb pytorch_basic.py README.md
git commit -m "Initial commit with notebook, script, and README"
git push origin main
