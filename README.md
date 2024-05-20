# Representation Manipulation with BERT

## Problem Definition

The goal of this project is to modify the output of a specific layer in a BERT model by integrating additional information through various methods. Specifically, we aim to manipulate the input of the 10th layer by adding an additional input vector. The solution should be flexible, allowing changes to the layer being modified and the method of integration.

## Solution

The project provides two primary approaches for integrating additional information into the BERT model:
1. **Hook-Based Approach**: Uses PyTorch hooks to modify the input of a specific layer during the forward pass.
2. **Custom Layer-Based Approach**: Defines a custom layer that integrates additional information.

Both approaches enable flexible manipulation of BERT's internal representations for various layers and integration methods.

## Project Structure

- `src`: Contains the source code for setting up the BERT model, defining integration methods, and implementing the hook-based and custom layer-based approaches.
- `test`: Contains test scripts to test the implementation.
- `notebooks`: Contains the Jupyter notebook.
