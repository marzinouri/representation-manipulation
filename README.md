# Representation Manipulation with BERT

## Problem Definition

The goal of this project is to modify the output of a specific layer in a BERT model by integrating additional information through various methods. The solution should be flexible, allowing changes to the layer being modified and the method of integration.

## Solution

The project provides two primary approaches for integrating additional information into the BERT model:
1. **Hook-Based Approach**: Uses PyTorch hooks to modify the input of a specific layer during the forward pass.
2. **Custom Layer-Based Approach**: Defines a custom layer that integrates additional information.

Both approaches enable flexible manipulation of BERT's internal representations for various layers and integration methods.

## Project Structure

- `src`: Contains the source code for setting up the BERT model, defining integration methods, and implementing the hook-based and custom layer-based approaches.
- `test`: Contains test scripts to test the implementation.
- `notebooks`: Contains Jupyter notebooks.

## Notebooks
### 1. BERT Representation Modification
#### Overview
This notebook demonstrates the implementation of representation manipulation in a BERT model. It includes both hook-based and custom layer-based approaches to modify the internal representations of the BERT model.

#### Contents
- **Hook-Based Approach**: Modify the input of a specific layer during the forward pass using PyTorch hooks.
- **Custom Layer-Based Approach**: Define a custom layer that integrates additional information.

### 2. Representation Manipulation in BERT Sentiment Classification
#### Overview
This notebook explores the impact of manipulating the internal representations of a BERT model fine-tuned for sentiment classification. The main objective is to understand how manipulating the [CLS] vector influences the model’s performance and confidence.

#### Contents
- **Dta Preparation**: Load the IMDB dataset.
- **Model Finetuning**: Fine-tune a pre-trained BERT model for sequence classification.
- **Representation Manipulation**: For each test instance, manipulate its [CLS] vector by subtracting the average positive [CLS] vector.
- **Analysis of Results**: Analyze Confusion Matrices, Classification Reports, Confidence Analysis, Example Cases and Label Changes.
