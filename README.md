# BERT Sentiment Analyzer

This project is a sentiment analysis model based on BERT (specifically, DistilBERT) fine-tuned on the IMDB dataset. The model is trained to classify movie reviews as either positive or negative.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/khalildobapxl/BERT-Sentiment-Analyzer.git
    cd BERT-Sentiment-Analyzer
    ```

2.  Create a virtual environment and activate it:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project is organized into three Jupyter notebooks that walk through the process of data exploration, model training, and inference.

### 1. Exploratory Data Analysis (EDA)

The `notebooks/01_eda_imdb.ipynb` notebook covers the initial exploration of the IMDB dataset. It includes steps to load the data, analyze the distribution of labels, and examine the text length of the reviews.

### 2. Model Training

The `notebooks/02_modeling.ipynb` notebook details the model training process. It includes:
*   Loading the pre-trained DistilBERT model and tokenizer.
*   Preprocessing the data.
*   Setting up the training arguments and trainer.
*   Training the model and saving the final checkpoint.

### 3. Inference

The `notebooks/03_inference.ipynb` notebook demonstrates how to use the fine-tuned model for sentiment analysis on new text. It shows how to load the saved model and use the `pipeline` function from the `transformers` library to make predictions.

## Model

The trained model is a fine-tuned version of `distilbert-base-uncased`. The model checkpoint is saved in the `models/imdb-distilbert/checkpoint-3126` directory. The model was trained for 2 epochs with a learning rate of 2e-5. The model configuration includes labels for "POSITIVE" and "NEGATIVE" sentiment.

## Data

This project uses the IMDB dataset from the Hugging Face `datasets` library. The dataset consists of 25,000 movie reviews for training and 25,000 for testing, each with a corresponding sentiment label (positive or negative).
