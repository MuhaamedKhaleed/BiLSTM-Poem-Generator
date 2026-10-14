# BiLSTM-Poem-Generator

A deep learning-based natural language generation engine that generates poetic verses using Bidirectional Long Short-Term Memory (BiLSTM) networks. The model processes poetic corpora, generates n-gram token sequences, and leverages a layered recurrent architecture with embedding and regularization techniques to synthesize coherent verse extensions from a seed phrase.

## Features

- **Bi-Directional Context Capture**: Leverages a `Bidirectional(LSTM)` layer combined with a secondary `LSTM` layer to understand both preceding and succeeding context during sequence training.
- **Overfitting Prevention**: Incorporates Dropout (0.2) and $L_2$ kernel weight regularization (`l2(0.01)`) to maintain generalization across custom poetic corpuses.
- **N-Gram Sequential Modeling**: Transforms arbitrary multi-line text into padded token sequences via Keras Tokenizer and one-hot encoded categorical labels.
- **Autonomous Text Generation**: Employs iterative next-token inference based on maximum probability (`argmax`) to construct extended sequences from custom seed text.

## Architecture

1. **Input Layer**: Sequence input shaped to `(max_sequence_len - 1,)`.
2. **Embedding Layer**: 100-dimensional dense word vector representations.
3. **Bidirectional LSTM Layer**: 150 units capturing forward and backward contextual states with sequence output preservation.
4. **Dropout Layer**: Rate 0.2 to mitigate co-adaptation of features.
5. **LSTM Layer**: 100 units collapsing sequential states into a single latent representation.
6. **Dense Projection Layer**: Hidden layer with `ReLU` activation and $L_2$ regularization.
7. **Output Layer**: Softmax projection over total vocabulary size.

## Tech Stack

- **Language**: Python 3.9+
- **Frameworks**: TensorFlow / Keras
- **Data Manipulation**: NumPy

## Project Structure

text
BiLSTM-Poem-Generator/
├── poem.txt             # Source text corpus (place your poetry here)
├── poem_generator.py    # Main training and generation script
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore patterns
└── README.md            # Project documentation


## Setup Instructions

### 1. Prerequisites
Ensure Python 3.9+ and pip are installed on your machine. For accelerated training, ensure appropriate GPU drivers and CUDA toolkit are installed.

### 2. Environment Setup
Clone the repository and initialize a virtual environment:

bash
git clone https://github.com/your-username/BiLSTM-Poem-Generator.git
cd BiLSTM-Poem-Generator
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
.\venv\Scripts\activate


### 3. Install Dependencies
bash
pip install --upgrade pip
pip install -r requirements.txt


### 4. Prepare Dataset
Place your training text file named `poem.txt` in the root directory. Each line should ideally represent a verse or poetic line.

### 5. Run Training and Generation
Execute the training script:

bash
python poem_generator.py


To change generation behavior, adjust the `seed_text` string and `next_words` integer at the bottom of the script.
