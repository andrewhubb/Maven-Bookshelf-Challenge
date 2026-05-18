# Maven Bookshelf Challenge

## What This Is
A large-scale NLP pipeline that processes approximately 1.1 million Goodreads book reviews
and 10,365 works records, applying sentiment analysis and emotion classification to each
review. Built as my entry for the Maven Analytics Bookshelf Challenge, and as a practical
application of the Maven Analytics NLP in Python course by Alice Zhao.

## What It Does
- Cleans and loads the Goodreads dataset into a MySQL database
- Runs VADER sentiment analysis across the full review dataset
- Runs transformer-based emotion classification (joy, sadness, anger, fear, surprise,
  disgust, neutral) using a fine-tuned DistilRoBERTa model
- Processes data in configurable chunks with full checkpoint/resume support —
  the pipeline can be interrupted and will pick up exactly where it left off
- Outputs results as Parquet files and writes the merged dataset to MySQL

## Tech Stack
- **Python 3.12**
- **PyTorch** — MPS backend (Apple Silicon) or CUDA backend (Windows/Linux)
- **Hugging Face Transformers** — `j-hartmann/emotion-english-distilroberta-base`
  for emotion classification, `distilbert-base-uncased-finetuned-sst-2-english`
  for sentiment
- **vaderSentiment** — lexicon-based sentiment scoring
- **pandas / PyArrow** — data processing and Parquet I/O
- **SQLAlchemy / PyMySQL** — MySQL database connectivity
- **tqdm** — progress tracking

## Dataset
Maven Analytics Bookshelf dataset:
- **reviews** — approximately 1.1 million Goodreads reviews
- **works** — 10,365 book records

## How to Run

### Prerequisites
- Python 3.12
- MySQL instance (local or networked)
- Apple Silicon Mac (MPS) or CUDA-capable GPU recommended for emotion analysis;
  CPU fallback is available but significantly slower

### Setup
```bash
git clone https://github.com/andrewhubb/Maven-Bookshelf-Challenge.git
cd Maven-Bookshelf-Challenge
pip install -r requirements.txt
```

Configure your MySQL connection details in the script before running.

### Run the pipeline

**Apple Silicon (MPS):**
```bash
python pipeline_apple_silicon.py
```

**Windows / CUDA GPU:**
```bash
python pipeline_cuda.py
```

The pipeline will resume automatically from the last saved checkpoint if interrupted.

## Project Status
Sentiment analysis and emotion classification stages are complete across both the reviews
and works datasets. NMF topic modelling is a potential future addition.

## License
Apache 2.0
