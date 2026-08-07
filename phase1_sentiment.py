import os
import scipy.io
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from transformers import pipeline
from datasets import load_dataset

class ZuCoDataLoader:
    """
    Data Loader for the ZuCo (Zurich Cognitive Language Processing) dataset.
    Specifically parses .mat files for Task 3 (Sentiment Reading).
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_subject_data(self, subject_id):
        """
        Loads the .mat file for a specific subject and extracts Task 3 data.
        Returns a structured DataFrame.
        """
        file_path = os.path.join(self.data_dir, f"task3-SR", f"{subject_id}.mat")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ZuCo data not found at {file_path}. Please download the dataset.")
        
        # Load Matlab file
        mat_data = scipy.io.loadmat(file_path)
        
        # The structure of ZuCo .mat files is complex (typically nested structs).
        # We will parse the sentence text and the corresponding EEG arrays.
        sentences_data = []
        
        # NOTE: This parsing logic assumes the standard ZuCo v1.0 / v2.0 struct format.
        # Usually data is stored in mat_data['sentenceData']
        if 'sentenceData' in mat_data:
            sentence_structs = mat_data['sentenceData'][0]
            for s in sentence_structs:
                content = s['content'][0] if len(s['content']) > 0 else ""
                # Mock extraction of EEG data corresponding to the reading epoch
                # In reality, this would be `s['word'][0]['EEG']` concatenated or epoch-based
                eeg_data = None # Placeholder for actual EEG array extraction
                sentences_data.append({
                    'text': content,
                    'eeg_raw': eeg_data
                })
        
        return pd.DataFrame(sentences_data)

class NLPSentimentPipeline:
    """
    Transformer-based NLP pipeline to extract sentiment from reading sentences.
    """
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        # Using a model pre-trained on SST-2 since ZuCo Task 3 sentences are from SST
        print(f"Loading NLP Model: {model_name}...")
        self.sentiment_model = pipeline("text-classification", model=model_name, device=-1) # -1 for CPU, 0 for GPU if available

    def extract_sentiment(self, texts):
        """
        Extracts continuous-like sentiment scores or polarity.
        """
        results = self.sentiment_model(texts)
        scores = []
        labels = []
        for res in results:
            # Convert label to polarity (-1.0 to 1.0)
            # POSITIVE approaches 1.0, NEGATIVE approaches -1.0
            label = res['label']
            score = res['score']
            if label == 'NEGATIVE':
                polarity = -score
                pred_label = 0
            else:
                polarity = score
                pred_label = 1
            scores.append(polarity)
            labels.append(pred_label)
            
        return scores, labels

def evaluate_nlp_model():
    """
    Evaluates the NLP Sentiment Model against SST-2 ground truth (which is the source for ZuCo Task 3).
    Returns the F1 Score.
    """
    print("Loading SST-2 Dataset for ground truth validation...")
    # Load the validation split of SST-2
    dataset = load_dataset("sst2", split="validation")
    
    texts = list(dataset['sentence'])
    true_labels = list(dataset['label'])
    
    nlp = NLPSentimentPipeline()
    
    print("Running inference on validation set...")
    _, pred_labels = nlp.extract_sentiment(texts)
    
    print("\n--- NLP Sentiment Validation Results ---")
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    print(f"Weighted F1 Score: {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=['Negative', 'Positive']))
    
    return f1

if __name__ == "__main__":
    # 1. Report NLP Model F1 score against ground truth as requested
    evaluate_nlp_model()
    
    # 2. Example usage of the ZuCo Data Loader (mocked execution)
    # data_loader = ZuCoDataLoader("path/to/zuco/data")
    # df = data_loader.load_subject_data("ZAB") # e.g., subject ZAB
    # print(df.head())
