import os
import glob
import scipy.io
import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
import mne
from mne.preprocessing import ICA
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore")

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# ==========================================
# DATA LOADING
# ==========================================
class ZuCoDataLoader:
    def __init__(self, data_dir="./Preprocessed"):
        self.data_dir = data_dir

    def load_all_subjects(self):
        """Loads ZuCo sentences and EEG data."""
        print("Loading sentences from sentencesTSR.mat...")
        sentences_path = os.path.join(self.data_dir, "sentencesTSR.mat")
        if not os.path.exists(sentences_path):
            print(f"Error: {sentences_path} not found.")
            return pd.DataFrame()
            
        mat_sentences = scipy.io.loadmat(sentences_path)
        # sentences is of shape (407, 1)
        all_sentences = [str(s[0][0]) for s in mat_sentences['sentences']]
        
        all_data = []
        subject_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for subj in subject_dirs:
            print(f"Processing real EEG for subject: {subj}")
            subj_path = os.path.join(self.data_dir, subj)
            # Find all EEG files for this subject
            eeg_files = sorted(glob.glob(os.path.join(subj_path, "*_EEG.mat")))
            
            # Combine the continuous EEG arrays for the subject
            subj_eeg_data = []
            for eeg_file in eeg_files:
                try:
                    with h5py.File(eeg_file, 'r') as f:
                        # EEGLAB h5py structure: eeg['EEG']['data']
                        data_matrix = f['EEG']['data'][()]
                        # transpose if needed to (channels, times)
                        if data_matrix.shape[0] > data_matrix.shape[1]:
                            data_matrix = data_matrix.T
                        subj_eeg_data.append(data_matrix)
                except Exception as e:
                    print(f"Error reading {eeg_file}: {e}")
            
            if not subj_eeg_data:
                continue
                
            # Concatenate all blocks for this subject
            continuous_eeg = np.concatenate(subj_eeg_data, axis=1)
            n_channels = continuous_eeg.shape[0]
            sfreq = 500 # standard ZuCo sfreq
            
            # Since we have 407 sentences, we will chunk EEG into 407 epochs.
            total_timepoints = continuous_eeg.shape[1]
            epoch_len = total_timepoints // len(all_sentences)
            
            for i, text in enumerate(all_sentences):
                start_idx = i * epoch_len
                end_idx = start_idx + epoch_len
                epoch_data = continuous_eeg[:, start_idx:end_idx]
                
                all_data.append({
                    'subject_id': subj,
                    'sentence_id': i,
                    'text': text,
                    'eeg_raw': epoch_data,
                    'n_channels': n_channels,
                    'sfreq': sfreq
                })
                
        return pd.DataFrame(all_data)

# ==========================================
# DUAL-STREAM FEATURE EXTRACTION
# ==========================================
class DualStreamNLP:
    def __init__(self, lexicon_path="./NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"):
        self.lexicon_path = lexicon_path
        print("Loading Valence Model (DistilBERT)...")
        self.valence_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
        self.arousal_dict = self._load_nrc_vad()

    def _load_nrc_vad(self):
        print(f"Loading Real NRC-VAD Lexicon from {self.lexicon_path}...")
        arousal_dict = {}
        if not os.path.exists(self.lexicon_path):
            print(f"Error: {self.lexicon_path} not found.")
            return {"happy": 0.7, "sad": 0.3}
            
        try:
            with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        try:
                            word, valence, arousal = parts[0], float(parts[1]), float(parts[2])
                            arousal_dict[word.lower()] = arousal
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Failed to load lexicon: {e}")
        return arousal_dict

    # ==========================================
    # PRE-PROCESSING
    # ==========================================
    def preprocess_text(self, raw_text):
        """
        Implements the complete NLP pipeline.
        Tokenization -> Stop Word Removal -> Lemmatization
        """
        # Step 1: Clean and Tokenization
        clean_text = re.sub(r'[^\w\s]', '', raw_text).lower()
        tokens = word_tokenize(clean_text)
        
        # Step 2: Stop Word Removal
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [w for w in tokens if w not in stop_words]
        
        # Step 3: Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
        
        return lemmatized_tokens

    # ==========================================
    # DistilBERT
    # ==========================================
    def get_valence(self, text):
        """
        Uses DistilBERT (Transformer Encoder) to predict sentence valence.
        """
        if not text: return 0.0
        res = self.valence_model(text[:512])[0]
        score = res['score']
        return score if res['label'] == 'POSITIVE' else -score

    def get_arousal(self, text):
        """
        Calculates arousal using the NRC-VAD Lexicon.
        Combines traditional Lexicon-based NLP with pre-processed tokens.
        """
        if not text: return 0.0
        # Use our pipeline for token extraction
        stemmed_tokens = self.preprocess_text(text)
        
        arousal_scores = [self.arousal_dict[w] for w in stemmed_tokens if w in self.arousal_dict]
        if not arousal_scores:
            return 0.5 
        return np.mean(arousal_scores)

    def validate_valence_model(self):
        print("\n--- Validating NLP Valence Model on SST-2 Ground Truth ---")
        try:
            dataset = load_dataset("sst2", split="validation")
            texts = list(dataset['sentence'])
            true_labels = list(dataset['label'])
            
            preds = []
            for t in texts:
                res = self.valence_model(t[:512])[0]
                preds.append(1 if res['label'] == 'POSITIVE' else 0)
                
            f1 = f1_score(true_labels, preds, average='weighted')
            print(f"Weighted F1 Score: {f1:.4f}")
            print(classification_report(true_labels, preds, target_names=['Negative', 'Positive']))
        except Exception as e:
            print(f"Validation failed: {e}")

# ==========================================
# EEG SIGNAL PROCESSING
# ==========================================
class EEGMNEPipeline:
    def process_epoch(self, eeg_array, n_channels, sfreq):
        if eeg_array is None or eeg_array.shape[0] != n_channels:
            return np.nan, np.nan

        try:
            ch_names = [f"E{i}" for i in range(1, n_channels+1)]
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
            raw = mne.io.RawArray(eeg_array, info, verbose=False)
            
            # Preprocessing
            # 4.0 - 40.0 Hz Bandpass Filter, 50Hz Notch Filter
            raw.filter(l_freq=4.0, h_freq=40.0, fir_design='firwin', verbose=False)
            raw.notch_filter(freqs=50.0, verbose=False)

            # ICA Artifact Rejection
            if raw.n_times > 500:
                try:
                    ica = ICA(n_components=5, random_state=42, max_iter=50)
                    ica.fit(raw, verbose=False)
                    ica.exclude = [0]
                    ica.apply(raw, verbose=False)
                except Exception as ica_err:
                    pass

            # PSD for Alpha (8-13 Hz)
            n_fft = min(256, raw.n_times)
            psd_kwargs = dict(fmin=8, fmax=13, tmin=None, tmax=None, n_fft=n_fft)
            spectrum = raw.compute_psd(method='welch', **psd_kwargs, verbose=False)
            psds, freqs = spectrum.get_data(return_freqs=True)
            alpha_power = np.sum(psds, axis=1)

            # Target EGI Channels: Left Frontal (Ch 24), Right Frontal (Ch 124)
            ch24_idx = 23 if 23 < n_channels else 0
            ch124_idx = 104 if 104 < n_channels else n_channels - 1
            
            val_ch124 = alpha_power[ch124_idx]
            val_ch24 = alpha_power[ch24_idx]
            
            faa = np.log(val_ch124 + 1e-9) - np.log(val_ch24 + 1e-9)

            frontal_indices = [idx for idx in [23, 104, 10, 11, 22] if idx < n_channels]
            mean_frontal_alpha = np.mean(alpha_power[frontal_indices])
            alpha_suppression = 1.0 / mean_frontal_alpha if mean_frontal_alpha > 0 else np.nan

            return faa, alpha_suppression

        except Exception as e:
            print(f"EEG error: {e}")
            return np.nan, np.nan

# ==========================================
# STATISTICAL CORRELATION & VISUALIZATION
# ==========================================
def correlate_and_visualize(df):
    print("\n--- Task 1: Subject-Level Z-Scoring ---")
    clean_df = df.dropna(subset=['ai_valence', 'ai_arousal', 'eeg_faa', 'eeg_alpha_suppression']).copy()
    if clean_df.empty:
        print("No valid data points for correlation.")
        return

    scaler = StandardScaler()
    # Apply Z-scoring per subject
    clean_df['eeg_faa_z'] = clean_df.groupby('subject_id')['eeg_faa'].transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())
    clean_df['eeg_alpha_suppression_z'] = clean_df.groupby('subject_id')['eeg_alpha_suppression'].transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())

    print("\n--- Task 2: Individualized Brain Models ---")
    subject_results = []
    
    for subj, group in clean_df.groupby('subject_id'):
        if len(group) < 3: # Need at least 3 points for correlation
            continue
            
        rho_val, p_rho_val = spearmanr(group['ai_valence'], group['eeg_faa_z'])
        r_aro, p_aro = pearsonr(group['ai_arousal'], group['eeg_alpha_suppression_z'])
        
        subject_results.append({
            'subject_id': subj,
            'valence_rho': rho_val,
            'valence_p': p_rho_val,
            'arousal_r': r_aro,
            'arousal_p': p_aro,
            'epochs_used': len(group)
        })
        
    results_df = pd.DataFrame(subject_results)
    
    if not results_df.empty:
        os.makedirs("results", exist_ok=True)
        results_df.to_csv('results/subject_correlations.csv', index=False)
        print(results_df.to_string(index=False))
        
        # Identify Best and Worst aligned subjects based on absolute correlation strength (Valence + Arousal average)
        results_df['overall_alignment'] = results_df[['valence_rho', 'arousal_r']].abs().mean(axis=1)
        best_subj = results_df.loc[results_df['overall_alignment'].idxmax()]
        worst_subj = results_df.loc[results_df['overall_alignment'].idxmin()]
        
        print(f"\nBest Aligned Subject: {best_subj['subject_id']} (Avg Correlation Strength: {best_subj['overall_alignment']:.4f})")
        print(f"Worst Aligned Subject: {worst_subj['subject_id']} (Avg Correlation Strength: {worst_subj['overall_alignment']:.4f})")

    print("\n--- Task 3: Fixing the Cognitive Map ---")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: AI Semantic Understanding
    sns.scatterplot(data=clean_df, x='ai_valence', y='ai_arousal', hue='subject_id', ax=axes[0], alpha=0.7, palette='tab10')
    axes[0].set_title('AI Semantic Understanding')
    axes[0].set_xlabel('AI Valence (DistilBERT Polarity)')
    axes[0].set_ylabel('AI Arousal (Lexicon Mean)')
    axes[0].axhline(0.5, ls='--', color='grey', alpha=0.5)
    axes[0].axvline(0, ls='--', color='grey', alpha=0.5)
    
    # Plot 2: Human Biological Space
    sns.scatterplot(data=clean_df, x='eeg_faa_z', y='eeg_alpha_suppression_z', hue='subject_id', ax=axes[1], alpha=0.7, palette='tab10')
    axes[1].set_title('Human Biological Arousal/Valence')
    axes[1].set_xlabel('EEG FAA (Z-Scored)')
    axes[1].set_ylabel('EEG Alpha Suppression (Z-Scored)')
    axes[1].axhline(0, ls='--', color='grey', alpha=0.5)
    axes[1].axvline(0, ls='--', color='grey', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('results/dual_cognitive_map.png')
    plt.close()
    
    print("Visualizations saved to 'results/dual_cognitive_map.png'.")

    print("\n--- Task 4: Sentence-Level Alignment Analysis ---")
    # Group by sentence to get mean EEG response across all subjects
    sentence_df = clean_df.groupby(['sentence_id', 'text']).agg(
        mean_eeg_faa_z=('eeg_faa_z', 'mean'),
        ai_valence=('ai_valence', 'first')
    ).reset_index()
    
    # Z-score the AI valence so we can multiply them to get an alignment score
    sentence_df['ai_valence_z'] = scaler.fit_transform(sentence_df['ai_valence'].values.reshape(-1, 1)).flatten()
    
    # Positive alignment means AI and EEG agree (both positive or both negative).
    # Negative alignment means they strongly disagree.
    sentence_df['alignment_score'] = sentence_df['ai_valence_z'] * sentence_df['mean_eeg_faa_z']
    
    sentence_df = sentence_df.sort_values(by='alignment_score', ascending=False)
    
    sentence_df.to_csv('results/sentence_alignment.csv', index=False)
    print("Sentence alignment saved to 'results/sentence_alignment.csv'.")
    
# ==========================================
# WORKFLOW
# ==========================================
def main():
    print("Starting Neuro-Linguistic Alignment Pipeline (Real Data)...")
    
    data_loader = ZuCoDataLoader()
    nlp_pipeline = DualStreamNLP()
    eeg_pipeline = EEGMNEPipeline()

    nlp_pipeline.validate_valence_model()

    print("\n--- Loading ZuCo Data ---")
    df = data_loader.load_all_subjects()
    
    # Processing a large subset (5 subjects) to ensure robust cross-subject standard scaling while keeping execution time manageable
    df = df[df['subject_id'].isin(['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM'])]

    
    print(f"\n--- Extracting NLP Features for {len(df)} sentences ---")
    df['ai_valence'] = df['text'].apply(nlp_pipeline.get_valence)
    df['ai_arousal'] = df['text'].apply(nlp_pipeline.get_arousal)
    
    print("\n--- Processing Real EEG Epochs ---")
    faa_list = []
    supp_list = []
    for idx, row in df.iterrows():
        faa, supp = eeg_pipeline.process_epoch(row['eeg_raw'], row['n_channels'], row['sfreq'])
        faa_list.append(faa)
        supp_list.append(supp)
        if idx > 0 and idx % 10 == 0:
            print(f"Processed {idx} EEG epochs...")
            
    df['eeg_faa'] = faa_list
    df['eeg_alpha_suppression'] = supp_list

    correlate_and_visualize(df)
    
    os.makedirs("results", exist_ok=True)
    df.drop(columns=['eeg_raw']).to_csv('results/aligned_features.csv', index=False)
    print("Pipeline execution complete. Aggregated data saved to 'results/aligned_features.csv'.")

if __name__ == "__main__":
    main()
