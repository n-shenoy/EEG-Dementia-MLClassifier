import os
import glob
import numpy as np
import pandas as pd
import mne
import scipy
import seaborn as sns
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler, LabelEncoder


def calculate_band_power(raw):
    """Calculates average power for standard frequency bands for all channels."""
    # Compute Power Spectral Density (PSD) using Welch's method
    # fmin and fmax restrict the computation to our bands of interest
    spectrum = raw.compute_psd(method='welch', fmin=1.0, fmax=45.0, n_fft=2048, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)

    # Handle dimensions: PSDs shape can be (channels, freqs) 
    # or (epochs, channels, freqs) 
    if psds.ndim == 3:
        # average across epochs (the first dimension)
        psds = np.mean(psds, axis = 0)

    # get channel names    
    ch_names = raw.info['ch_names']
    features = {}
    
    # Iterate through each channel
    for i, ch in enumerate(ch_names):
        # Iterate through each frequency band
        for band_name, (fmin, fmax) in freq_bands.items():
            # Find the indices of frequencies that fall within the current band
            freq_idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
            
            # Extract the PSD values for those frequencies and calculate the mean power
            # check if freq_idx is empty
            if len(freq_idx) > 0:
                # calculate mean power
                mean_power = np.mean(psds[i, freq_idx])
                # use log10 to normalize the highly skewed power values
                mean_power_log = np.log10(mean_power + 1e-10)
            else:
                mean_power_log = 0
            
            # Create a column name like 'Fp1_Alpha'
            col_name = f"{ch}_{band_name}"
            features[col_name] = mean_power_log
            
    return features

def load_any_eeg(filepath):
    """
    Tries to load a .set file as Raw, and if that fails, 
    tries loading it as Epochs (already divided into time series).
    """
    if filepath.endswith('.set'):
        try:
            # First, try loading as continuous data
            return mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
        except:
            # If Raw fails, it's likely Epoched data
            # We use read_epochs_eeglab instead
            return mne.read_epochs_eeglab(filepath, verbose=True)
    elif filepath.endswith('.edf'):
        return mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    


def drop_highly_correlated_features(df, features, threshold=0.9):
    """Removes features that highly correlate with another feature
        based on a threshold. (using spearman correlation)
    """
    corr_matrix = df[features].corr(method='spearman').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    return [f for f in features if f not in to_drop], to_drop


def run_pca(X):
    """ PCA helper function """
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca.explained_variance_ratio_

    
