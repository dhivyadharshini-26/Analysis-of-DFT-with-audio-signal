# EXP 1 :  ANALYSIS OF DFT WITH AUDIO SIGNAL

# AIM: 

  To analyze audio signal by removing unwanted frequency. 

# APPARATUS REQUIRED: 
   
   PC installed with SCILAB/Python. 

# PROGRAM: 

// analyze audio signal
```
# ==============================
# AUDIO DFT ANALYSIS IN COLAB
# ==============================

# Step 1: Install required packages
!pip install -q librosa soundfile

# Step 2: Upload audio file
from google.colab import files
uploaded = files.upload()   # choose your .wav / .mp3 / .flac file
filename = next(iter(uploaded.keys()))
print("Uploaded:", filename)

# Step 3: Load audio
import librosa, librosa.display
import numpy as np
import soundfile as sf

y, sr = librosa.load(filename, sr=None, mono=True)  # keep original sample rate
duration = len(y) / sr
print(f"Sample rate = {sr} Hz, duration = {duration:.2f} s, samples = {len(y)}")

# Step 4: Play audio
from IPython.display import Audio, display
display(Audio(y, rate=sr))

# Step 5: Full FFT (DFT) analysis
import matplotlib.pyplot as plt

n_fft = 2**14   # choose large power of 2 for smoother spectrum
Y = np.fft.rfft(y, n=n_fft)
freqs = np.fft.rfftfreq(n_fft, 1/sr)
magnitude = np.abs(Y)

plt.figure(figsize=(12,4))
plt.plot(freqs, magnitude)
plt.xlim(0, sr/2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Magnitude Spectrum (linear scale)")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,4))
plt.semilogy(freqs, magnitude+1e-12)
plt.xlim(0, sr/2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (log scale)")
plt.title("FFT Magnitude Spectrum (log scale)")
plt.grid(True)
plt.show()

# Step 6: Top 10 dominant frequencies
N = 10
idx = np.argsort(magnitude)[-N:][::-1]
print("\nTop 10 Dominant Frequencies:")
for i, k in enumerate(idx):
    print(f"{i+1:2d}. {freqs[k]:8.2f} Hz  (Magnitude = {magnitude[k]:.2e})")

# Step 7: Spectrogram (STFT)
n_fft = 2048
hop_length = n_fft // 4
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(12,5))
librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='hz')
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")
plt.ylim(0, sr/2)
plt.show()
```

# OUTPUT: 


Top 10 Dominant Frequencies:
 1.    69.98 Hz  (Magnitude = 6.79e+00)
 2.   282.62 Hz  (Magnitude = 5.87e+00)
 3.   258.40 Hz  (Magnitude = 5.65e+00)
 4.    67.29 Hz  (Magnitude = 5.61e+00)
 5.   301.46 Hz  (Magnitude = 5.03e+00)
 6.   261.09 Hz  (Magnitude = 4.82e+00)
 7.   279.93 Hz  (Magnitude = 4.69e+00)
 8.   298.77 Hz  (Magnitude = 4.41e+00)
 9.   253.02 Hz  (Magnitude = 4.36e+00)
10.   304.16 Hz  (Magnitude = 4.30e+00)

<img width="1236" height="495" alt="image" src="https://github.com/user-attachments/assets/6dd1a503-b26b-4779-ab17-08d69e9afec9" />

<img width="1272" height="489" alt="image" src="https://github.com/user-attachments/assets/9c0cb4df-3d27-46dc-bd9d-3245cc75891a" />

<img width="1198" height="582" alt="image" src="https://github.com/user-attachments/assets/60d0c432-5af3-4a7c-80cf-93c45097e744" />







# RESULTS:

Thus,the analysis of DFT with audio signal is verified
