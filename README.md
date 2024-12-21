# Capuchin Bird Call Detection 🐦🎶

This project focuses on detecting Capuchin bird calls in audio recordings using advanced audio processing and machine learning techniques. The goal is to classify audio clips into two categories: 
- **1**: Capuchin bird call is present.
- **0**: Capuchin bird call is absent.

## Features ✨
- **Preprocessing:** Handles raw audio data, removes noise, and normalizes signals.
- **Machine Learning:** Implements deep learning techniques to classify bird calls.
- **Visualization:** Displays confusion matrices and model performance metrics.
- **Export Results:** Generates CSV files with predictions for easy interpretation.

## Dataset 📂
The dataset used for this project includes:
1. Forest recordings with mixed audio.
2. Parsed audio clips containing Capuchin bird calls.
3. Parsed non-Capuchin bird clips.

**Audio Format:** `.wav`

## Technologies Used 🛠️
- **Programming Language:** Python
- **Libraries:** TensorFlow, NumPy, Librosa, Matplotlib, Pandas
- **Deep Learning Model:** CNN-based architecture
- **Other Tools:** Grad-CAM for visualization

## How to Use 🚀
1. Clone this repository:
   ```bash
   git clone https://github.com/sukritraj02/Capuchin_Bird_Call_Detection.git
   cd Capuchin_Bird_Call_Detection
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset in the `data` folder with the following structure:
   ```
   data/
   ├── Parsed_Capuchinbird_Clips
   ├── Parsed_Non_Capuchinbird_Clips
   ├── Forest_Recordings
   ```
4. Run the preprocessing script:
   ```bash
   python preprocess.py
   ```
5. Train the model:
   ```bash
   python train_model.py
   ```
6. Evaluate the model:
   ```bash
   python evaluate_model.py
   ```
7. Export predictions:
   ```bash
   python export_results.py
   ```

## Results 📊
- **Accuracy:** Achieved XX% accuracy on the test dataset.
- **Confusion Matrix:** Displays true positive and false positive rates for classification.
- **CSV Output:** Generated predictions with timestamps for bird calls.

## Contributions 💖
Thanks to the following amazing contributors for their efforts in building this project:

- [@Sukrit](https://github.com/sukritraj02) 🐦 — Lead developer .
- [@Apoorva](https://github.com/stays1lly) 🛠️ — Assisted in dataset preprocessing and audio visualization.
- [@Anshita](https://github.com/Anshita121004) 🔬 — Worked on model training and result analysis.



Feel free to contribute by submitting issues or pull requests to improve the project further!
