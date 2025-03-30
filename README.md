# Description
This project is designed for processing and classifying electrocardiogram (ECG) data, supporting both time-series and image-based data processing methods.

# Project Directory Structure

```
├── data
│   ├── ecg_data
│   │   ├── pdf # Raw ECG PDF files
│   │   ├── datasets # Time-series ECG datasets
│   │   ├── image_datasets # Image-based ECG datasets
│
├── utils
│   ├── Utility functions and helper scripts
│
├── ecgtizer
│   ├── Tool for digitizing ECG PDFs
│
├── models
│   ├── DNN Models
│   │   ├── cnn_lstm_transformer.py        # Time-series based ECG classification model using CNN, LSTM, and Attention
│   │   ├── cnn_lstm_transformer_image.py  # Image-based ECG classification model using CNN, LSTM, and Attention
│
├── ecg_time.py           # PyTorch implementation of time-series ECG classification
├── ecg_image.py          # PyTorch implementation of image-based ECG classification
├── ecg_data_digitizer.py # Generates time-series ECG datasets
├── ecg_image_data.py     # Generates image-based ECG datasets
```

## Directory Description
- **data/**: Stores raw ECG PDF files and processed datasets.
- **utils/**: Contains utility functions and helper scripts.
- **ecgtizer/**: A tool for digitizing ECG PDFs.
- **models/**: Directory for Deep Neural Network (DNN) models.
  - `cnn_lstm_transformer.py`: A time-series ECG classification model utilizing CNN, LSTM, and Attention mechanisms.
  - `cnn_lstm_transformer_image.py`: An image-based ECG classification model utilizing CNN, LSTM, and Attention mechanisms.
- **ecg_time.py**: PyTorch implementation of time-series ECG classification.
- **ecg_image.py**: PyTorch implementation of image-based ECG classification.
- **ecg_data_digitizer.py**: Generates time-series ECG datasets.
- **ecg_image_data.py**: Generates image-based ECG datasets.

# Dependencies
Ensure the following dependencies are installed:

python 3.11

torch >= 2.6

opencv >= 4.9

numpy >= 1.14.3

pandas >= 2.2.2

matplotlib >= 3.2

# Usage
## Data Processing
Run ecg_data_digitizer.py to generate time-series ECG datasets.

Run ecg_image_data.py to generate image-based ECG datasets.

## Model Training
Run ecg_time.py to train the time-series-based classification model.

Run ecg_image.py to train the image-based classification model.



