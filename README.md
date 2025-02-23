# Sign Language Recognition

This repository contains a sign language recognition system designed for ASL users to collect data, train a model, and make real-time predictions. The system supports custom gesture recording, model training, and inference using a webcam.

## Features
- **Data Collection**: Record sign language gestures using a webcam.
- **Model Training**: Train a convolutional neural network (CNN) on collected sign data.
- **Real-Time Prediction**: Recognize gestures in real time using a trained model.
- **Cross-Platform Compatibility**: Supports macOS, Windows, and Linux.
- **Dynamic Labeling**: Automatically detects labels based on collected data.

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed on your system.

### Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/loyolaaiclub/Sign-Language-Recognition.git
cd Sign-Language-Recognition
pip install -r requirements.txt
```

## Usage

### 1. Collecting Data
Run the data collection script to record a gesture:

```bash
python user_train.py --label hello --samples 30 --duration 2
```

Options:
- `--label` (required): Name of the gesture (e.g., "hello", "bye").
- `--samples` (optional, default=30): Number of recordings to collect.
- `--duration` (optional, default=2): Duration (in seconds) per recording.

On macOS, use the macOS-specific script:

```bash
python user_train_macos.py --label hello --samples 30 --duration 2
```

### 2. Training the Model
Train the model using collected data:

```bash
python model_training.py
```

This script will:
1. Load the recorded gesture data from the `data/` directory.
2. Train a CNN model.
3. Save the trained model as `collected_data_model.h5`.

### 3. Running Real-Time Prediction
Run the model to recognize gestures in real time:

```bash
python app.py
```

Press `q` to exit the application.

### 4. Viewing Predictions
Predictions are stored in `predictions.txt` and will update in real time.

## Troubleshooting
- **Camera Not Detected**:
  - Ensure your camera is connected and permissions are granted.
  - Restart the application.
- **Low Prediction Accuracy**:
  - Collect more training data for better performance.
  - Ensure consistent lighting and background conditions.
- **Application Freezes**:
  - Restart the script and ensure other apps are not using the webcam.

## Contribution
Pull requests and contributions are welcome! If you have ideas for improving the system, please open an issue.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

#
