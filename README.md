# 🌿 AI Plant Disease Detection System

A comprehensive plant disease detection system using deep learning with real-time camera diagnosis, scientific analysis, and batch testing capabilities.

![Plant Disease Detection Demo](docs/images/demo.gif)

## ✨ Features

- 🎥 **Live Camera Diagnosis** - Real-time plant disease detection
- 🔬 **Scientific Analysis** - Detailed diagnostic reports
- 📊 **Batch Testing** - Test multiple images and generate reports  
- 🤖 **Multi-Model Support** - Binary and multi-class classification
- 📈 **Performance Analytics** - Confidence scores and accuracy metrics
- 🎯 **Treatment Recommendations** - AI-powered care suggestions

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Test with sample image
python scripts/test_model.py data/sample_images/diseased_sample.jpg

# Live camera diagnosis
python scripts/live_diagnosis.py

# Scientific analysis with detailed report
python scripts/scientific_diagnosis.py data/sample_images/diseased_sample.jpg
```

## 📋 System Requirements

- Python 3.9+
- OpenCV for camera functionality
- PyTorch with CUDA support (recommended for training)
- 4GB RAM minimum, 8GB+ recommended
- Webcam for live diagnosis features

## 🔧 Advanced Usage

### Training Your Own Model

```bash
# Organize your dataset
python scripts/organize_dataset.py

# Train the model
python scripts/train_model.py --data_dir ./organized_dataset --epochs 10

# Batch testing
python scripts/batch_test.py --test_dir ./test_images
```

### Model Performance

| Model Type | Accuracy | Inference Time | Classes |
|------------|----------|----------------|---------|
| Binary     | 94.2%    | 0.15s         | 2       |
| Multi-class| 87.8%    | 0.18s         | 38      |

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Usage Instructions](docs/usage.md)
- [API Reference](docs/api_reference.md)
- [Examples](docs/examples/)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  Acknowledgments

- PlantVillage Dataset
- Hugging Face Transformers
- OpenCV Community


---

⭐ If this project helped you, please give it a star!


