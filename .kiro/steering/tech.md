# Technology Stack

## Core Technologies
- **Python 3.8+**: Primary programming language
- **XGBoost**: Machine learning model for regression
- **Streamlit**: Web application framework
- **scikit-learn**: Data preprocessing and model evaluation
- **pandas/numpy**: Data manipulation and numerical computing

## Key Libraries
```
streamlit>=1.15          # Web app framework
pandas>=1.3              # Data handling
numpy>=1.21              # Numerical computing
scikit-learn>=1.0        # ML preprocessing and metrics
xgboost>=1.5             # Gradient boosting model
joblib>=1.0              # Model serialization
matplotlib>=3.4          # Plotting
seaborn>=0.11            # Statistical visualization
```

## Project Structure
- **Data**: CSV files in `data/processed/`
- **Models**: Serialized models in `models/saved/`
- **Training**: Standalone Python scripts
- **Web App**: Streamlit application with CSS styling
- **Testing**: Comprehensive test suite

## Common Commands

### Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Model Training
```bash
# Train the presentation model (recommended)
python train_model_for_presentation.py

# Check model accuracy
python check_presentation_model.py
```

### Testing
```bash
# Run comprehensive test suite
python test_saudi_pipeline.py
```

### Web Application
```bash
# Launch Streamlit app
streamlit run app_english.py

# Access at: http://localhost:8501
```

## Development Guidelines
- Use virtual environments for dependency isolation
- Follow scikit-learn patterns for preprocessing
- Serialize models with joblib for consistency
- Include comprehensive error handling
- Use English translations for user-facing content