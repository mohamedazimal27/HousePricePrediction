# Project Structure

## Directory Organization

```
saudi-housing-prediction/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── style.css                          # Streamlit app styling
│
├── data/
│   └── processed/
│       └── saudi_housing_english.csv  # Main dataset (English translated)
│
├── models/
│   ├── configs/                       # Model configuration files
│   └── saved/                         # Serialized model artifacts
│       ├── presentation_model.pkl     # Trained XGBoost model
│       ├── presentation_scaler.pkl    # Feature scaler
│       ├── presentation_features.pkl  # Feature list
│       └── presentation_encoders.pkl  # Categorical encoders
│
├── outputs/                           # Generated visualizations and reports
│
├── src/                              # Source code modules (currently empty)
│
├── venv/                             # Virtual environment (gitignored)
│
├── app_english.py                    # Streamlit web application
├── train_model_for_presentation.py   # Model training script
├── check_presentation_model.py       # Model validation script
└── test_saudi_pipeline.py           # Comprehensive test suite
```

## File Naming Conventions

### Scripts
- `train_model_*.py`: Model training scripts
- `check_*.py`: Validation and testing scripts  
- `test_*.py`: Test suites
- `app_*.py`: Web application files

### Data Files
- Use descriptive names with language suffix: `*_english.csv`
- Store processed data in `data/processed/`
- Raw data would go in `data/raw/` (not present in this project)

### Model Artifacts
- `*_model.pkl`: Trained models
- `*_scaler.pkl`: Feature scalers
- `*_features.pkl`: Feature lists
- `*_encoders.pkl`: Categorical encoders
- Use consistent prefixes (e.g., `presentation_*`)

## Code Organization Patterns

### Training Scripts
- Load data from `data/processed/`
- Apply consistent preprocessing pipeline
- Save all model artifacts to `models/saved/`
- Generate visualizations to `outputs/`
- Include comprehensive logging and metrics

### Web Applications
- Load models from `models/saved/`
- Use caching decorators (`@st.cache_data`, `@st.cache_resource`)
- Separate styling in external CSS files
- Include data validation and error handling

### Test Files
- Test data integrity, model loading, and prediction pipeline
- Verify file existence and model compatibility
- Include both unit tests and integration tests

## Development Workflow
1. Data preprocessing → `data/processed/`
2. Model training → `models/saved/`
3. Model validation → console output + `outputs/`
4. Web app development → `app_*.py`
5. Testing → `test_*.py`

## Best Practices
- Keep root directory clean with only essential files
- Use descriptive file names that indicate purpose
- Maintain consistent naming across related files
- Store all model artifacts together for easy deployment
- Include comprehensive documentation in README.md