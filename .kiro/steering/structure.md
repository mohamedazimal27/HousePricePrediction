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
│       ├── model.pkl                  # Trained XGBoost model
│       ├── scaler.pkl                 # RobustScaler for features
│       ├── features.pkl               # Feature list
│       └── encoders.pkl               # Categorical encoders
│
├── outputs/                           # Generated visualizations and reports
│
├── src/                              # Source code modules (currently empty)
│
├── venv/                             # Virtual environment (gitignored)
│
├── app.py                            # Streamlit web application
├── train_model.py                    # Model training script
├── check_model.py                    # Model validation script
└── test_saudi_pipeline.py           # Comprehensive test suite
```

## File Naming Conventions

### Scripts
- `train_model.py`: Model training script
- `check_model.py`: Model validation script
- `test_*.py`: Test suites
- `app.py`: Web application file

### Data Files
- Use descriptive names with language suffix: `*_english.csv`
- Store processed data in `data/processed/`
- Raw data would go in `data/raw/` (not present in this project)

### Model Artifacts
- `model.pkl`: Trained XGBoost model
- `scaler.pkl`: RobustScaler for feature scaling
- `features.pkl`: Feature list for model input
- `encoders.pkl`: Categorical encoders for preprocessing

## Code Organization Patterns

### Training Scripts
- Load data from `data/processed/`
- Apply consistent preprocessing pipeline
- Save all model artifacts to `models/saved/`
- Generate visualizations to `outputs/`
- Include comprehensive logging and metrics

### Web Application
- Load models from `models/saved/`
- Use caching decorators (`@st.cache_data`, `@st.cache_resource`)
- Separate styling in external CSS files
- Include comprehensive error handling and input validation
- Provide user feedback with confidence scores and warnings

### Test Files
- Test data integrity, model loading, and prediction pipeline
- Verify file existence and model compatibility
- Include both unit tests and integration tests

## Development Workflow
1. Data preprocessing → `data/processed/`
2. Model training → `models/saved/`
3. Model validation → console output + `outputs/`
4. Web app development → `app.py`
5. Testing → `test_*.py`

## Best Practices
- Keep root directory clean with only essential files
- Use descriptive file names that indicate purpose
- Maintain consistent naming across related files
- Store all model artifacts together for easy deployment
- Include comprehensive documentation in README.md