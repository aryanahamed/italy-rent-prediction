# Italy Rent Prediction
This repository aims to analyse and predict rent prices  in various regions of Italy using machine learning algorithms. This is a fun hobby project.

## Installation
To get started with this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/aryanahamed/italy-rent-prediction.git
cd italy-rent-prediction
pip install -r requirements.txt
```
Alternatively, make changes on the notebook directly after cloning.

## Data Collection
All the data is collected from the following database.

https://www.kaggle.com/datasets/tommasoramella/italy-house-prices/data.

## Usage
Check the following page to try out the model.

[Italy Rent Prediction](https://italy-rent-prediction.streamlit.app/)

### Running Locally
To run the Streamlit app locally:

```bash
streamlit run Home.py
```

## Technical Details

### Model
- **Algorithm**: XGBoost Regressor
- **Features**: 21 features including location, size, amenities, and condition
- **Target**: Log-transformed rent price (to handle skewness)

### Confidence Score Calculation
The confidence score (0-100%) is estimated using feature perturbation analysis:
- **High confidence**: Predictions remain stable with small input variations
- **Low confidence**: Predictions vary significantly, indicating uncertainty

### Feature Importance
Feature contributions are calculated using the model's learned importance weights, providing monetary impact for each property attribute:
- Location features are aggregated for clarity
- Both positive and negative impacts are shown
- Top 5 most influential features are displayed


## Sample Plots
[![Heatmap](https://i.ibb.co/Lz79PC8/output.png)](https://ibb.co/kgCqyhx)
[![Rent Price vs Area](https://i.ibb.co/C1wh61T/output2.png)](https://ibb.co/Tvwm1vF)
[![Flats in each city](https://i.ibb.co/3dmjXLs/newplot.png)](https://ibb.co/4tRXy5f)


## Contributing
Contributions are welcome and appreciated!

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
