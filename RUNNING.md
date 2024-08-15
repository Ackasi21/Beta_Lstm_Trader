# Running the Beta_Lstm

## 0.Colab Link (Easy Public Acess) 
https://colab.research.google.com/drive/1LX4xBbMmEnRGfgmtEwWS4LmpA9OUNpT_#scrollTo=u89pAf3C51MR 




## 1. Prerequisites
- **Python 3.x:** Ensure that you have Python 3.x installed on your system. You can download it [here](https://www.python.org/downloads/).
- **Virtual Environment (Optional):** It’s recommended to use a virtual environment to manage dependencies.
- **Required Libraries:**
  - TensorFlow
  - Keras
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib (if you're visualizing results)

## 2. Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/Ackasi21/Beta_Lstm.git
cd Beta_Lstm

Step 2: Set Up the Virtual Environment (Optional)

python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Step 3: Install the Dependencies

pip install -r requirements.txt

3. Running the Project
Step 1: Prepare the Data

Ensure you have the necessary data files in the correct format. If you're using a dataset from a specific source (like Yahoo Finance), include a script or instructions on how to download and preprocess this data.

Step 2: Run the Model

To train the model:

python train_model.py

To make predictions:

python predict.py --input <input_data.csv> --output <output_results.csv>

Step 3: View the Results
After running the model, results and any visualizations will be saved to the outputs/ directory (or wherever you specify in your scripts).

4. Example Usage

python predict.py --input sample_input.csv --output sample_output.csv

