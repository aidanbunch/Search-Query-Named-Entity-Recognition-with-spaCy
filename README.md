# Search Query Named Entity Recognition with spaCy
This repository contains a project for extracting search filters from natural language queries using custom named entity recognition with spaCy and machine learning. I trained a custom NER model to extract these entities using a dataset saved in CSV format, which is cleaned, converted to the spaCy tuple format in preprocessing and used to train/validate the model.

## Dataset
Put your CSV input data into the data/raw directory. Make sure the first column is titled "Text" and contains the full string, while every subsequent column represents an entity name and contains the extracted entity values. Something like the following schema should work:

<img width="1235" alt="Screenshot 2023-07-29 at 12 05 45 PM" src="https://github.com/aidanbunch/Search-Query-Named-Entity-Recognition-with-spaCy/assets/44245721/99baa431-f280-425f-8c13-64868446fdef">

## Usage
1. Open a terminal or command prompt and navigate to the project directory.
2. Create a virtual environment by running the following command:
```
python -m venv venv
```
3. Activate the virtual environment:
For Windows:
```
venv\Scripts\activate
```
For macOS/Linux:
```
source venv/bin/activate
```
4. Install the required dependencies by running the following command:
```
pip install -r requirements.txt
```
5. Clean the data by running the following command:
```
cd cleaning && python clean_data.py
```
6. Prepare the data for training by running the following command:
```
cd .. && cd preprocessing && python prepare_data.py
```
7. Train the NER model by running the following command:
```
cd .. && python train_ner.py
```
<img width="699" alt="Screenshot 2023-07-28 at 7 41 32 PM" src="https://github.com/aidanbunch/Search-Query-Named-Entity-Recognition-with-spaCy/assets/44245721/b0a74339-8368-411a-ae3a-68d1e4366030">

8. After the model has been generated, you can test it by running the following command:
```
cd testing && python test_ner.py
```
<img width="443" alt="Screenshot 2023-07-28 at 7 44 15 PM" src="https://github.com/aidanbunch/Search-Query-Named-Entity-Recognition-with-spaCy/assets/44245721/8e68bc0e-9595-4614-ae13-7bca09a1375d">
