import spacy
import os

current_dir = os.getcwd()
models_dir = os.path.join(current_dir, '..', 'models')

nlp = spacy.load(f'{models_dir}/ner_model_v3')
sample_queries = [
    "Office spaces with a cap rate under 5 in downtown",
    "Homes with an acreage between 1 and 5 acres in the suburbs",
    "Retail spaces with a minimum square ft of 2000 in the city center",
    "Apartments with an asking price below $1 million in the waterfront area",
    "Warehouses with a cap rate higher than 7 in industrial zones",
    "Commercial properties with an occupancy rate of at least 90 in the business district",
    "Land for sale in opportunity zones",
    "Looking for a commercial property with cap rate above 6%",
    "Hotels with a minimum term of 10 years",
    "Industrial properties with a maximum asking price of $5 million",
    "Looking for a car wash with acreage between 5 and 10 acres"
]
for index, query in enumerate(sample_queries):
    doc = nlp(query)
    print(f"Query #{index + 1}")
    for ent in doc.ents:
        print(f"{ent.label_.upper():{15}}- {ent.text}")
    print(f"\n")