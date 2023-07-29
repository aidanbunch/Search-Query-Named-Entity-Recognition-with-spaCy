import pandas as pd
import re
import os

current_dir = os.getcwd()
data_dir = os.path.join(current_dir, '..', 'data')

def csv_to_spacy_format(file_path):
    df = pd.read_csv(file_path, keep_default_na=False, dtype=str)
    
    def format_numbers(s):
        if bool(re.search(r'\b\d+\b', s)):
            s = s.replace(",", "").strip()
        return s

    df = df.applymap(format_numbers)
    data = []

    for index, row in df.iterrows():
        text = row['Text']
        entities = []
        
        for entity in df.columns[1:]:
            e_val = str(row[entity]).strip()
            if e_val and e_val.lower() in text.lower():
                e_val_pos = text.lower().find(e_val.lower())
                if text[e_val_pos:e_val_pos+len(e_val)] != e_val:
                    e_val_pos = text.lower().rfind(e_val.lower())
                entities.append((e_val_pos, e_val_pos+len(e_val), entity))
        
        entities = sorted(entities, key=lambda x: x[0])  # sort the entities for overlaps
        final_entities = []
        curr_end = -1

        # Handle the overlaps:
        for ent in entities:
            if ent[0] > curr_end:   # if start of current entity is after end of previous entity, then it is not an overlap
                final_entities.append(ent)
                curr_end = ent[1]

        data.append((text, {"entities": final_entities}))
    return data

def write_to_file(destination_file, training_data):
    contents = f"TRAIN_DATA = {training_data}"
    with open(destination_file, 'w') as f:
        f.write(contents)

if __name__ == "__main__":
    training_data = csv_to_spacy_format("output.csv")
    write_to_file(f'{data_dir}/processed/training_data.py', training_data=training_data)