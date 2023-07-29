import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from data.processed.training_data import TRAIN_DATA
import random
from pathlib import Path
import os

current_dir = os.getcwd()
model_dir = os.path.join(current_dir, 'models')

def train_ner_model(train_data, model=None, output_dir=None, n_iter=100, patience=5):
    random.shuffle(train_data)

    # Split the data for training and validation
    # 80% for training, 20% for validation
    split_point = int(0.8*len(train_data))
    training_data = train_data[:split_point]
    validation_data = train_data[split_point:]

    if model is not None:
        nlp = spacy.load(model)
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")

    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner', last=True)
    else:
        ner = nlp.get_pipe('ner')

    for _, annotations in train_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    train_examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in training_data]
    valid_examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in validation_data]
    
    nlp.initialize(lambda: train_examples)

    best_loss = float("inf")
    patience_count = 0

    with nlp.select_pipes(enable="ner"):
        sizes = compounding(1.0, 4.0, 1.001)
        
        for itn in range(n_iter):
            
            random.shuffle(train_examples)
            batches = minibatch(train_examples, size=sizes)
            losses = {}

            for batch in batches:
                nlp.update(batch, sgd=None, losses=losses, drop=0.5)

            print(f"Iteration {itn}, Losses {losses}")

            # Validate against the validation examples
            val_losses = {}
            for example in valid_examples:
                nlp.update([example], sgd=None, losses=val_losses, drop=0.0)

            if val_losses['ner'] < best_loss:
                best_loss = val_losses['ner']
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= patience:
                break

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

    return nlp

def train_ner_model_without_early_stopping(train_data, model=None, output_dir=None, n_iter=100):

    if model is not None:
        nlp = spacy.load(model)
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en") 
        print("Created blank 'en' model")

    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner', last=True)
    else:
        ner = nlp.get_pipe('ner')

    for _, annotations in train_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])      

    examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in train_data]
    nlp.initialize(lambda: examples)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        for itn in range(n_iter):
            random.shuffle(examples)
            losses = {}
            for batch in spacy.util.minibatch(examples, size=2):
                nlp.update(
                    batch,  # pass a batch of Example objects
                    drop=0.5,  
                    losses=losses,
                )
                print('Losses', losses)

        # Save model 
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            if model is None:
                nlp.meta['name'] = 'custom_ner_model'
            else:
                nlp.meta['name'] = model
            nlp.to_disk(output_dir)
            print("Saved model to", output_dir)

if __name__ == "__main__":
    train_ner_model(TRAIN_DATA, output_dir=f"{model_dir}/ner_model_v3")