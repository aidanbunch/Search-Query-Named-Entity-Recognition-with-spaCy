import csv
import pandas as pd
import os

current_dir = os.getcwd()
data_dir = os.path.join(current_dir, '..', 'data')

with open(f'{data_dir}/raw/input.csv', 'r') as infile:
    lines = infile.readlines()

num_columns = len(lines[0].split(','))

# Remove extra commas from each line
lines = [line.rstrip().split(',') for line in lines]
for i, line in enumerate(lines):
    while len(line) > num_columns:
        line[-2:] = [','.join(line[-2:])]
    lines[i] = line

# Add extra commas to rows with less than the total amount of commas
for i, line in enumerate(lines):
    while len(line) < num_columns:
        line.append('')
    lines[i] = line

with open(f'{data_dir}/interim/formatted_for_cleaning.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(lines)

df = pd.read_csv(f'{data_dir}/interim/formatted_for_cleaning.csv')

df.drop_duplicates(inplace=True)

column_names = df.columns.to_list()

second_column = column_names[1]
third_column = column_names[2]
fourth_column = column_names[3]

# Check second column and append contents to the first column if it has any digits, then remove the entire string from the second column
df[second_column] = df.apply(lambda row: str(row[second_column]) + ' ' + str(row[third_column]) if any(char.isdigit() for char in str(row[second_column])) else str(row[second_column]), axis=1)
df[second_column] = df.apply(lambda row: '' if any(char.isdigit() for char in str(row[second_column])) else str(row[second_column]), axis=1)
df[second_column] = df[second_column].replace('nan', '', regex=True)

# Remove values from the third column that contain non-digit characters
df[third_column] = df[third_column].apply(lambda x: ''.join(filter(str.isdigit, str(x))) if isinstance(x, str) else x)
df[fourth_column] = df[fourth_column].apply(lambda x: ''.join(filter(str.isdigit, str(x))) if isinstance(x, str) else x)

df.to_csv(f'{data_dir}/interim/cleaned_input.csv', index=False)