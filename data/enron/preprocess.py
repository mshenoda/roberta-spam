import pandas as pd

# Read the CSV file
df = pd.read_csv('enron_spam_data.csv')

# Filter the desired columns
df_filtered = df[['Spam/Ham', 'Message']]

# Rename the column headers
df_filtered.rename(columns={'Spam/Ham': 'label', 'Message': 'text'}, inplace=True)

# Drop rows with empty message values
df_filtered.dropna(subset=['text'], inplace=True)

# Convert cells to a single line
df_filtered['text'] = df_filtered['text'].apply(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)

# Save the filtered and modified DataFrame to a new CSV file
df_filtered.to_csv('enron_spam.csv', index=False)
