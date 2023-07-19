import pandas as pd

df_csv_append = pd.DataFrame()

# append the CSV files
def attachCSVs(fcsv, scsv):
    df_csv_append = pd.DataFrame()
    csvOne = pd.read_csv(fcsv)
    csvTwo = pd.read_csv(scsv)
    df_csv_append = df_csv_append._append(csvOne, ignore_index=True)
    df_csv_append = df_csv_append._append(csvTwo, ignore_index=True)

    pd.DataFrame.to_csv(df_csv_append, 'fullFeatures_extra_col.csv')

def split_csv_by_feature(input_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Group the DataFrame by the 'feature' column
    grouped = df.groupby('Half an hour')

    # Split the data into separate DataFrames based on the 'feature' values
    dataframes = {group: data for group, data in grouped}
    print(dataframes)
    # Save each DataFrame to a separate CSV file
    for group, data in dataframes.items():
        output_file = f'output_feature_{group}.csv'
        data.to_csv(output_file, index=False)

def dropfirstCol(data):
    df = pd.read_csv(data)
    df = df.iloc[:, 1:]
    # Save the modified DataFrame to a new CSV file
    df.to_csv('full_features.csv', index=False)

if __name__ == "__main__":
    # split_csv_by_feature('fullFeatures_extra_col.csv')
    # attachCSVs('csv1.csv', 'features.csv')
    dropfirstCol('fullFeatures_extra_col.csv')