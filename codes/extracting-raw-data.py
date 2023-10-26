import pandas as pd

# 1. Load the Excel file into a DataFrame
file_path = '../reference/ETRI 표준 분류 XYZ 20230719_04_이윤주.xlsx'
# Setting the header to the 3rd row based on your requirement
df = pd.read_excel(file_path, header=2)

# 2. Select the columns and rows of interest
selected_columns = ['적용단계', '스마트 인프라', '인프라분류','상호운용성수준','세부분류키워드' ]
# Selecting rows from 4 to 4757 (1-based indexing to 0-based indexing)
selected_rows = df.loc[1:4756, selected_columns]

# 3. Convert the selected DataFrame to JSON format
json_data = selected_rows.to_json(orient='records', force_ascii=False)

# 4. Save the JSON data to a file
json_file_path = '../data/raw_data.json'
with open(json_file_path, 'w', encoding='utf-8') as f:
    f.write(json_data)
