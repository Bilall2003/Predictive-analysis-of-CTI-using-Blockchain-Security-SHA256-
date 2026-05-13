file_list = ["Monday-WorkingHours.csv", "Tuesday-WorkingHours.csv", "Wednesday-WorkingHours.csv", "Thursday-WorkingHours.csv"]

# this is a flag
# controls append or replace
# if true then the first chunk replace table
# else later chunks append data
is_first_upload = True

# loop that takes one, processes it and then moves to the other one
for file in file_list:
    print(f"Starting file: {file}")

    # inside the read_csv using chunksize to stream file from disk
    # only read 50000 at a time
    reader = pd.read_csv(file, chunksize=50000, low_memory=False)
    
    # now process 50k rows at a time piece by piece
    for chunk in reader:
        # this part remove infinity values and replaces them with missing/nan values because sql can't store infinity values
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        #clean columns, remove spaces and convert to lowercase
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        
        # Determine mode: if first time then replace table and if not then after that replace tables
        mode = 'replace' if is_first_upload else 'append'
        
        # his part sends the chunk to mysql table called raw_sentiment_data
        chunk.to_sql(name='raw_sentiment_data', con=ENGINE, if_exists=mode, index=False)
        
        # now the first chunk is uploaded everything(data) is then appended/insert
        is_first_upload = False
        
    print(f"Finished uploading all chunks of {file}")

print("\n--- ALL 1.4M ROWS SUCCESSFULLY STREAMED TO MYSQL ---")