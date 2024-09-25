
import pickle
def save_embeddings(embeddings_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(embeddings_dict, f)


def load_embeddings(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    



    
    
    # Get all the unique "target_language-language" pairs as columns
    columns = []
    for target_language in embeddings_dict:
        for language in embeddings_dict[target_language]:
            column_name = f"{target_language}-{language}"
            columns.append(column_name)
    
    # Assuming all embeddings have the same index length
    index_length = len(next(iter(next(iter(embeddings_dict.values())).values())))
    # Prepare the data to be written to CSV
    data = []
    for i in range(index_length):
        row = []
        for target_language in embeddings_dict:
            for language in embeddings_dict[target_language]:
                embedding = embeddings_dict[target_language][language][i]
                row.append(embedding)
        data.append(row)

    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write the header (columns)
        csvwriter.writerow(columns)
        
        # Write the data (rows)
        csvwriter.writerows(data)

    print("Embedding saved!")