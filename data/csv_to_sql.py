import sqlite3
from tqdm import tqdm

def main():
    # Create database file if it does not exist
    with open("data/assertions.sqlite", "w") as f:
        pass

    # Create a connection to the database
    conn = sqlite3.connect('data/assertions.sqlite')
    c = conn.cursor()

    # Create the table with the following columns, all with string type: ID, Relation, Vertex1, Vertex2, Info
    c.execute('''CREATE TABLE assertions (ID text, Relation text, Vertex1 text, Vertex2 text, Info text)''')

    with open("data/assertions.csv", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    print("Number of lines:", line_count)

    with open("data/assertions.csv", encoding="utf-8") as f:
        for _ in tqdm(range(line_count)):
            line = f.readline()
            if not line:
                continue
            parts = line.strip().split("\t")
            c.execute("INSERT INTO assertions VALUES (?, ?, ?, ?, ?)", parts)
    
    # Commit the changes and close the connection
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
