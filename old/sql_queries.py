# sql_queries.py
import sqlite3
conn = sqlite3.connect('data/pbart_data.db')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(trials)")
columns = cursor.fetchall()
for col in columns:
    print(col[1])  # Print column name
conn.close()

def timestamp_game_state_change():
    if NewGameState != OldGameState:
        switch(NewGameState):
            case("Waitingforresponse"):
                