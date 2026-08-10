import sqlite3

conn = sqlite3.connect("clinical_assistant.db")
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", cursor.fetchall())

print("\n--- Consultations ---")
cursor.execute("SELECT * FROM consultations")
for row in cursor.fetchall():
    print(row)

print("\n--- Transcript Segments ---")
cursor.execute("SELECT * FROM transcript_segments")
for row in cursor.fetchall():
    print(row)

print("\n--- SOAP Notes ---")
cursor.execute("SELECT * FROM soap_notes")
for row in cursor.fetchall():
    print(row)

conn.close()
