import pyodbc
import pandas as pd

connStr = (r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};" r"DBQ=filepath.accdb;")
cnxn = pyodbc.connect(connStr) #make sure Access file is closed before running this script
sql = "SELECT * FROM Pubmed_result"
df = pd.read_sql(sql, cnxn)
df.head()