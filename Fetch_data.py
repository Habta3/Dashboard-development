#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import pandas as pd
import mysql.connector as mysql
from mysql.connector import Error

def DBConnect(dbName="Telecom"):
    conn = mysql.connect(host='localhost', user='root', password="1234",
                         database=dbName, buffered=True)
    cur = conn.cursor()
    return conn, cur



def db_execute_fetch(*args, many=False, tablename='satisfactionData2', rdf=True, **kwargs) -> pd.DataFrame:
    
    connection, cursor1 = DBConnect(**kwargs)
    if many:
        cursor1.executemany(*args)
    else:
        cursor1.execute(*args)

    # get column names
    field_names = [i[0] for i in cursor1.description]

    # get column values
    res = cursor1.fetchall()

    # get row count and show info
    nrow = cursor1.rowcount
    if tablename:
        print(f"{nrow} recrods fetched from {tablename} table")

    cursor1.close()
    connection.close()

    # return result
    if rdf:
        return pd.DataFrame(res, columns=field_names)
    else:
        return res





# In[ ]:




