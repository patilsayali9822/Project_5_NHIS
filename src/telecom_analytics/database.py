import os
import pandas as pd
from sqlalchemy import create_engine

def export_mysql(df, table_name="customer_satisfaction"):
    url = os.getenv("MYSQL_URL")
    if not url:
        return {"status": "skipped", "message": "MYSQL_URL is not set."}
    engine = create_engine(url)
    df.to_sql(table_name, engine, if_exists="replace", index=False, chunksize=5000)
    return {"status": "success", "message": f"Exported {len(df)} rows to {table_name}."}
