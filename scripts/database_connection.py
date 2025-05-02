import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

def get_connection():
    # Database credentials (replace with your own)
    try:
        load_dotenv()

        username = os.getenv("MYSQL_USER")
        password = os.getenv("MYSQL_PASSWORD")
        host = os.getenv("MYSQL_HOST", "localhost")
        port = os.getenv("MYSQL_PORT", "3306")
        database = os.getenv("MYSQL_DB")

        connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        engine = create_engine(connection_string)

        print("connection to mysql Database has been secured succesfully.")
    except Exception as e:
        print(e.message)

    return engine 
