import os
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import create_engine

def get_connection():
    try:
        load_dotenv()
        username = os.getenv("MYSQL_USER")
        password = quote_plus(os.getenv("MYSQL_PASSWORD"))
        host = os.getenv("MYSQL_HOST", "localhost")
        port = int(os.getenv("MYSQL_PORT", "3306"))
        database = os.getenv("MYSQL_DB")

        print("Username:", username)
        print("Password:", '*' * len(password) if password else None)
        print("Host:", host)
        print("Port:", port)
        print("Database:", database)

        connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        engine = create_engine(connection_string, pool_pre_ping=True)
        # Try opening a connection to test it right away
        with engine.connect() as conn:
            print("✅ Connection to MySQL Database secured successfully.")

        return engine

    except Exception as e:
        print("❌ Failed to create engine:", str(e))
        return None
