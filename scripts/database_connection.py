from sqlalchemy import create_engine

def get_connection():
    # Database credentials (replace with your own)
    try:

        username = "root"
        password = "alborz1382"
        host = "localhost"   
        port = "3006"
        database = "SER_DB"

        connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        engine = create_engine(connection_string)

        print("connection to mysql Database has been secured succesfully.")
    except Exception as e:
        print(e.message)

    return engine 

