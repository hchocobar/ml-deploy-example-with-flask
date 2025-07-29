from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd


load_dotenv()  # load the .env file variables


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine
