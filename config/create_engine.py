from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus
import logging

DB_USERNAME= 'root'
DB_PASSWORD= ''
DB_HOST= 'localhost'
DB_NAME= 'FYP'

print("Database/SQL Credentials loaded successfully.")

PASSWORD=quote_plus(DB_PASSWORD)

print('Password successfully encoded.')

try:
    ENGINE=create_engine(
    f"mysql+pymysql://{DB_USERNAME}:{PASSWORD}@{DB_HOST}/{DB_NAME}"
    )

    if ENGINE.connect:
        print('Database connected successfully.')
        
        logging.info("Connection Built Successfully..........")
    
    else:
        print('Connection Failed.')

        logging.error("Something Went Wrong WHILE **CONNECTING** .Try again")

except SQLAlchemyError as e:
    
    print('SQLAlchemy error occured.')
    
    logging.error(e)