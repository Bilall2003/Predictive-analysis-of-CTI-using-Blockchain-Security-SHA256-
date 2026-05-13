from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus
import logging

DB_USERNAME= 'root'
DB_PASSWORD= ''
DB_HOST= 'localhost'
DB_NAME= 'FYP'

PASSWORD=quote_plus(DB_PASSWORD)

try:
    ENGINE=create_engine(
    f"mysql+pymysql://{DB_USERNAME}:{PASSWORD}@{DB_HOST}/{DB_NAME}"
    )

    if ENGINE.connect:
        
        logging.info("Connection Built Successfully..........")
    
    else:
        logging.error("Something Went Wrong WHILE **CONNECTING** .Try again")

except SQLAlchemyError as e:
    
    logging.error(e)