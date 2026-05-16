#importing libraries
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus
import logging

#loading credentials
DB_USERNAME= 'root'
DB_PASSWORD= ''
DB_HOST= 'localhost'
DB_NAME= 'FYP'

PASSWORD=quote_plus(DB_PASSWORD)

print('Password successfully encoded.')

try:
    # create_engine = an sqlalchemy function that creates a database connection setting up instructions and configuration to connect to mysql
    # mysql+pymysql:// = tells sqlalchemy to use mysql and use pymysql driver to connect
    ENGINE=create_engine(
    f"mysql+pymysql://{DB_USERNAME}:{PASSWORD}@{DB_HOST}/{DB_NAME}"
    )

    # checks if the connection is a success or not
    with ENGINE.connect() as conn:
        
        logging.info("Connection Built Successfully..........")
    
except SQLAlchemyError as e:
    print("SQLAlchemy error occurred")
    logging.error(e)