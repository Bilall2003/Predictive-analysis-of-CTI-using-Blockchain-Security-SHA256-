from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus
import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DB_USERNAME = 'root'
DB_PASSWORD = 'hanzala@1234'
DB_HOST = 'localhost'
DB_NAME = 'cti_fyp_2'

PASSWORD = quote_plus(DB_PASSWORD)

try:
    ENGINE = create_engine(
        f"mysql+pymysql://{DB_USERNAME}:{PASSWORD}@{DB_HOST}/{DB_NAME}"
    )

    # REAL connection test
    with ENGINE.connect() as conn:
        print(f"Connecting to DB: {DB_HOST}/{DB_NAME}")
        logging.info("Connection Built Successfully")

except SQLAlchemyError as e:
    logging.error(f"SQLAlchemy error: {e}")


    