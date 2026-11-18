
import logging
import os
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DB_USERNAME = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_HOST = os.environ.get(
    "DB_HOST",
    "host.docker.internal" if os.environ.get("DOCKER_CONTAINER") else "localhost"
)
DB_PORT = os.environ.get("DB_PORT", "3306")
DB_NAME = os.environ.get("DB_NAME", "ledger_logs")

PASSWORD = quote_plus(DB_PASSWORD)

try:
    ENGINE = create_engine(
        f"mysql+pymysql://{DB_USERNAME}:{PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        pool_pre_ping=True,
    )

    with ENGINE.connect() as conn:
        print(f"Connecting to DB: {DB_HOST}:{DB_PORT}/{DB_NAME}")
        logging.info("Connection Built Successfully")

except SQLAlchemyError as e:
    logging.error(f"SQLAlchemy error: {e}")
    raise
