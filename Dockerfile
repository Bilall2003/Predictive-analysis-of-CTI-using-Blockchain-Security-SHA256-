# Recipe for the dashboard image
FROM python:3.12-slim

# xgboost needs this small system library
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) PyTorch (CPU version = much smaller download). Put YOUR torch version here, e.g. torch==2.3.1
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2) All the other libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Your code and model files
COPY . .

EXPOSE 8501

# 4) Start the dashboard (change 3mileui.py if your main file has another name)
CMD ["streamlit", "run", "3mileui.py", "--server.port=8501", "--server.address=0.0.0.0"]
