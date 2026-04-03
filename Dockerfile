# 1. Base Image: Use a lightweight, official Python environment
FROM python:3.10-slim

# 2. System Setup: Create a working directory inside the container
WORKDIR /app

# 3. Dependencies: Copy the requirements list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Application Code: Copy your entire project into the container
COPY . .

# 5. Network: Expose port 5000 so the outside world can see the UI
EXPOSE 5000

# 6. Boot Command: Start the Flask Web Server when the container turns on
CMD ["python", "dashboard/dashboard_server.py"]