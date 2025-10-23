# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Create a startup script in /usr/local/bin (not in /app so it won't be overwritten by volume mount)
RUN echo '#!/bin/bash\n\
cd /app\n\
if [ ! -f "agent1_final.pkl" ]; then\n\
    echo "No trained models found. Training agents..."\n\
    python Qlearning.py\n\
fi\n\
echo "Starting Streamlit app..."\n\
streamlit run streamlit_app.py' > /usr/local/bin/start.sh && chmod +x /usr/local/bin/start.sh

# Run the startup script
CMD ["/usr/local/bin/start.sh"]