FROM python:3.9

# Set working directory
WORKDIR /opt/program

# Set environment variables
ENV PATH="/opt/program:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx

# Copy requirements and install Python dependencies
COPY requirements.txt /opt/program/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code to the container
COPY src /opt/program/src
COPY models /opt/ml/model/

# Make the SageMaker handler executable
COPY deployment/sagemaker/serve /opt/program/
RUN chmod +x /opt/program/serve

# Set the entrypoint
ENTRYPOINT ["/opt/program/serve"] 