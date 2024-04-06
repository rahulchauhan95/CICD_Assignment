# Use a minimal Python image
FROM python:3.11

# Set the working directory
WORKDIR /code

# Copy all the data to the working directory
COPY . /code/.

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Train the model during build
RUN python train.py

# Set the entrypoint to execute test.py when the container runs
ENTRYPOINT ["python", "test.py"]