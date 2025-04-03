# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /ContainerWD

# Copy the requirements text file into the container at /app
COPY ./requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
#   This is done after because Docker caches the steps and performs them in order
#   So if requirements.txt doesn't change, it won't re-install the packages
#   every time you build the image
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["uvicorn", "AppFiles.main:FastAPI_object", "--host", "0.0.0.0", "--port", "80"]