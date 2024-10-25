FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    python3-dev

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Django project into the container
COPY . /app/

# Expose the port Django will run on
EXPOSE 8000

# Run Django migrations and collect static files
RUN python manage.py collectstatic --noinput
RUN python manage.py migrate

# Set the command to run the Django app using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "blockhouse.wsgi:application"]
