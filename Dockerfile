FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Install OpenCV dependencies for image processing (RapidOCR)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Set up a new user named "user" with user ID 1000 for Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# Copy the rest of the application
COPY --chown=user . /home/user/app

# Hugging Face Spaces expose port 7860
CMD ["uvicorn", "dawai_api:app", "--host", "0.0.0.0", "--port", "7860"]
