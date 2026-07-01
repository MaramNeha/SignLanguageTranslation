# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.10

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Run Flask app on port 7860
CMD ["python", "app.py"]
