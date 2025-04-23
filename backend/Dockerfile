FROM continuumio/miniconda3

WORKDIR /app

RUN conda create -n ox -c conda-forge --strict-channel-priority osmnx

COPY requirements.txt .

RUN /bin/bash -c "source activate ox && pip install --no-cache-dir -r requirements.txt"

COPY . .

CMD ["/bin/bash", "-c", "source activate ox && uvicorn main:app --host 0.0.0.0 --port 8000"]