FROM nvcr.io/nvidia/tritonserver:24.03-py3

COPY ./flood_model /models/flood_model

CMD ["tritonserver", "--model-repository=/models"]