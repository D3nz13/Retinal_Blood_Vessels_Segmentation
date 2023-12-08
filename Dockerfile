FROM tensorflow/tensorflow:2.13.0-gpu-jupyter

WORKDIR /dl_project

RUN apt-get update && apt-get install libgl1 -y

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8888

ENTRYPOINT [ "jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]