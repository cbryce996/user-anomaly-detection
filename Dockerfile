FROM python:latest

RUN pip install numpy

RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu

ADD src/test.py /

CMD ["python", "./test.py"]