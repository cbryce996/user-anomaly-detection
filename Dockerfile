FROM python:latest

RUN pip install numpy

RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY . ./

RUN chmod -R +x ./scripts

RUN chmod +x ./boot.sh

CMD ./boot.sh