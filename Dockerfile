FROM python:latest

RUN pip install numpy

RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install kneed
RUN pip install pandas
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install numpy
RUN pip install scikit-learn
RUN pip install tk

COPY . ./

#RUN chmod -R +x ./scripts

RUN chmod +x ./boot.sh

CMD ./boot.sh