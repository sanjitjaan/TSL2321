FROM miniconda3:4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt-get install -y default-jdk
RUN pip3 install numpy pandas
RUN pip3 install python-javabridge
RUN pip3 install python-weka-wrapper3
