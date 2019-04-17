FROM tvmai/demo-cpu
WORKDIR /root

COPY initialize.py /root

RUN pip3 install rpyc mxnet
RUN python3 'initialize.py' 'r100-ii' 'https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip?dl=1'

COPY main.py /root

EXPOSE 50052
CMD ["python3", "main.py"]