FROM ibotdotout/python-opencv
MAINTAINER Lee Archer <la12808@my.bristol.ac.uk>

WORKDIR /app
RUN mkdir /app/images
COPY requirements.txt /app/
COPY configs/ /app/configs/
COPY analysis *.py /app/
COPY servers.yaml /app/
COPY facerec/ /app/facerec
COPY .pgpass /root/

ENV ER_SERVERS /app/servers.yaml

RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get remove -y python-pip && easy_install pip
RUN apt-get update && apt-get install -y python-{matplotlib,psycopg2,scipy,numpy,skimage,sklearn,pil} vim 
RUN pip install -r requirements.txt

CMD /bin/bash
