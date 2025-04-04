FROM apache/nifi:latest
RUN mkdir -p /home/nifi
WORKDIR /home/nifi
COPY requirements.txt ./

USER root
RUN apt-get update
RUN apt install -y python3
RUN apt install -y python3-pip
RUN pip3 install --no-cache-dir -r requirements.txt


# Download nltk stopwords
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader vader_lexicon

WORKDIR /opt/nifi/nifi-current
