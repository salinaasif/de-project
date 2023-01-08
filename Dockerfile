FROM tensorflow/tensorflow:latest@sha256:eea5989852623037f354c49404b66761467516b79ab7af26e643b5ac7382c53f

WORKDIR '/de-project'

COPY requirements.txt ./requirements.txt                                                                                                                                                                                                
RUN python3 -m pip install -r ./requirements.txt
RUN python3 -m nltk.downloader stopwords

COPY ./data/financial_phrasebank.csv ./data/financial_phrasebank.csv

COPY source.py source.py

CMD [ "python3", "source.py"]