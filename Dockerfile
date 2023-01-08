FROM tensorflow/tensorflow

WORKDIR '/de-project'

COPY requirements.txt ./requirements.txt                                                                                                                                                                                                
RUN python3 -m pip install -r ./requirements.txt
RUN python3 -m nltk.downloader stopwords

COPY ./data/financial_phrasebank.csv ./data/financial_phrasebank.csv

COPY source.py source.py

CMD [ "python3", "source.py"]