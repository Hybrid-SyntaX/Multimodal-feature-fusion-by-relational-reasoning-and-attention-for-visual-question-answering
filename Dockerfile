FROM pytorchlightning/pytorch_lightning:latest-py3.9-torch1.9
RUN pip install h5py
RUN pip install -U pip setuptools wheel
RUN pip install -U spacy[cuda114]
RUN python -m spacy download en_core_web_lg
