FROM floydhub/dl-docker:cpu

RUN pip install --upgrade \
  keras \
  tensorflow \
  pandas \
  numpy
