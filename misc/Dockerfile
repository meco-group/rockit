FROM conda/miniconda3-centos7

RUN yum -y install make

RUN conda create --name ocpx python=3.6
RUN conda install --name ocpx matplotlib
RUN conda install --name ocpx scipy
RUN conda install --name ocpx ipython
RUN conda install --name ocpx pylint
RUN conda install --name ocpx nose
RUN conda install --name ocpx sphinx
