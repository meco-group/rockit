FROM ubuntu:20.04

ENV DEBIAN_FRONTEND="noninteractive"
RUN apt update
RUN apt install -y cmake git wget g++
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN chmod +x miniconda.sh
RUN ./miniconda.sh -b
ENV PATH="/root/miniconda3/bin:${PATH}"

RUN echo "${PATH}"
RUN conda install matplotlib scipy pylint nose sphinx
    
RUN mkdir -p .pip
RUN pip --cache-dir=.pip install casadi sphinx-gallery sphinx_rtd_theme coverage pillow

RUN git clone https://github.com/jgillis/acados.git --branch python_fixes
WORKDIR acados
RUN git submodule init
RUN git submodule update
RUN ls
RUN mkdir build
RUN cd build && cmake .. && make install && cd ..
RUN cd interfaces/acados_template && pip install . && cd ../..
WORKDIR /
ENV ACADOS_SOURCE_DIR=/acados
ENV ACADOS_PATH=/acados

RUN wget https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux -O /acados/bin/t_renderer
RUN chmod +x /acados/bin/t_renderer

ENV LD_LIBRARY_PATH="/acados/lib:${LD_LIBRARY_PATH}"

CMD ["/bin/bash"]
