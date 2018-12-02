# docker-keras - Keras in Docker with Python 3 and tensorflow on CPU

FROM debian:stretch

#Don't ask questions during install
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    # install essentials
    build-essential \
    g++ \
    git \
    wget \
    openssh-client \
    locales\
    nano\
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Anaconda installing
RUN wget --no-check-certificate https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
RUN rm Anaconda3-5.0.1-Linux-x86_64.sh
# Set path to conda
ENV PATH /root/anaconda3/bin:$PATH
ENV PYTHONIOENCODING=utf-8

# Updating Anaconda packages
RUN conda update conda
RUN conda update anaconda
RUN conda update --all


RUN pip install --upgrade pip
RUN pip install tensorflow
RUN pip install keras
#RUN pip install theano
#RUN git config --local http.sslBackend "openssl"

#RUN pip install --no-dependencies git+https://github.com/fchollet/keras.git@${KERAS_VERSION}

# Install Theano and set up Theano config (.theanorc) OpenBLAS
#RUN pip --no-cache-dir install git+git://github.com/Theano/Theano.git@${THEANO_VERSION} && \
#	\
#	echo "[global]\ndevice=cpu\nfloatX=float32\nmode=FAST_RUN \
#		\n[lib]\ncnmem=0.95 \
#		\n[nvcc]\nfastmath=True \
#		\n[blas]\nldflag = -L/usr/lib/openblas-base -lopenblas \
#		\n[DebugMode]\ncheck_finite=1" \
#	> /root/.theanorc

# quick test and dump package lists
#RUN python -c "import theano; print(theano.__version__)" \
# && dpkg-query -l > /dpkg-query-l.txt \
# && pip freeze > /pip-freeze.txt

RUN pip install pandas dash dash_core_components dash_html_components plotly flask_caching dash_table

#ports
EXPOSE 8050

COPY . /SG_merqueo/
WORKDIR /SG_merqueo/src/

RUN chmod +x run_dashboard.sh

CMD ./run_dashboard.sh
