FROM arm64v8/ubuntu:20.04

# Download and install Miniconda
RUN apt-get update && apt-get install -y wget
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh
RUN bash /tmp/miniconda.sh -b -p /opt/conda
RUN rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Create a new conda environment and install packages
#RUN conda create -n myenv python=3.8
#RUN echo "source activate myenv" >> ~/.bashrc
#RUN conda activate myenv
RUN echo "source /opt/conda/bin/activate" >> ~/.bashrc
RUN conda install -y pyyaml tensorflow pandas scikit-learn plotly seaborn
RUN conda install -c apple tensorflow-deps
RUN python -m pip install tensorflow-macos==2.9 && \
    python -m pip install tensorflow-metal==0.5.0
~                                                    
