FROM tensorflow/tensorflow:latest

RUN apt-get update && apt-get install -y \
    git \
    python3-tk \
    bash-completion

RUN echo 'if [ -f /etc/bash_completion ] && ! shopt -oq posix; then' >> /root/.bashrc && \
    echo '    . /etc/bash_completion' >> /root/.bashrc && \
    echo 'fi' >> /root/.bashrc
