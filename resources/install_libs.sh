# Arquivo para instalar as bibliotecas necessárias do python para o escopo do projeto
# OBS: Feito para rodar em ambientes derivados do Debian/Ubuntu

pip3 install opencv-python opencv-contrib-python cvzone mediapipe flet numpy pillow matplotlib
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


#sudo apt install python3-pip python3-virtualenv -y
sudo apt install -y libgtk-3-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
                    libmpv-dev mpv libmpv2

sudo ln -s /usr/lib/x86_64-linux-gnu/libmpv.so /usr/lib/libmpv.so.1

#install docker
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSl https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
sudo apt install docker-ce 