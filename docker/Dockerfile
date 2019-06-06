FROM alpine 
MAINTAINER youht

RUN echo "#install python3"
RUN apk add python3 --no-cache
RUN pip3 install flask flask-socketio requests bash pyyaml state_machine
RUN apk add gcc g++ make libffi-dev openssl-dev --no-cache
RUN apk add python3-dev --no-cache
RUN pip3 install pycrypto

RUN apk add curl --no-cache
RUN apk add nodejs nodejs-npm --no-cache

RUN npm config set registry https://registry.npm.taobao.org
RUN npm init --yes
RUN npm install webpack webpack-cli --save-dev
RUN npm install webpack-dev-server  --save-dev

RUN npm install react react-dom --save
RUN npm install react-router react-router-dom --save
RUN npm install react-hot-loader --save-dev

RUN npm install style-loader css-loader url-loader --save-dev
RUN npm install html-webpack-plugin html-loader --save-dev
RUN npm install babel-core babel-preset-env babel-loader babel-preset-react --save-dev

RUN npm install babel-plugin-import --save-dev
RUN npm install antd --save

RUN npm install jquery underscore moment --save
RUN npm install socket.io socket.io-client --save

RUN echo {presets:[\'env\',\'react\']} > .babelrc

#patch
RUN pip3 install flask-cors pytz
RUN pip3 install socketio-client
RUN pip3 install pymongo bson

RUN npm install react-redux --save
RUN npm install echarts --save

RUN apk add ipython3 --no-cache
RUN apk add jupyter --no-cache

RUN pip3 install pandas
RUN apk add libxml2 libxml2-dev libxslt libxslt-dev --no-cache
RUN pip3 install lxml html5lib beautifulsoup4 xlrd
#RUN pip3 install pillow
RUN pip3 install matplotlib
RUN pip3 install gym
RUN pip3 install psutil

RUN pip3 install pybind11
RUN pip3 install mako
RUN pip3 install pyopencl

RUN pip3 install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN pip3 install jupyter_nbextensions_configurator
RUN jupyter nbextensions_configurator enable --user
RUN pip3 install jupyterthemes

RUN jt -t oceans16 -T -N

WORKDIR /python
