FROM golang:1.12

WORKDIR /go/src/github.com/owulveryck/onnx-go/
COPY . /go/src/github.com/owulveryck/onnx-go/

RUN go get -v -d && \
    go test && \
    cd backend/x/gorgonnx && \
    go get -v -d -t && \
    go test

