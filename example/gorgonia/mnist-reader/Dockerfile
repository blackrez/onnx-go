FROM golang:alpine AS build-env
COPY main.go /tmp
RUN mkdir -p /src/src/github.com/owulveryck/ && \
    cd /src/src/github.com/owulveryck/ && \
    apk add --no-cache git curl && \
    git clone https://github.com/owulveryck/onnx-go.git && \
    cd onnx-go && \
    git checkout v0.1-mnist-cli && \
    cd example/gorgonia/mnist-reader && \
    cp /tmp/main.go main.go && \
    #curl -LO https://gist.githubusercontent.com/blackrez/ae113858d823e2904cf25862164f9508/raw/a3c37dbc1c6a40eb9326f27fa279005213a12515/main.go && \
    export GOPATH=$GOPATH:/src/ && \
    go get && \
    go build -o /src/mnist-app

FROM alpine
WORKDIR /app
COPY --from=build-env /src/mnist-app /app/
RUN apk add --no-cache curl ca-certificates && \
    mkdir /data && \
    curl https://www.cntk.ai/OnnxModels/mnist/opset_7/mnist.tar.gz | tar -C /data -xzf - && \
    apk del curl
CMD ["./mnist-app", "-model", "/data/mnist/model.onnx", "-p", "8080"]
