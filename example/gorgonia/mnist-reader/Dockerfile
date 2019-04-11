FROM golang:alpine AS build-env
RUN mkdir -p /src/src/github.com/owulveryck/ && \
    cd /src/src/github.com/owulveryck/ && \
    apk add --no-cache git && \
    git clone https://github.com/owulveryck/onnx-go.git && \
    cd onnx-go && \
    git checkout cloud-run && \
    cd example/gorgonia/mnist-reader && \
    # Replacing the file because the API of onnx-go is not stable yet and their is vendored libs
    # will fix later when 1/ onnx-go will be stable 2/ I will time for this
    curl -LO https://raw.githubusercontent.com/blackrez/onnx-go/cloud-run/example/gorgonia/mnist-reader/main.go && \
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
