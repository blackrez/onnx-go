package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"image"
	"io"
	"io/ioutil"
	"log"
	"math"
	"net/http"

	"cloud.google.com/go/storage"
	"github.com/disintegration/imaging"
	"github.com/owulveryck/onnx-go"
	"github.com/vincent-petithory/dataurl"
	"gorgonia.org/gorgonia/node"
	gorgonnx "gorgonia.org/gorgonia/onnx"
	"gorgonia.org/tensor"
)

const size = 28

var (
	graph  *gorgonnx.Graph
	model  *onnx.Model
	client *storage.Client
)

func main() {
	port := flag.String("p", "8100", "port to serve on")
	directory := flag.String("d", ".", "the directory of static file to host")
	onnxModel := flag.String("model", "", "the onnx model to use")
	flag.Parse()

	graph = gorgonnx.NewGraph()
	model = onnx.NewModel(graph)

	b, err := ioutil.ReadFile(*onnxModel)
	if err != nil {
		log.Fatal(err)
	}
	//b := mnist.GetMnist()
	err = model.Unmarshal(b)
	if err != nil {
		log.Fatal(err)
	}

	if len(model.Input) != 1 {
		log.Fatal("Expected only one input")
	}
	http.Handle("/", http.FileServer(http.Dir(*directory)))
	http.HandleFunc("/picture", getPicture)
	log.Printf("Serving %s on HTTP port: %s\n", *directory, *port)
	log.Fatal(http.ListenAndServe(":"+*port, nil))
}

func enableCors(w *http.ResponseWriter) {
	(*w).Header().Set("Access-Control-Allow-Origin", "*")
}

func getPicture(w http.ResponseWriter, r *http.Request) {
	dataURL, err := dataurl.Decode(r.Body)
	defer r.Body.Close()
	enableCors(&w)
	if err != nil {
		log.Println(err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if dataURL.ContentType() == "image/png" {
		rawimg := bytes.NewReader(dataURL.Data)
		m, _, err := image.Decode(rawimg)
		if err != nil {
			log.Println(err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		hash := sha256.New()
		_, err = io.Copy(hash, rawimg)
		if err != nil {
			log.Fatal(err)
		}
		sum := hex.EncodeToString(hash.Sum(nil))

		if client == nil {
			// Pre-declare an err variable to avoid shadowing client.
			var err error
			client, err = storage.NewClient(context.Background())
			if err != nil {
				http.Error(w, "Internal error", http.StatusInternalServerError)
				log.Printf("storage.NewClient: %v", err)
				return
			}
		}
		ctx := context.Background()
		_, err = rawimg.Seek(0, 0)
		if err != nil {
			log.Fatal(err)
		}
		wc := client.Bucket("demo-onnx-gorgonia").Object("images/" + sum + ".png").NewWriter(ctx)
		if _, err = io.Copy(wc, rawimg); err != nil {
			log.Println(err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if err := wc.Close(); err != nil {
			log.Println(err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		output, err := analyze(m)
		if err != nil {
			log.Println(err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		log.Println("Sending result", output)
		fmt.Fprintf(w, "%v", output)
	} else {
		http.Error(w, "not a png", http.StatusBadRequest)
	}
}

func analyze(m image.Image) (int, error) {
	// resize the image
	img := imaging.Resize(m, size, 0, imaging.Lanczos)
	t := make([]float32, size*size)
	for i := 0; i < size*size*4; i += 4 {
		t[i/4] = float32(img.Pix[i])
	}
	T := tensor.New(tensor.WithBacking(t), tensor.WithShape(1, 1, size, size))
	err := gorgonnx.Let(graph.Node(model.Input[0]).(node.Node), T)
	if err != nil {
		log.Println(err)
		return 0, err
	}
	// create a VM to run the program on
	machine := gorgonnx.NewTapeMachine(graph)

	// Run the program
	err = machine.RunAll()
	if err != nil {
		log.Println(err)
		return 0, nil
	}
	val := float32(-math.MaxFloat32)
	res := 0
	for i, v := range graph.Node(model.Output[0]).(node.Node).Value().Data().([]float32) {
		if v > val {
			res = i
			val = v
		}
	}
	return res, nil
}
