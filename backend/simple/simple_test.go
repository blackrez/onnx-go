package simple

import (
	"log"
	"testing"

	onnx "github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/internal/examples/mnist"
	pb "github.com/owulveryck/onnx-go/internal/pb-onnx"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/traverse"
)

func TestMnist(t *testing.T) {

	g := NewGraph()
	m := onnx.NewModel(g)
	b := mnist.GetMnist()
	err := m.Unmarshal(b)
	if err != nil {
		t.Fatal(err)
	}

	sampleTestData := new(pb.TensorProto)
	err = sampleTestData.XXX_Unmarshal(mnist.GetTest1Input0())
	if err != nil {
		t.Fatal(err)
	}
	inputT, err := sampleTestData.Tensor()
	if err != nil {
		t.Fatal(err)
	}

	if len(m.Input) != 1 {
		t.Fatal("Expected only one input")
	}
	err = g.Node(m.Input[0]).(*Node).ApplyTensor(inputT)
	if err != nil {
		t.Fatal(err)
	}
	n := make([]graph.Node, 0)
	bf := traverse.BreadthFirst{
		EdgeFilter: nil,
		Visit: func(u, v graph.Node) {
			if len(n) == 0 || n[len(n)-1] != u {
				n = append(n, u)
			}
		},
	}
	bf.Walk(g, g.Node(m.Output[0]), nil)
	for i := len(n) - 1; i >= 0; i-- {
		log.Println(n[i])
	}

}
