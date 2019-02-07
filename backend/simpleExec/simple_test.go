package simple

import (
	"testing"

	onnx "github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/internal/examples/mnist"
	pb "github.com/owulveryck/onnx-go/internal/pb-onnx"
)

func TestMnist(t *testing.T) {

	graph := NewGraph()
	m := onnx.NewModel(graph)
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
	err = graph.Node(m.Input[0]).(*Node).ApplyTensor(inputT)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range m.Output {
		t.Log(graph.Node(v).(*Node).Data())
	}
}
