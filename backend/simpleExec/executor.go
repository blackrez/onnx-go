package simple

import (
	onnx "github.com/owulveryck/onnx-go"
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/tensor"
)

// NewOp returns an operation that should be compatible with Operationa
// This methods is part of the OperationApplyer interface
func (g *Graph) NewOp(s string) (onnx.Op, error) {
	return nil, nil
}

// Apply apply the operation to the node n
func (g *Graph) Apply(op onnx.Op, n graph.Node) error {
	return nil
}

// noop is a noop that simple display its name and its arguments
type noop struct {
	name string
}

// Do ...
func (*noop) Do(input ...tensor.Tensor) error {
	return nil
}

func (d *noop) Apply(g graph.WeightedDirected, n graph.Node) error {
	n.(*Node).operation = d
	return nil
}
