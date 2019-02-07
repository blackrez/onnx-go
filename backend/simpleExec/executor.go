package simple

import (
	"errors"

	onnx "github.com/owulveryck/onnx-go"
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/tensor"
)

// NewOp returns an operation that should be compatible with Operationa
// This methods is part of the OperationApplyer interface
func (g *Graph) NewOp(s string) (onnx.Op, error) {
	switch s {
	case "Reshape":
		return &reshape{}, nil
	default:
		return nil, &onnx.ErrNotImplemented{
			Operator: s,
		}
	}
}

// Apply apply the operation to the node n
func (g *Graph) Apply(op onnx.Op, n graph.Node) error {
	return op.Make(g.g, n)
}

// noop is a noop that simple display its name and its arguments
type noop struct {
	name string
}

// Do ...
func (*noop) Do(input ...tensor.Tensor) (tensor.Tensor, error) {
	if len(input) != 1 {
		return nil, errors.New("[noop] bad arity")
	}
	return input[0].Clone().(tensor.Tensor), nil
}

func (d *noop) Make(g graph.WeightedDirected, n graph.Node) error {
	n.(*Node).operation = d
	return nil
}
