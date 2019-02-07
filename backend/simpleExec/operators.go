package simple

import (
	"errors"

	"gonum.org/v1/gonum/graph"
	"gorgonia.org/tensor"
)

type reshape struct{}

// Do ...
func (*reshape) Do(t tensor.Tensor, input ...tensor.Tensor) error {
	if len(input) != 1 {
		return errors.New("[reshape] bad arity")
	}
	return t.Reshape(input[0].Shape()...)
}

// Make ...
func (d *reshape) Make(g graph.WeightedDirected, n graph.Node) error {
	n.(*Node).operation = d
	return nil
}
