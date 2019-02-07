package simple

import (
	"errors"

	"gonum.org/v1/gonum/graph"
	"gorgonia.org/tensor"
)

type reshape struct{}

// Do ...
func (*reshape) Do(input ...tensor.Tensor) (tensor.Tensor, error) {
	if len(input) != 2 {
		return nil, errors.New("[reshape] bad arity")
	}
	output := input[0].Clone().(tensor.Tensor)
	err := output.Reshape(input[1].Shape()...)
	return output, err
}

// Make ...
func (d *reshape) Make(g graph.WeightedDirected, n graph.Node) error {
	n.(*Node).operation = d
	return nil
}
