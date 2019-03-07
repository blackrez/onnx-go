package onnx

import (
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/internal/engine"
)

// Dropout operator ...
type Dropout struct {
	Ratio float64 `attributeName:"ratio"`
}

// NewDropout with a default value
func NewDropout() *Dropout {
	return &Dropout{
		Ratio: 0.5,
	}
}

// Constructor to fulfil the interface ...
func (r *Dropout) Constructor() func(g graph.WeightedDirected, n graph.Node) (interface{}, error) {
	return func(g graph.WeightedDirected, n graph.Node) (interface{}, error) {
		return engine.NewDropoutOperation(r.Ratio)(g, n.(*engine.Node))
	}
}
