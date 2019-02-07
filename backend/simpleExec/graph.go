package simple

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
	"gonum.org/v1/gonum/graph/simple"
	"gorgonia.org/tensor"
)

// Graph is a simple directed graph
type Graph struct {
	g *simple.WeightedDirectedGraph
}

// Node of the graph
type Node struct {
	id          int64
	name        string
	description string
	value       tensor.Tensor
	opType      string
	operation   operation

	//attributes  []*pb.Attribute
}

// An operation is a type evaluated at runtime
type operation interface {
	Do(tensor.Tensor, ...tensor.Tensor) error
}

// ID to fulfil the graph.Node interface
func (n *Node) ID() int64 {
	return n.id
}

// Attributes it a method to fulfil the encoding/dot package
func (n *Node) Attributes() []encoding.Attribute {
	var value string
	value = fmt.Sprintf(`%v`, n.name)
	if n.opType != "" {
		value = fmt.Sprintf(`%v|{Op|%v}`, value, n.opType)
	}
	if n.value != nil {
		value = fmt.Sprintf(`%v|{shape|%v}|{type|%v}`, value, n.value.Shape(), n.value.Dtype())
	}
	value = fmt.Sprintf(`"%v"`, value)
	return []encoding.Attribute{
		encoding.Attribute{
			Key:   "shape",
			Value: "Mrecord",
		},
		encoding.Attribute{
			Key:   "label",
			Value: value,
		},
	}
}

// SetName to fulfil the Namer interface
func (n *Node) SetName(desc string) {
	n.name = desc
}

// GetName to fulfil the Namer interface
func (n *Node) GetName() string {
	return n.name
}

// SetDescription to fulfil the Namer interface
func (n *Node) SetDescription(desc string) {
	n.description = desc
}

// GetDescription to fulfil the Namer interface
func (n *Node) GetDescription() string {
	return n.description
}

// ApplyTensor to fulfil the TensorCarrier interface
func (n *Node) ApplyTensor(t tensor.Tensor) error {
	n.value = t
	return nil
}

type attributer []encoding.Attribute

func (a attributer) Attributes() []encoding.Attribute { return a }

// DOTAttributers is used for representation in graphviz
func (g *Graph) DOTAttributers() (graph, node, edge encoding.Attributer) {
	graphAttributes := attributer{
		encoding.Attribute{
			Key:   "label",
			Value: "name",
		},
		encoding.Attribute{
			Key:   "rankdir",
			Value: `"LR"`,
		},
	}
	nodeAttributes := attributer{
		encoding.Attribute{
			Key:   "style",
			Value: `"rounded,filled"`,
		},
		encoding.Attribute{
			Key:   "shape",
			Value: "Mrecord",
		},
		encoding.Attribute{
			Key:   "fillcolor",
			Value: "white",
		},
	}
	return graphAttributes, nodeAttributes, attributer{}
}

// NewGraph ...
func NewGraph() *Graph {
	return &Graph{
		g: simple.NewWeightedDirectedGraph(math.MaxFloat64, -1),
	}
}

// SetWeightedEdge ...
func (g *Graph) SetWeightedEdge(e graph.WeightedEdge) {
	g.g.SetWeightedEdge(e)

}

// NewWeightedEdge ...
func (g *Graph) NewWeightedEdge(from, to graph.Node, w float64) graph.WeightedEdge {
	return g.g.NewWeightedEdge(from, to, w)

}

// AddNode ...
func (g *Graph) AddNode(n graph.Node) {
	g.g.AddNode(n)

}

// NewNode ...
func (g *Graph) NewNode() graph.Node {
	n := g.g.NewNode()
	return &Node{
		id: n.ID(),
	}
}

// Node ...
func (g *Graph) Node(id int64) graph.Node {
	return g.g.Node(id)
}

// Nodes ...
func (g *Graph) Nodes() graph.Nodes {
	return g.g.Nodes()
}

// From ...
func (g *Graph) From(id int64) graph.Nodes {
	return g.g.From(id)
}

// HasEdgeBetween ...
func (g *Graph) HasEdgeBetween(xid, yid int64) bool {
	return g.g.HasEdgeBetween(xid, yid)
}

// Edge ...
func (g *Graph) Edge(uid, vid int64) graph.Edge {
	return g.g.Edge(uid, vid)
}

// HasEdgeFromTo ...
func (g *Graph) HasEdgeFromTo(uid, vid int64) bool {
	return g.g.HasEdgeFromTo(uid, vid)
}

// To ...
func (g *Graph) To(id int64) graph.Nodes {
	return g.g.To(id)
}

// Data returns the underlying tensor
func (n *Node) Data() tensor.Tensor {
	return n.value
}

// WeightedEdge returns the weighted edge from u to v
// if such an edge exists and nil otherwise.
// The node v must be directly reachable from u as defined by the From method.
func (g *Graph) WeightedEdge(uid, vid int64) graph.WeightedEdge {
	return g.g.WeightedEdge(uid, vid)
}
