package simple

import (
	"errors"
	"sort"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/iterator"
	"gonum.org/v1/gonum/graph/traverse"
	"gorgonia.org/tensor"
)

// Compute the graph to get the result of the node
func Compute(g *Graph, node int64) error {
	// Walk the graph
	n := make([]int64, 0)
	bf := traverse.BreadthFirst{
		EdgeFilter: nil,
		Visit: func(u, v graph.Node) {
			if len(n) == 0 || n[len(n)-1] != u.ID() {
				n = append(n, u.ID())
			}
		},
	}

	bf.Walk(g, g.Node(node), nil)
	if len(n) == 0 {
		return errors.New("unable to compute node, empty path")
	}
	for i := len(n) - 1; i >= 0; i-- {
		inputs := getChildrenValues(g, n[i])
		if inputs == nil || len(inputs) == 0 {
			return errors.New("input node's children don't have any value")
		}
		err := g.Node(n[i]).(*Node).operation.Do(g.Node(n[i]).(*Node).value, inputs...)
		if err != nil {
			return err
		}
	}

	return nil
}

func getChildrenValues(g *Graph, node int64) []tensor.Tensor {
	// Get all the edges that reach the node n
	children := g.From(node)
	// Now get the edges
	if children.Len() == 0 {
		return nil
	}
	edges := make([]graph.WeightedEdge, children.Len())
	for i := 0; children.Next(); i++ {
		edges[i] = g.WeightedEdge(node, children.Node().ID())
	}
	sort.Sort(byWeight(edges))

	children.Reset()
	orderWeightedEdges := iterator.NewOrderedWeightedEdges(edges)
	nodes := make([]graph.Node, children.Len())
	for i := 0; orderWeightedEdges.Next(); i++ {
		nodes[i] = orderWeightedEdges.WeightedEdge().To()
	}
	it := iterator.NewOrderedNodes(nodes)
	values := make([]tensor.Tensor, it.Len())
	for i := 0; it.Next(); i++ {
		values[i] = it.Node().(*Node).value
	}
	return values
}

type byWeight []graph.WeightedEdge

func (c byWeight) Len() int { return len(c) }
func (c byWeight) Less(i, j int) bool {
	return c[i].Weight() < c[j].Weight()
}
func (c byWeight) Swap(i, j int) { c[i], c[j] = c[j], c[i] }
