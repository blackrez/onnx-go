package onnx

import (
	"github.com/gogo/protobuf/proto"
	pb "github.com/owulveryck/onnx-go/internal/pb-onnx"
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/tensor"
)

// Model is a top level structure that carry the execution graph.
// Input holds the ID of the nodes that are not initialized.
// Output hlds the ID of the top level nodes.
type Model struct {
	g        graph.DirectedWeightedBuilder
	dbByName map[string]graph.Node
	Input    []int64
	Output   []int64
}

// NewModel create a model and sets the internal graph to "dst"
func NewModel(dst graph.DirectedWeightedBuilder) *Model {
	return &Model{
		g: dst,
	}
}

// Unmarshal the ONNX data in the model, filling the
// graph as well as Input and Output.
func (m *Model) Unmarshal(data []byte) error {
	model := &pb.ModelProto{}
	err := proto.Unmarshal(data, model)
	if err != nil {
		return err
	}
	return m.unmarshal(model)
}

// GetNodeByName ...
func (m *Model) GetNodeByName(name string) (graph.Node, bool) {
	n, ok := m.dbByName[name]
	return n, ok
}

// Unmarshal a NN model encoded in ONNX-Protobuf format into a graph .
// The weight of the edges represent the indices of the children (therefore their order).
// The first child weights 0.
//
// Executable graphs
//
// If dst fulfils the OperationCarrier interface, the corresponding methods are called after the initialization of
// the structure.
//
// Node values
//
// If the graph nodes are fulfilling the TensorCarrier interface, this function set their values and shapes by calling
// the corresponding methods.
func Unmarshal(data []byte, dst graph.DirectedWeightedBuilder) error {
	model := &pb.ModelProto{}
	err := proto.Unmarshal(data, model)
	if err != nil {
		return err
	}
	m := &Model{
		g: dst,
	}
	return m.unmarshal(model)
}

// processValue creates a new node and if the node is a TensorCarrier sets its value.
func (m *Model) processValue(io *pb.ValueInfoProto) (graph.Node, error) {
	dst := m.g
	n := dst.NewNode()
	if _, ok := n.(Namer); ok {
		n.(Namer).SetName(io.Name)
	}
	dst.AddNode(n)
	m.dbByName[io.Name] = n
	if _, ok := n.(TensorCarrier); !ok {
		return n, nil
	}
	ttype := io.Type.GetTensorType()
	shape := make([]int, len(ttype.Shape.Dim))
	for i, d := range ttype.Shape.Dim {
		shape[i] = int(d.GetDimValue())
	}
	dtype, err := pb.TensorProto_DataType(ttype.GetElemType()).Dtype()
	if err != nil {
		return n, err
	}
	t := tensor.New(tensor.WithShape(shape...), tensor.Of(dtype))
	err = n.(TensorCarrier).ApplyTensor(t)
	if err != nil {
		return n, err
	}

	return n, nil
}

func (m *Model) unmarshal(model *pb.ModelProto) error {
	m.Input = make([]int64, len(model.Graph.Input))
	m.Output = make([]int64, len(model.Graph.Output))
	m.dbByName = make(map[string]graph.Node, len(model.Graph.Output)+len(model.Graph.Input))
	dst := m.g
	// Well...
	for i, io := range model.Graph.Input {
		n, err := m.processValue(io)
		if err != nil {
			return err
		}
		m.Input[i] = n.ID()
	}
	for _, io := range model.Graph.ValueInfo {
		_, err := m.processValue(io)
		if err != nil {
			return err
		}
	}
	for i, io := range model.Graph.Output {
		n, err := m.processValue(io)
		if err != nil {
			return err
		}
		m.Output[i] = n.ID()
	}
	for _, tensorProto := range model.Graph.GetInitializer() {
		name := tensorProto.GetName()
		if name == "" {
			return errors.New("initializer should have a name")
		}
		n, ok := m.dbByName[name]
		if !ok {
			return errors.New("invalid model: initializer has not been defined in input, output or value")
		}
		// Remove it from the input
		// find the ID
		for i := 0; i < len(m.Input); i++ {
			if m.Input[i] == n.ID() {
				m.Input = append(m.Input[:i], m.Input[i+1:]...)
			}
		}
		if _, ok := n.(TensorCarrier); !ok {
			continue
		}
		t, err := tensorProto.Tensor()
		if err != nil {
			return err
		}
		err = n.(TensorCarrier).ApplyTensor(t)
		if err != nil {
			return err
		}
	}
	for _, node := range model.Graph.Node {
		for _, output := range node.Output {
			var ok bool
			var no graph.Node
			if no, ok = m.dbByName[output]; !ok {
				return &ErrInvalidModel{
					NodeNotDefined: output,
				}
			}
			// input should be ordered for non-commutatives operations
			for i, input := range node.Input {
				var ni graph.Node
				var ok bool
				if ni, ok = m.dbByName[input]; !ok {
					return &ErrInvalidModel{
						NodeNotDefined: input,
					}
				}
				e := dst.NewWeightedEdge(no, ni, float64(i))
				dst.SetWeightedEdge(e)
			}
			// The graph can apply operations
			if _, ok := dst.(OperationCarrier); ok {
				op, err := dst.(OperationCarrier).NewOp(node.OpType)
				if err != nil {
					return err
				}
				// TODO check the pointer
				err = pb.UnmarshalAttributes(node.GetAttribute(), op)
				if err != nil {
					return err
				}

				err = dst.(OperationCarrier).Apply(op, no)
				if err != nil {
					return err
				}

			}
		}
	}
	return nil
}

// OperationCarrier is any graph that can apply operations on its nodes
// in regard to the structure of the graph
type OperationCarrier interface {
	NewOp(s string) (Op, error)
	Apply(operation Op, n graph.Node) error
}

// Op represent an operation to be applied on a node N of the graph G.
type Op interface {
	// Make build the operation regarding the node's shape and its surrounding
	// elements inside the graph.
	// For example, the Make operation of the Add operator can take care of broadcasting.
	// please take care that within this method you can use shape, type and dimensions of
	// the children nodes, but (most of the time) not their actual values.
	//
	// WARNING: this is not the actual execution of the operation that should be performed
	// with whatever method at runtime while walking the graph. The order of execution
	// of the operators is not guaranteed at the build phase.
	Make(g graph.WeightedDirected, n graph.Node) error
}
