package onnxtest

// this file is auto-generated... DO NOT EDIT

import (
	"github.com/owulveryck/onnx-go/backend/testbackend"
	"gorgonia.org/tensor"
)

// NewTestConcat1dAxis0 version: 3.
func NewTestConcat1dAxis0() *testbackend.TestCase {
	return &testbackend.TestCase{
		Title:  "TestConcat1dAxis0",
		ModelB: []byte{0x8, 0x3, 0x12, 0xc, 0x62, 0x61, 0x63, 0x6b, 0x65, 0x6e, 0x64, 0x2d, 0x74, 0x65, 0x73, 0x74, 0x3a, 0x88, 0x1, 0xa, 0x2d, 0xa, 0x6, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x30, 0xa, 0x6, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x31, 0x12, 0x6, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x22, 0x6, 0x43, 0x6f, 0x6e, 0x63, 0x61, 0x74, 0x2a, 0xb, 0xa, 0x4, 0x61, 0x78, 0x69, 0x73, 0x18, 0x0, 0xa0, 0x1, 0x2, 0x12, 0x15, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x63, 0x6f, 0x6e, 0x63, 0x61, 0x74, 0x5f, 0x31, 0x64, 0x5f, 0x61, 0x78, 0x69, 0x73, 0x5f, 0x30, 0x5a, 0x14, 0xa, 0x6, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x30, 0x12, 0xa, 0xa, 0x8, 0x8, 0x1, 0x12, 0x4, 0xa, 0x2, 0x8, 0x2, 0x5a, 0x14, 0xa, 0x6, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x31, 0x12, 0xa, 0xa, 0x8, 0x8, 0x1, 0x12, 0x4, 0xa, 0x2, 0x8, 0x2, 0x62, 0x14, 0xa, 0x6, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x12, 0xa, 0xa, 0x8, 0x8, 0x1, 0x12, 0x4, 0xa, 0x2, 0x8, 0x4, 0x42, 0x2, 0x10, 0x9},

		/*

		   &pb.NodeProto{
		     Input:     []string{"value0", "value1"},
		     Output:    []string{"output"},
		     Name:      "",
		     OpType:    "Concat",
		     Attributes: ([]*pb.AttributeProto) (len=1 cap=1) {
		    (*pb.AttributeProto)(0xc00010b600)(name:"axis" type:INT )
		   }
		   ,
		   },


		*/

		Input: []tensor.Tensor{

			tensor.New(
				tensor.WithShape(2),
				tensor.WithBacking([]float32{1, 2}),
			),

			tensor.New(
				tensor.WithShape(2),
				tensor.WithBacking([]float32{3, 4}),
			),
		},
		ExpectedOutput: []tensor.Tensor{

			tensor.New(
				tensor.WithShape(4),
				tensor.WithBacking([]float32{1, 2, 3, 4}),
			),
		},
	}
}
