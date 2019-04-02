package onnxtest

// this file is auto-generated... DO NOT EDIT

import (
	"github.com/owulveryck/onnx-go/backend/testbackend"
	"gorgonia.org/tensor"
)

// NewTestThresholdedreluExample version: 4.
func NewTestThresholdedreluExample() *testbackend.TestCase {
	return &testbackend.TestCase{
		Title:  "TestThresholdedreluExample",
		ModelB: []byte{0x8, 0x4, 0x12, 0xc, 0x62, 0x61, 0x63, 0x6b, 0x65, 0x6e, 0x64, 0x2d, 0x74, 0x65, 0x73, 0x74, 0x3a, 0x6a, 0xa, 0x28, 0xa, 0x1, 0x78, 0x12, 0x1, 0x79, 0x22, 0xf, 0x54, 0x68, 0x72, 0x65, 0x73, 0x68, 0x6f, 0x6c, 0x64, 0x65, 0x64, 0x52, 0x65, 0x6c, 0x75, 0x2a, 0xf, 0xa, 0x5, 0x61, 0x6c, 0x70, 0x68, 0x61, 0x15, 0x0, 0x0, 0x0, 0x40, 0xa0, 0x1, 0x1, 0x12, 0x1c, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x74, 0x68, 0x72, 0x65, 0x73, 0x68, 0x6f, 0x6c, 0x64, 0x65, 0x64, 0x72, 0x65, 0x6c, 0x75, 0x5f, 0x65, 0x78, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x5a, 0xf, 0xa, 0x1, 0x78, 0x12, 0xa, 0xa, 0x8, 0x8, 0x1, 0x12, 0x4, 0xa, 0x2, 0x8, 0x5, 0x62, 0xf, 0xa, 0x1, 0x79, 0x12, 0xa, 0xa, 0x8, 0x8, 0x1, 0x12, 0x4, 0xa, 0x2, 0x8, 0x5, 0x42, 0x2, 0x10, 0xa},

		/*

		   &pb.NodeProto{
		     Input:     []string{"x"},
		     Output:    []string{"y"},
		     Name:      "",
		     OpType:    "ThresholdedRelu",
		     Attributes: ([]*pb.AttributeProto) (len=1 cap=1) {
		    (*pb.AttributeProto)(0xc00010b800)(name:"alpha" type:FLOAT f:2 )
		   }
		   ,
		   },


		*/

		Input: []tensor.Tensor{

			tensor.New(
				tensor.WithShape(5),
				tensor.WithBacking([]float32{-1.5, 0, 1.2, 2, 2.2}),
			),
		},
		ExpectedOutput: []tensor.Tensor{

			tensor.New(
				tensor.WithShape(5),
				tensor.WithBacking([]float32{0, 0, 0, 0, 2.2}),
			),
		},
	}
}
