package onnxtest

// this file is auto-generated... DO NOT EDIT

import (
	"github.com/owulveryck/onnx-go/backend/testbackend"
	"gorgonia.org/tensor"
)

// NewTestConvtransposeKernelShape version: 3.
func NewTestConvtransposeKernelShape() *testbackend.TestCase {
	return &testbackend.TestCase{
		Title:  "TestConvtransposeKernelShape",
		ModelB: []byte{0x8, 0x3, 0x12, 0xc, 0x62, 0x61, 0x63, 0x6b, 0x65, 0x6e, 0x64, 0x2d, 0x74, 0x65, 0x73, 0x74, 0x3a, 0xf1, 0x1, 0xa, 0x77, 0xa, 0x1, 0x58, 0xa, 0x1, 0x57, 0x12, 0x1, 0x59, 0x1a, 0x4, 0x74, 0x65, 0x73, 0x74, 0x22, 0xd, 0x43, 0x6f, 0x6e, 0x76, 0x54, 0x72, 0x61, 0x6e, 0x73, 0x70, 0x6f, 0x73, 0x65, 0x2a, 0x15, 0xa, 0xc, 0x6b, 0x65, 0x72, 0x6e, 0x65, 0x6c, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x40, 0x3, 0x40, 0x3, 0xa0, 0x1, 0x7, 0x2a, 0x17, 0xa, 0xe, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x5f, 0x70, 0x61, 0x64, 0x64, 0x69, 0x6e, 0x67, 0x40, 0x1, 0x40, 0x1, 0xa0, 0x1, 0x7, 0x2a, 0x15, 0xa, 0xc, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x40, 0xa, 0x40, 0x8, 0xa0, 0x1, 0x7, 0x2a, 0x10, 0xa, 0x7, 0x73, 0x74, 0x72, 0x69, 0x64, 0x65, 0x73, 0x40, 0x3, 0x40, 0x2, 0xa0, 0x1, 0x7, 0x12, 0x1f, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x63, 0x6f, 0x6e, 0x76, 0x74, 0x72, 0x61, 0x6e, 0x73, 0x70, 0x6f, 0x73, 0x65, 0x5f, 0x6b, 0x65, 0x72, 0x6e, 0x65, 0x6c, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x5a, 0x1b, 0xa, 0x1, 0x58, 0x12, 0x16, 0xa, 0x14, 0x8, 0x1, 0x12, 0x10, 0xa, 0x2, 0x8, 0x1, 0xa, 0x2, 0x8, 0x1, 0xa, 0x2, 0x8, 0x3, 0xa, 0x2, 0x8, 0x3, 0x5a, 0x1b, 0xa, 0x1, 0x57, 0x12, 0x16, 0xa, 0x14, 0x8, 0x1, 0x12, 0x10, 0xa, 0x2, 0x8, 0x1, 0xa, 0x2, 0x8, 0x2, 0xa, 0x2, 0x8, 0x3, 0xa, 0x2, 0x8, 0x3, 0x62, 0x1b, 0xa, 0x1, 0x59, 0x12, 0x16, 0xa, 0x14, 0x8, 0x1, 0x12, 0x10, 0xa, 0x2, 0x8, 0x1, 0xa, 0x2, 0x8, 0x2, 0xa, 0x2, 0x8, 0xa, 0xa, 0x2, 0x8, 0x8, 0x42, 0x2, 0x10, 0x9},

		/*

		   &pb.NodeProto{
		     Input:     []string{"X", "W"},
		     Output:    []string{"Y"},
		     Name:      "test",
		     OpType:    "ConvTranspose",
		     Attributes: ([]*pb.AttributeProto) (len=4 cap=4) {
		    (*pb.AttributeProto)(0xc00014a700)(name:"kernel_shape" type:INTS ints:3 ints:3 ),
		    (*pb.AttributeProto)(0xc00014a800)(name:"output_padding" type:INTS ints:1 ints:1 ),
		    (*pb.AttributeProto)(0xc00014a900)(name:"output_shape" type:INTS ints:10 ints:8 ),
		    (*pb.AttributeProto)(0xc00014aa00)(name:"strides" type:INTS ints:3 ints:2 )
		   }
		   ,
		   },


		*/

		Input: []tensor.Tensor{

			tensor.New(
				tensor.WithShape(1, 1, 3, 3),
				tensor.WithBacking([]float32{0, 1, 2, 3, 4, 5, 6, 7, 8}),
			),

			tensor.New(
				tensor.WithShape(1, 2, 3, 3),
				tensor.WithBacking([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
			),
		},
		ExpectedOutput: []tensor.Tensor{

			tensor.New(
				tensor.WithShape(1, 2, 10, 8),
				tensor.WithBacking([]float32{0, 0, 1, 1, 3, 2, 2, 0, 0, 0, 1, 1, 3, 2, 2, 0, 0, 0, 1, 1, 3, 2, 2, 0, 3, 3, 7, 4, 9, 5, 5, 0, 3, 3, 7, 4, 9, 5, 5, 0, 3, 3, 7, 4, 9, 5, 5, 0, 6, 6, 13, 7, 15, 8, 8, 0, 6, 6, 13, 7, 15, 8, 8, 0, 6, 6, 13, 7, 15, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 2, 2, 0, 0, 0, 1, 1, 3, 2, 2, 0, 0, 0, 1, 1, 3, 2, 2, 0, 3, 3, 7, 4, 9, 5, 5, 0, 3, 3, 7, 4, 9, 5, 5, 0, 3, 3, 7, 4, 9, 5, 5, 0, 6, 6, 13, 7, 15, 8, 8, 0, 6, 6, 13, 7, 15, 8, 8, 0, 6, 6, 13, 7, 15, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
			),
		},
	}
}
