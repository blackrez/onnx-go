package onnxtest

// this file is auto-generated... DO NOT EDIT

import (
	"github.com/owulveryck/onnx-go/backend/testbackend"
	"gorgonia.org/tensor"
)

// NewTestReduceMinDefaultAxesKeepdimsRandom version: 3.
func NewTestReduceMinDefaultAxesKeepdimsRandom() *testbackend.TestCase {
	return &testbackend.TestCase{
		Title:  "TestReduceMinDefaultAxesKeepdimsRandom",
		ModelB: []byte{0x8, 0x3, 0x12, 0xc, 0x62, 0x61, 0x63, 0x6b, 0x65, 0x6e, 0x64, 0x2d, 0x74, 0x65, 0x73, 0x74, 0x3a, 0x96, 0x1, 0xa, 0x2b, 0xa, 0x4, 0x64, 0x61, 0x74, 0x61, 0x12, 0x7, 0x72, 0x65, 0x64, 0x75, 0x63, 0x65, 0x64, 0x22, 0x9, 0x52, 0x65, 0x64, 0x75, 0x63, 0x65, 0x4d, 0x69, 0x6e, 0x2a, 0xf, 0xa, 0x8, 0x6b, 0x65, 0x65, 0x70, 0x64, 0x69, 0x6d, 0x73, 0x18, 0x1, 0xa0, 0x1, 0x2, 0x12, 0x2c, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x72, 0x65, 0x64, 0x75, 0x63, 0x65, 0x5f, 0x6d, 0x69, 0x6e, 0x5f, 0x64, 0x65, 0x66, 0x61, 0x75, 0x6c, 0x74, 0x5f, 0x61, 0x78, 0x65, 0x73, 0x5f, 0x6b, 0x65, 0x65, 0x70, 0x64, 0x69, 0x6d, 0x73, 0x5f, 0x72, 0x61, 0x6e, 0x64, 0x6f, 0x6d, 0x5a, 0x1a, 0xa, 0x4, 0x64, 0x61, 0x74, 0x61, 0x12, 0x12, 0xa, 0x10, 0x8, 0x1, 0x12, 0xc, 0xa, 0x2, 0x8, 0x3, 0xa, 0x2, 0x8, 0x2, 0xa, 0x2, 0x8, 0x2, 0x62, 0x1d, 0xa, 0x7, 0x72, 0x65, 0x64, 0x75, 0x63, 0x65, 0x64, 0x12, 0x12, 0xa, 0x10, 0x8, 0x1, 0x12, 0xc, 0xa, 0x2, 0x8, 0x1, 0xa, 0x2, 0x8, 0x1, 0xa, 0x2, 0x8, 0x1, 0x42, 0x2, 0x10, 0x9},

		/*

		   &pb.NodeProto{
		     Input:     []string{"data"},
		     Output:    []string{"reduced"},
		     Name:      "",
		     OpType:    "ReduceMin",
		     Attributes: ([]*pb.AttributeProto) (len=1 cap=1) {
		    (*pb.AttributeProto)(0xc00010a800)(name:"keepdims" type:INT i:1 )
		   }
		   ,
		   },


		*/

		Input: []tensor.Tensor{

			tensor.New(
				tensor.WithShape(3, 2, 2),
				tensor.WithBacking([]float32{0.9762701, 4.303787, 2.0552676, 0.89766365, -1.526904, 2.9178822, -1.2482557, 7.83546, 9.273255, -2.3311696, 5.834501, 0.5778984}),
			),
		},
		ExpectedOutput: []tensor.Tensor{

			tensor.New(
				tensor.WithShape(1, 1, 1),
				tensor.WithBacking([]float32{-2.3311696}),
			),
		},
	}
}
