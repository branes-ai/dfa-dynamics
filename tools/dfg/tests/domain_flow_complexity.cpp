#include <iostream>
#include <iomanip>

#include <dfa/dfa.hpp>

int main() {
	using namespace sw::dfa;

    DomainFlowGraph dfg("my-test-graph");
 
    dfg.addNode(DomainFlowNode(DomainFlowOperator::ADD, "test.Add").addOperand("tensor<4xf32>").addOperand("tensor<4xf32>").addResult("out", "tensor<4xf32>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::ADD, "test.Add").addOperand("tensor<4xf16>").addOperand("tensor<4xf16>").addResult("out", "tensor<4xf16>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::ADD, "test.Add").addOperand("tensor<4xf8>").addOperand("tensor<4xf8>").addResult("out", "tensor<4xf8>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::ADD, "test.Add").addOperand("tensor<4xi32>").addOperand("tensor<4xi32>").addResult("out", "tensor<4xi32>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::ADD, "test.Add").addOperand("tensor<4xi16>").addOperand("tensor<4xi16>").addResult("out", "tensor<4xi16>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::ADD, "test.Add").addOperand("tensor<4xi8>").addOperand("tensor<4xi8>").addResult("out", "tensor<4xi8>"));

    dfg.addNode(DomainFlowNode(DomainFlowOperator::MUL, "test.Mul").addOperand("tensor<1x4xf32>").addOperand("tensor<1x4xf32>").addResult("out", "tensor<1x4xf32>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::MUL, "test.Mul").addOperand("tensor<1x4xf16>").addOperand("tensor<1x4xf16>").addResult("out", "tensor<1x4xf16>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::MUL, "test.Mul").addOperand("tensor<1x4xf8>").addOperand("tensor<1x4xf8>").addResult("out", "tensor<1x4xf8>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::MUL, "test.Mul").addOperand("tensor<1x4xi32>").addOperand("tensor<1x4xi32>").addResult("out", "tensor<1x4xi32>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::MUL, "test.Mul").addOperand("tensor<1x4xi16>").addOperand("tensor<1x4xi16>").addResult("out", "tensor<1x4xi16>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::MUL, "test.Mul").addOperand("tensor<1x4xi8>").addOperand("tensor<1x4xi8>").addResult("out", "tensor<1x4xi8>"));

    dfg.addNode(DomainFlowNode(DomainFlowOperator::MATMUL, "test.Matmul").addOperand("tensor<1x4x4xf32>").addOperand("tensor<1x4x4xf32>").addResult("out", "tensor<1x4x4xf32>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::MATMUL, "test.Matmul").addOperand("tensor<2x4x4xf16>").addOperand("tensor<1x4x4xf16>").addResult("out", "tensor<1x4x4xf16>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::MATMUL, "test.Matmul").addOperand("tensor<4x4x4xf8>").addOperand("tensor<1x4x4xf8>").addResult("out", "tensor<1x4x4xf8>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::MATMUL, "test.Matmul").addOperand("tensor<8x4x4xi32>").addOperand("tensor<1x4x4xi32>").addResult("out", "tensor<1x4x4xi32>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::MATMUL, "test.Matmul").addOperand("tensor<16x4x4xi16>").addOperand("tensor<1x4x4xi16>").addResult("out", "tensor<1x4x4xi16>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::MATMUL, "test.Matmul").addOperand("tensor<32x4x4xi8>").addOperand("tensor<1x4x4xi8>").addResult("out", "tensor<1x4x4xi8>"));

    // TOSA compliant 4D tensor = (N, H, W, C)
    dfg.addNode(DomainFlowNode(DomainFlowOperator::CONV2D, "test.Conv2D").addOperand("tensor<1x224x224x3xf32>").addOperand("tensor<32x3x3x3xf32>").addResult("out", "tensor<1x112x112x32xf32>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::CONV2D, "test.Conv2D").addOperand("tensor<1x224x224x3xf16>").addOperand("tensor<32x3x3x3xf16>").addResult("out", "tensor<1x112x112x32xf16>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::CONV2D, "test.Conv2D").addOperand("tensor<1x224x224x3xf8>").addOperand("tensor<32x3x3x3xf8>").addResult("out", "tensor<1x112x112x32xf8>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::CONV2D, "test.Conv2D").addOperand("tensor<1x224x224x3xsi32>").addOperand("tensor<32x3x3x3xsi32>").addResult("out", "tensor<1x112x112x32xsi32>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::CONV2D, "test.Conv2D").addOperand("tensor<1x224x224x3xsi16>").addOperand("tensor<32x3x3x3xsi16>").addResult("out", "tensor<1x112x112x32xsi16>"));
    dfg.addNode(DomainFlowNode(DomainFlowOperator::CONV2D, "test.Conv2D").addOperand("tensor<1x224x224x3xsi8>").addOperand("tensor<32x3x3x3xsi8>").addResult("out", "tensor<1x112x112x32xsi8>"));

    reportOperatorStats(dfg);
    reportArithmeticComplexity(dfg);
    reportNumericalComplexity(dfg);

    return EXIT_SUCCESS;
}