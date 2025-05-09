#pragma once
#if WIN32
#pragma warning(disable : 4244 4267 4996)
#endif

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include <iostream>

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Attributes.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>
#include <queue>

// this module should not be used alone
// we are enforcing that by not including the dfg.hpp header
// That dfg.hpp header brings all these dependencies together

namespace sw {
    namespace dfa {

        // Helper struct to store parsed operand information
        struct OperandInfo {
            std::string name;
            mlir::Type type;
            size_t index;
        };

        // Helper struct to store parsed attribute information
        struct AttributeInfo {
            std::string name;
            std::string valueStr;
            mlir::Attribute attr;
        };

        // Helper struct for Clamp specific parsed attributes
        struct ClampAttributes {
            uint64_t minInt, maxInt;
            float minFp, maxFp;
        };

        // Helper struct for Conv2D specific parsed attributes
        struct Conv2DAttributes {
            std::vector<int64_t> pad;
            std::vector<int64_t> stride;
            std::vector<int64_t> dilation;
        };

        // Function to extract operand information from an operation (using reference)
        std::vector<OperandInfo> parseOperands(mlir::Operation& op) {
            std::vector<OperandInfo> operands;

            for (size_t i = 0; i < op.getNumOperands(); ++i) {
                mlir::Value operand = op.getOperand(i);
                OperandInfo info;
                info.index = i;
                info.type = operand.getType();

                // Try to get operand name from its defining op if available
                if (auto definingOp = operand.getDefiningOp()) {
                    info.name = definingOp->getName().getStringRef().str();
                }
                else {
                    info.name = "block_arg_" + std::to_string(i);
                }

                operands.push_back(info);
            }

            return operands;
        }

        // Function to extract attribute information from an operation
        std::vector<AttributeInfo> parseAttributes(mlir::Operation& op) {
            std::vector<AttributeInfo> attributes;

            for (mlir::NamedAttribute namedAttr : op.getAttrs()) {
                AttributeInfo info;
                info.name = namedAttr.getName().strref().str();
                info.attr = namedAttr.getValue();

                std::string attrStr;
                llvm::raw_string_ostream os(attrStr);
                namedAttr.getValue().print(os);
                info.valueStr = os.str();

                attributes.push_back(info);
            }

            return attributes;
        }

        // Function to extract block information from an operation (using reference)
        void parseBlocks(mlir::Operation& op, llvm::raw_ostream& os) {
            os << "Operation has " << op.getNumRegions() << " regions\n";

            for (size_t i = 0; i < op.getNumRegions(); ++i) {
                mlir::Region& region = op.getRegion(i);
                os << "  Region " << i << " has " << region.getBlocks().size() << " blocks\n";

                for (mlir::Block& block : region.getBlocks()) {
                    os << "    Block with " << block.getNumArguments() << " arguments and "
                        << block.getOperations().size() << " operations\n";

                    for (mlir::Operation& nestedOp : block.getOperations()) {
                        os << "      Operation: " << nestedOp.getName().getStringRef().str() << "\n";
                    }
                }
            }
        }

        static constexpr bool bTrace = false;

        // Function to extract Const specific attributes
        DomainFlowNode parseConst(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            auto constOp = mlir::cast<mlir::tosa::ConstOp>(op);
            // Parse basic operation information
            if constexpr (bTrace) os << "TOSA Const Operation:\n";
            // Parse operands
            // constOp does not have any operands
            // Parse attributes
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            if constexpr (bTrace)  os << "Attributes (" << attributes.size() << "):\n";
            //for (const auto& attr : attributes) {
            //    os << "  " << attr.name << "\n";
            //    //os << "  " << attr.name << ": " << attr.valueStr << "\n";
            //}
            // report result type
            if constexpr (bTrace) os << "Result:\n";
            if constexpr (bTrace) os << "  " << constOp.getOutput().getType() << "\n";
        
            // TODO: bring the attribute processing up to this function
            std::string opName = op.getName().getStringRef().str();
			// we do not need to store the attribute in the node as that
			// is the actual value of the constant and we are not using
			// it in the graph
            return DomainFlowNode(DomainFlowOperator::CONSTANT, opName);
        }

        // Function to extract Clamp specific attributes
        ClampAttributes parseClampAttributes(mlir::tosa::ClampOp clampOp) {
            ClampAttributes result{};

            // Extract min_int attribute
            if (auto minIntAttr = clampOp.getMinIntAttr()) {
                result.minInt = minIntAttr.getValue().getSExtValue();
            }

            // Extract max_int attribute
            if (auto maxIntAttr = clampOp.getMaxIntAttr()) {
                result.maxInt = maxIntAttr.getValue().getSExtValue();
            }

            // Extract min_fp attribute
            if (auto minFpAttr = clampOp.getMinFpAttr()) {
                result.minFp = minFpAttr.getValue().convertToDouble();
            }

            // Extract max_fp attribute
            if (auto maxFpAttr = clampOp.getMaxFpAttr()) {
                result.maxFp = maxFpAttr.getValue().convertToDouble();
            }

            return result;
        }

        // A specialized function to parse TOSA Clamp operation
        DomainFlowNode parseTosaClamp(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {

            auto clampOp = mlir::cast<mlir::tosa::ClampOp>(op);
            // Parse Clamp specific attributes
            ClampAttributes clampAttrs = parseClampAttributes(clampOp);

            // Parse basic operation information
            if constexpr (bTrace) {
                os << "TOSA Clamp Operation:\n";

                // Parse operands
                os << "Operands:\n";
                os << "  Input: " << clampOp.getInput().getType() << "\n";


                os << "Attributes:\n";
                os << "  Min Int: " << clampAttrs.minInt << "\n";
                os << "  Max Int: " << clampAttrs.maxInt << "\n";
                os << "  Min FP: " << clampAttrs.minFp << "\n";
                os << "  Max FP: " << clampAttrs.maxFp << "\n";


                // Parse result
                os << "Result:\n";
                os << "  " << clampOp.getOutput().getType() << "\n";
            }
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::CLAMP, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }

        // Function to extract Conv2D specific attributes
        Conv2DAttributes parseConv2DAttributes(mlir::tosa::Conv2DOp convOp) {
            Conv2DAttributes result;

            // Extract pad attribute
            if (auto padAttr = convOp.getPadAttr()) {
                auto padValues = padAttr.asArrayRef();
                result.pad.assign(padValues.begin(), padValues.end());
            }

            // Extract stride attribute
            if (auto strideAttr = convOp.getStrideAttr()) {
                auto strideValues = strideAttr.asArrayRef();
                result.stride.assign(strideValues.begin(), strideValues.end());
            }

            // Extract dilation attribute
            if (auto dilationAttr = convOp.getDilationAttr()) {
                auto dilationValues = dilationAttr.asArrayRef();
                result.dilation.assign(dilationValues.begin(), dilationValues.end());
            }

            return result;
        }

        // A specialized function to parse TOSA Conv2D operations
        DomainFlowNode parseTosaConv2D(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {

            auto convOp = mlir::cast<mlir::tosa::Conv2DOp>(op);

            // Parse basic operation information
            if constexpr (bTrace) {
                // Parse Conv2D specific attributes
                Conv2DAttributes convAttrs = parseConv2DAttributes(convOp);

                os << "TOSA Conv2D Operation:\n";

                // Parse operands
                os << "Operands:\n";
                os << "  Input: " << convOp.getInput().getType() << "\n";
                os << "  Weight: " << convOp.getWeight().getType() << "\n";
                if (convOp.getBias())
                    os << "  Bias: " << convOp.getBias().getType() << "\n";

                os << "Attributes:\n";
                os << "  Pad: [";
                for (size_t i = 0; i < convAttrs.pad.size(); ++i) {
                    if (i > 0) os << ", ";
                    os << convAttrs.pad[i];
                }
                os << "]\n";
                os << "  Stride: [";
                for (size_t i = 0; i < convAttrs.stride.size(); ++i) {
                    if (i > 0) os << ", ";
                    os << convAttrs.stride[i];
                }
                os << "]\n";
                os << "  Dilation: [";
                for (size_t i = 0; i < convAttrs.dilation.size(); ++i) {
                    if (i > 0) os << ", ";
                    os << convAttrs.dilation[i];
                }
                os << "]\n";

                // Parse result
                os << "Result:\n";
                os << "  " << convOp.getOutput().getType() << "\n";
            }
            
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
			auto node = DomainFlowNode(DomainFlowOperator::CONV2D, opName);
			for (const auto& attr : attributes) {
				node.addAttribute(attr.name, attr.valueStr);
			}
            return node;
        }

        // A specialized function to parse TOSA Conv2D operations
        DomainFlowNode parseTosaConv3D(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::CONV3D, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }

        // A specialized function to parse TOSA Reshape operations
        DomainFlowNode parseTosaReshape(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::RESHAPE, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA Transpose operations
        DomainFlowNode parseTosaTranspose(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::TRANSPOSE, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA DepthwiseConv2D operations
        DomainFlowNode parseTosaDepthwiseConv2D(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::DEPTHWISE_CONV2D, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA TransposeConv2D operations
        DomainFlowNode parseTosaTransposeConv2D(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::TRANSPOSE_CONV2D, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA FullyConnected operations
        DomainFlowNode parseTosaFullyConnected(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::FC, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA Matmul operations
        DomainFlowNode parseTosaMatmul(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::MATMUL, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA Add operations
        DomainFlowNode parseTosaAdd(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::ADD, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA Sub operations
        DomainFlowNode parseTosaSub(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::SUB, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA Mul operations
        DomainFlowNode parseTosaMul(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::MUL, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA Negate operations
        DomainFlowNode parseTosaNegate(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::NEGATE, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }

        // A specialized function to parse TOSA Pad operations
        DomainFlowNode parseTosaPad(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::PAD, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA Cast operations
        DomainFlowNode parseTosaCast(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::CAST, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA Gather operations
        DomainFlowNode parseTosaGather(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::GATHER, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }

        // function ops
        // A specialized function to parse TOSA Reciprocal operations
        DomainFlowNode parseTosaReciprocal(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::RECIPROCAL, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA ReduceAll operations
        DomainFlowNode parseTosaReduceAll(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::REDUCE_ALL, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA ReduceMax operations
        DomainFlowNode parseTosaReduceMax(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::REDUCE_MAX, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA ReduceMin operations
        DomainFlowNode parseTosaReduceMin(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::REDUCE_MIN, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA ReduceSum operations
        DomainFlowNode parseTosaReduceSum(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::REDUCE_SUM, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA ReduceProd operations
        DomainFlowNode parseTosaReduceProd(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::REDUCE_PROD, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }

        // A specialized function to parse TOSA Exp operations
        DomainFlowNode parseTosaExp(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::EXP, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA Abs operations
        DomainFlowNode parseTosaAbs(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::ABS, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }
        // A specialized function to parse TOSA Concat operations
        DomainFlowNode parseTosaConcat(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            std::string opName = op.getName().getStringRef().str();
            std::vector<AttributeInfo> attributes = parseAttributes(op);
            auto node = DomainFlowNode(DomainFlowOperator::CONCAT, opName);
            for (const auto& attr : attributes) {
                node.addAttribute(attr.name, attr.valueStr);
            }
            return node;
        }


        // Parse the TOSA Op and add to the graph
        DomainFlowNode parseOperation(DomainFlowGraph& gr, mlir::Operation& op, llvm::raw_ostream& os) {
            DomainFlowNode node;

            if (mlir::isa<mlir::tosa::ConstOp>(op)) {
                node = parseConst(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::Conv2DOp>(op)) {
                node = parseTosaConv2D(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::Conv3DOp>(op)) {
                node = parseTosaConv3D(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::ClampOp>(op)) {
                node = parseTosaClamp(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::ReshapeOp>(op)) {
                node = parseTosaReshape(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::TransposeOp>(op)) {
                node = parseTosaTranspose(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::DepthwiseConv2DOp>(op)) {
                node = parseTosaDepthwiseConv2D(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::TransposeConv2DOp>(op)) {
                node = parseTosaTransposeConv2D(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::PadOp>(op)) {
                node = parseTosaPad(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::FullyConnectedOp>(op)) {
                node = parseTosaFullyConnected(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::MatMulOp>(op)) {
                node = parseTosaMatmul(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::AddOp>(op)) {
                node = parseTosaAdd(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::SubOp>(op)) {
                node = parseTosaSub(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::MulOp>(op)) {
                node = parseTosaMul(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::NegateOp>(op)) {
                node = parseTosaNegate(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::ExpOp>(op)) {
                node = parseTosaExp(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::AbsOp>(op)) {
                node = parseTosaAbs(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::ConcatOp>(op)) {
                node = parseTosaConcat(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::CastOp>(op)) {
                node = parseTosaCast(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::GatherOp>(op)) {
                node = parseTosaGather(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::ReciprocalOp>(op)) {
                node = parseTosaReciprocal(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::ReduceAllOp>(op)) {
                node = parseTosaReduceAll(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::ReduceMaxOp>(op)) {
                node = parseTosaReduceMax(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::ReduceMinOp>(op)) {
                node = parseTosaReduceMin(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::ReduceSumOp>(op)) {;
                node = parseTosaReduceSum(gr, op, os);
            }
            else if (mlir::isa<mlir::tosa::ReduceProdOp>(op)) {
                node = parseTosaReduceProd(gr, op, os);
            }

            else {

                std::string opName = op.getName().getStringRef().str();
                if (opName == "func.return") {
					node = DomainFlowNode(DomainFlowOperator::FUNCTION_RETURN, opName);
                }
                else {
                    os << "\nDetected untracked TOSA operation:\n";
                    os << "Operation: " << opName << "\n";
                    node = DomainFlowNode(opName);

                    // Parse operands
                    std::vector<OperandInfo> operands = parseOperands(op);
                    os << "Operands (" << operands.size() << "):\n";
                    for (const auto& operand : operands) {
                        os << "  " << operand.index << ": " << operand.name << " of type " << operand.type << "\n";
                    }

                    // Parse attributes
                    std::vector<AttributeInfo> attributes = parseAttributes(op);
                    os << "Attributes (" << attributes.size() << "):\n";
                    for (const auto& attr : attributes) {
                        os << "  " << attr.name << "\n";
                        //os << "  " << attr.name << ": " << attr.valueStr << "\n";
                    }
                }
            }

            // VALIDATE that this nesting is correct
            if (op.getNumRegions() > 0) {
                // Parse blocks
                parseBlocks(op, os);
            }

            return node;
        }


    }
}