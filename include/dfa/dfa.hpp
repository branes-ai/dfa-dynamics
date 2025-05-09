#pragma once

// core linear algebra abstractions for dealing with vector spaces
#include <dfa/vector.hpp>
#include <dfa/matrix.hpp>

// core dependency graph abstractions
#include <dfa/affine_map.hpp>
#include <dfa/recurrence_var.hpp>
#include <dfa/dependency_graph.hpp>

// core domain flow abstractions
#include <dfa/hyperplane.hpp>
#include <dfa/constraint_set.hpp>
#include <dfa/index_point.hpp>
#include <dfa/index_space.hpp>
#include <dfa/schedule.hpp>
#include <dfa/projection.hpp>
#include <dfa/domain_flow_edge.hpp>    // definition of the edge
#include <dfa/domain_flow_node.hpp>    // definition of the node
#include <dfa/domain_flow_graph.hpp>   // definition of the graph

// unifying space of aligned domains of computation, domain flow, and wavefronts
#include <dfa/domain_flow.hpp>
