# OVERVIEW: IREE compilation driver

```
USAGE: iree-compile.exe [options] <input file or '-' for stdin>

OPTIONS:

CUDA HAL Target:

  --iree-cuda-target=<string>                                              - CUDA target as expected by LLVM NVPTX backend; e.g., 'sm_80'/'sm_90' for targeting Ampere/Hopper GPUs. Additionally this also supports architecture code names like 'turing'/'ampere' or some product names like 'a100'/'rtx3090ti' for a better experience. See https://iree.dev/guides/deployment-configurations/gpu-cuda for more details.
  --iree-cuda-target-features=<string>                                     - CUDA target features as expected by LLVM NVPTX backend; e.g. use '+ptxNN' to set PTX version to NN.
  --iree-cuda-use-ptxas                                                    - Whether to use the ptxas tool to assemble the generated PTX code and put the generated CUBIN binary file into the executable. If not set, directly embeds the PTX into the executable. To specify the exact ptxas tool path, use '--iree-cuda-use-ptxas-from'. To pass additional parameters to ptxas, use '--iree-cuda-use-ptxas-params', e.g. '--iree-cuda-use-ptxas-params=-v'
  --iree-cuda-use-ptxas-from=<string>                                      - Uses the ptxas tool from the given path. Requires '--iree-cuda-use-ptxas' to be true.
  --iree-cuda-use-ptxas-params=<string>                                    - Passes the given additional parameters to ptxas. Requires '--iree-cuda-use-ptxas' to be true.

Color Options:

  --color                                                                  - Use colors in output (default=autodetect)

General options:

  --aarch64-neon-syntax=<value>                                            - Choose style of NEON code to emit from AArch64 backend:
    =generic                                                               -   Emit generic NEON assembly
    =apple                                                                 -   Emit Apple-style NEON assembly
  --aarch64-use-aa                                                         - Enable the use of AA during codegen.
  --abort-on-max-devirt-iterations-reached                                 - Abort when the max iterations for devirtualization CGSCC repeat pass is reached
  --allow-ginsert-as-artifact                                              - Allow G_INSERT to be considered an artifact. Hack around AMDGPU test infinite loops.
  --amdgpu-atomic-optimizer-strategy=<value>                               - Select DPP or Iterative strategy for scan
    =DPP                                                                   -   Use DPP operations for scan
    =Iterative                                                             -   Use Iterative approach for scan
    =None                                                                  -   Disable atomic optimizer
  --amdgpu-bypass-slow-div                                                 - Skip 64-bit divide for dynamic 32-bit values
  --amdgpu-disable-loop-alignment                                          - Do not align and prefetch loops
  --amdgpu-dpp-combine                                                     - Enable DPP combiner
  --amdgpu-dump-hsa-metadata                                               - Dump AMDGPU HSA Metadata
  --amdgpu-enable-merge-m0                                                 - Merge and hoist M0 initializations
  --amdgpu-enable-power-sched                                              - Enable scheduling to minimize mAI power bursts
  --amdgpu-indirect-call-specialization-threshold=<uint>                   - A threshold controls whether an indirect call will be specialized
  --amdgpu-kernarg-preload-count=<uint>                                    - How many kernel arguments to preload onto SGPRs
  --amdgpu-module-splitting-max-depth=<uint>                               - maximum search depth. 0 forces a greedy approach. warning: the algorithm is up to O(2^N), where N is the max depth.
  --amdgpu-promote-alloca-to-vector-limit=<uint>                           - Maximum byte size to consider promote alloca to vector
  --amdgpu-sdwa-peephole                                                   - Enable SDWA peepholer
  --amdgpu-use-aa-in-codegen                                               - Enable the use of AA during codegen.
  --amdgpu-verify-hsa-metadata                                             - Verify AMDGPU HSA Metadata
  --amdgpu-vgpr-index-mode                                                 - Use GPR indexing mode instead of movrel for vector indexing
  --arm-add-build-attributes                                               - 
  --arm-implicit-it=<value>                                                - Allow conditional instructions outside of an IT block
    =always                                                                -   Accept in both ISAs, emit implicit ITs in Thumb
    =never                                                                 -   Warn in ARM, reject in Thumb
    =arm                                                                   -   Accept in ARM, reject in Thumb
    =thumb                                                                 -   Warn in ARM, emit implicit ITs in Thumb
  --atomic-counter-update-promoted                                         - Do counter update using atomic fetch add  for promoted counters only
  --atomic-first-counter                                                   - Use atomic fetch add for first counter in a function (usually the entry counter)
  --bounds-checking-single-trap                                            - Use one trap block per function
  --cfg-hide-cold-paths=<number>                                           - Hide blocks with relative frequency below the given value
  --cfg-hide-deoptimize-paths                                              - 
  --cfg-hide-unreachable-paths                                             - 
  --check-functions-filter=<regex>                                         - Only emit checks for arguments of functions whose names match the given regular expression
  --compile-from=<value>                                                   - Compilation phase to resume from, starting with the following phase.
    =start                                                                 -   Entry point to the compilation pipeline.
    =input                                                                 -   Performs input processing and lowering into core IREE input dialects (linalg/etc).
    =abi                                                                   -   Adjusts program ABI for the specified execution environment.
    =preprocessing                                                         -   Compiles up to the `preprocessing` specified
    =global-optimization                                                   -   Compiles up to global optimization.
    =dispatch-creation                                                     -   Compiles up to dispatch creation.
    =flow                                                                  -   Compiles up to the `flow` dialect.
    =stream                                                                -   Compiles up to the `stream` dialect.
    =executable-sources                                                    -   Compiles up to just before `hal.executable`s are configured, excluding codegen.
    =executable-configurations                                             -   Compiles up to just before `hal.executable`s are translated, including selection of translation strategies for codegen.
    =executable-targets                                                    -   Compiles up to translated `hal.executable`s, including codegen.
    =hal                                                                   -   Compiles up to the `hal` dialect, including codegen.
    =vm                                                                    -   Compiles up to the `vm` dialect.
    =end                                                                   -   Complete the full compilation pipeline.
  --compile-to=<value>                                                     - Compilation phase to run up until before emitting output.
    =start                                                                 -   Entry point to the compilation pipeline.
    =input                                                                 -   Performs input processing and lowering into core IREE input dialects (linalg/etc).
    =abi                                                                   -   Adjusts program ABI for the specified execution environment.
    =preprocessing                                                         -   Compiles up to the `preprocessing` specified
    =global-optimization                                                   -   Compiles up to global optimization.
    =dispatch-creation                                                     -   Compiles up to dispatch creation.
    =flow                                                                  -   Compiles up to the `flow` dialect.
    =stream                                                                -   Compiles up to the `stream` dialect.
    =executable-sources                                                    -   Compiles up to just before `hal.executable`s are configured, excluding codegen.
    =executable-configurations                                             -   Compiles up to just before `hal.executable`s are translated, including selection of translation strategies for codegen.
    =executable-targets                                                    -   Compiles up to translated `hal.executable`s, including codegen.
    =hal                                                                   -   Compiles up to the `hal` dialect, including codegen.
    =vm                                                                    -   Compiles up to the `vm` dialect.
    =end                                                                   -   Complete the full compilation pipeline.
  --conditional-counter-update                                             - Do conditional counter updates in single byte counters mode)
  --cost-kind=<value>                                                      - Target cost kind
    =throughput                                                            -   Reciprocal throughput
    =latency                                                               -   Instruction latency
    =code-size                                                             -   Code size
    =size-latency                                                          -   Code size and latency
  --debug-info-correlate                                                   - Use debug info to correlate profiles. (Deprecated, use -profile-correlate=debug-info)
  --debugify-func-limit=<ulong>                                            - Set max number of processed functions per pass.
  --debugify-level=<value>                                                 - Kind of debug info to add
    =locations                                                             -   Locations only
    =location+variables                                                    -   Locations and Variables
  --debugify-quiet                                                         - Suppress verbose debugify output
  --disable-auto-upgrade-debug-info                                        - Disable autoupgrade of debug info
  --disable-i2p-p2i-opt                                                    - Disables inttoptr/ptrtoint roundtrip optimization
  --disable-promote-alloca-to-lds                                          - Disable promote alloca to LDS
  --disable-promote-alloca-to-vector                                       - Disable promote alloca to vector
  --do-counter-promotion                                                   - Do counter register promotion
  --dot-cfg-mssa=<file name for generated dot file>                        - file name for generated dot file
  --dump-compilation-phases-to=<string>                                    - Dumps IR at the end of each compilation phase to the given directory.
  --emit-mlir-bytecode                                                     - Emit bytecode when generating compile-to or VM MLIR output.
  --emit-mlir-bytecode-version=<value>                                     - Use specified bytecode version when generating compile-to or VM MLIR output.
  --emscripten-cxx-exceptions-allowed=<string>                             - The list of function names in which Emscripten-style exception handling is enabled (see emscripten EMSCRIPTEN_CATCHING_ALLOWED options)
  --enable-cse-in-irtranslator                                             - Should enable CSE in irtranslator
  --enable-cse-in-legalizer                                                - Should enable CSE in Legalizer
  --enable-emscripten-cxx-exceptions                                       - WebAssembly Emscripten-style exception handling
  --enable-emscripten-sjlj                                                 - WebAssembly Emscripten-style setjmp/longjmp handling
  --enable-gvn-hoist                                                       - Enable the GVN hoisting pass (default = off)
  --enable-gvn-memdep                                                      - 
  --enable-gvn-sink                                                        - Enable the GVN sinking pass (default = off)
  --enable-jump-table-to-switch                                            - Enable JumpTableToSwitch pass (default = off)
  --enable-load-in-loop-pre                                                - 
  --enable-load-pre                                                        - 
  --enable-loop-simplifycfg-term-folding                                   - 
  --enable-name-compression                                                - Enable name/filename string compression
  --enable-split-backedge-in-load-pre                                      - 
  --enable-split-loopiv-heuristic                                          - Enable loop iv regalloc heuristic
  --enable-vtable-profile-use                                              - If ThinLTO and WPD is enabled and this option is true, vtable profiles will be used by ICP pass for more efficient indirect call sequence. If false, type profiles won't be used.
  --enable-vtable-value-profiling                                          - If true, the virtual table address will be instrumented to know the types of a C++ pointer. The information is used in indirect call promotion to do selective vtable-based comparison.
  --expand-variadics-override=<value>                                      - Override the behaviour of expand-variadics
    =unspecified                                                           -   Use the implementation defaults
    =disable                                                               -   Disable the pass entirely
    =optimize                                                              -   Optimise without changing ABI
    =lowering                                                              -   Change variadic calling convention
  --experimental-debug-variable-locations                                  - Use experimental new value-tracking variable locations
  --experimental-debuginfo-iterators                                       - Enable communicating debuginfo positions through iterators, eliminating intrinsics. Has no effect if --preserve-input-debuginfo-format=true.
  --force-tail-folding-style=<value>                                       - Force the tail folding style
    =none                                                                  -   Disable tail folding
    =data                                                                  -   Create lane mask for data only, using active.lane.mask intrinsic
    =data-without-lane-mask                                                -   Create lane mask with compare/stepvector
    =data-and-control                                                      -   Create lane mask using active.lane.mask intrinsic, and use it for both data and control flow
    =data-and-control-without-rt-check                                     -   Similar to data-and-control, but remove the runtime check
    =data-with-evl                                                         -   Use predicated EVL instructions for tail folding. If EVL is unsupported, fallback to data-without-lane-mask.
  --fs-profile-debug-bw-threshold=<uint>                                   - Only show debug message if the source branch weight is greater  than this value.
  --fs-profile-debug-prob-diff-threshold=<uint>                            - Only show debug message if the branch probility is greater than this value (in percentage).
  --generate-merged-base-profiles                                          - When generating nested context-sensitive profiles, always generate extra base profile for function with all its context profiles merged into it.
  --hash-based-counter-split                                               - Rename counter variable of a comdat function based on cfg hash
  --hot-cold-split                                                         - Enable hot-cold splitting pass
  --hwasan-percentile-cutoff-hot=<int>                                     - Hot percentile cuttoff.
  --hwasan-random-rate=<number>                                            - Probability value in the range [0.0, 1.0] to keep instrumentation of a function.
  --import-all-index                                                       - Import all external functions in index.
  --instcombine-code-sinking                                               - Enable code sinking
  --instcombine-guard-widening-window=<uint>                               - How wide an instruction window to bypass looking for another guard
  --instcombine-max-num-phis=<uint>                                        - Maximum number phis to handle in intptr/ptrint folding
  --instcombine-max-sink-users=<uint>                                      - Maximum number of undroppable users for instruction sinking
  --instcombine-maxarray-size=<uint>                                       - Maximum array size considered when doing a combine
  --instcombine-negator-enabled                                            - Should we attempt to sink negations?
  --instcombine-negator-max-depth=<uint>                                   - What is the maximal lookup depth when trying to check for viability of negation sinking.
  --instrprof-atomic-counter-update-all                                    - Make all profile counter updates atomic (for testing only)
  --internalize-public-api-file=<filename>                                 - A file containing list of symbol names to preserve
  --internalize-public-api-list=<list>                                     - A list of symbol names to preserve
  --iree-codegen-dump-tuning-specs-to=<string>                             - Dump the final tuning spec modules to the specified directory. When set to '-', prints the tuning spec to stdout.
  --iree-codegen-enable-default-tuning-specs                               - Whether to enable default tuning spec transform libraries shipped with the compiler
  --iree-codegen-gpu-native-math-precision                                 - Skip polynomial lowering for math op natively available on GPU
  --iree-codegen-linalg-max-constant-fold-elements=<long>                  - Maximum number of elements to try to constant fold.
  --iree-codegen-llvm-verbose-debug-info                                   - Emit verbose debug information in LLVM IR.
  --iree-codegen-llvmgpu-matmul-c-matrix-threshold=<int>                   - matmul c matrix element count threshold to be considered as small vs. large when deciding MMA schedule
  --iree-codegen-llvmgpu-test-tile-and-fuse-matmul                         - test the the tile and fuse pipeline for matmul
  --iree-codegen-llvmgpu-test-tile-and-fuse-vectorize                      - test the tile and fuse pipeline for all supported operations
  --iree-codegen-llvmgpu-use-igemm                                         - Enable implicit gemm for convolutions.
  --iree-codegen-llvmgpu-use-mma-sync                                      - force use mma sync instead of wmma ops
  --iree-codegen-llvmgpu-use-tile-and-fuse-convolution                     - enable the tile and fuse pipeline for supported convolutions
  --iree-codegen-llvmgpu-use-unaligned-gemm-vector-distribution            - enable the usage of the vector distribution pipeline for unaligned GEMMs when supported
  --iree-codegen-llvmgpu-use-vector-distribution                           - enable the usage of the vector distribution pipeline
  --iree-codegen-llvmgpu-use-wmma                                          - force use of wmma operations for tensorcore
  --iree-codegen-llvmgpu-vectorize-pipeline                                - forces use of the legacy LLVMGPU vectorize pipeline
  --iree-codegen-mmt4d-use-intrinsics                                      - Whether to use instrinsics when lowering vector contracts generated from mmt4d matmuls (as opposed to inline asm). Not for production use.
  --iree-codegen-notify-transform-strategy-application                     - Emit a remark when a transform configuration strategy successfully applies on a function. This is intended for testing/debuging.
  --iree-codegen-reorder-workgroups-strategy=<value>                       - Reorder workgroup IDs using the selected strategy
    =none                                                                  -   No workgroup reordering
    =transpose                                                             -   Transpose
  --iree-codegen-transform-dialect-library=<string>                        - File path to a module containing a library of transform dialectstrategies. Can be suffixed with the name of a transform sequencewithin the library to run as preprocessing per executable variant.This is specified as <file-path>@<sequence-name>. If not specified,this will default to `__kernel_config`.
  --iree-codegen-tuning-spec-path=<string>                                 - File path to a module containing a tuning spec (transform dialect library).
  --iree-config-add-tuner-attributes                                       - Adds attribute for tuner.
  --iree-consteval-jit-debug                                               - Prints debugging information to stderr (useful since when consteval has issues, it is often in production on the largest models where we don't want to run a debug compiler).
  --iree-consteval-jit-target-device=<string>                              - Overrides the target device used for JIT'ing.
  --iree-dispatch-creation-collapse-reduction-dims                         - Enable collapsing of reduction dims
  --iree-dispatch-creation-element-wise-fuse-multi-reduction               - Enable element-wise fusion of multi-reduction loop ops.
  --iree-dispatch-creation-enable-aggressive-fusion                        - Aggressive fusion opportunities that are behind a flag since all backends dont support it yet
  --iree-dispatch-creation-enable-detensoring                              - Enable changing of tensor operations into scalar operations.
  --iree-dispatch-creation-enable-fuse-horizontal-contractions             - Enables horizontal fusion of contractions with one common operand
  --iree-dispatch-creation-enable-fuse-padding-into-linalg-consumer-ops    - Enable fusing tensor.pad ops into Linalg consumer ops.
  --iree-dispatch-creation-enable-fuse-padding-into-linalg-producer-ops    - Enable fusing tensor.pad ops into Linalg consumer ops.
  --iree-dispatch-creation-experimental-data-tiling                        - Enable data-tiling at flow level, i.e., it sets encodings in dispatch regions, hoist them out of region, and enables fusion for the set_encodings. This is still an experimental path. The current main data tiling path is iree-opt-data-tiling, which is on by default. To use this path, --iree-opt-data-tiling=false must be set as wells
  --iree-dispatch-creation-fuse-multi-use                                  - Fuse multi-use ops.
  --iree-dispatch-creation-pad-factor=<int>                                - Provides padding size hints that will be attached to encodings. This only affects the experimental data tiling path in DispatchCreation with iree-dispatch-creation-experimental-data-tiling.
  --iree-dispatch-creation-split-matmul-reduction=<long>                   - split ratio
  --iree-dispatch-creation-topk-split-reduction=<long>                     - comma separated list of split ratios
  --iree-experimental-packed-i1-storage                                    - Experimental feature: enable i1 data type support in codegen
  --iree-flow-break-dispatch=<string>                                      - Enables inserting a break after a specified dispatch. Supports two modes; breaking on the dispatch ordinal before deduplication (@function_name:<index>) and breaking on the dispatch symbol.
  --iree-flow-dump-dispatch-graph                                          - Dump a dot graph for dispatches.
  --iree-flow-dump-dispatch-graph-output-file=<string>                     - Output file name for a dispatch graph dump.
  --iree-flow-enable-gather-fusion                                         - Fuse gather-like ops with consumer.
  --iree-flow-enable-pad-handling                                          - Enable native handling of tensor.pad operations.
  --iree-flow-export-benchmark-funcs                                       - Exports one function per original module entry point and unique flow.executable that dispatches with dummy arguments.
  --iree-flow-inline-constants-max-byte-length=<int>                       - Maximum byte-length of tensor constant that can be inlined into a dispatch region or 0 to disable inlining.
  --iree-flow-trace-dispatch=<string>                                      - Enables tracing tensors at specified dispatches. Supports two modes; tracing the dispatch by ordinal before deduplication (@function_name:<index>) and tracing all occurrences of the dispatch symbol.
  --iree-flow-trace-dispatch-tensors                                       - Trace runtime input/output tensors for each dispatch function.
  --iree-flow-zero-fill-empty-tensors                                      - Zero fill empty tensors instead of leaving them uninitialized.
  --iree-global-opt-enable-demote-contraction-inputs-to-bf16=<value>       - Demotes inputs (LHS, RHS) of contraction ops to BF16. Selects types of contraction ops to demote.
    =all                                                                   -   Demote all contraction ops.
    =conv                                                                  -   Only demote convolution ops.
    =matmul                                                                -   Only demote matmul ops.
    =none                                                                  -   Demote no contraction ops.
  --iree-global-opt-enable-early-materialization                           - Enables early materialization on encodings. Note, this flag should be false eventually. This does not work for heterogeneous computing.
  --iree-global-opt-enable-fuse-silu-horizontal-matmul                     - Enables fusing specifically structured matmuls (experimental).
  --iree-global-opt-enable-quantized-matmul-reassociation                  - Enables reassociation of quantized matmul ops (experimental).
  --iree-global-opt-experimental-rocm-data-tiling                          - Enables data-tiling materializatino for rocm backends (experimental).
  --iree-global-opt-pad-factor=<int>                                       - provides padding size hints that will be attached to encodings.
  --iree-global-opt-propagate-transposes                                   - Enables propagation of transpose ops to improve fusion chances.
  --iree-gpu-test-target=<string>                                          - The target for IR LIT tests. Format is '<arch>:<feature>@<api>', where <feature> and <api> are optional; e.g., 'gfx942:+sramecc,-xnack@hip'. If <api> is missing, it will be deduced from <arch>; e.g., 'gfx*' defaults to HIP, 'sm_*' defaults to CUDA
  --iree-hal-benchmark-dispatch-repeat-count=<uint>                        - The number of times to repeat each hal.command_buffer.dispatch op. This simply duplicates the dispatch op and inserts barriers. It's meant for command buffers having linear dispatch structures.
  --iree-hal-executable-object-search-path=<string>                        - Additional search paths for resolving #hal.executable.object file references.
  --iree-hal-force-indirect-command-buffers                                - Forces indirect command buffers when they would otherwise not be chosen due to the values they capture. They may not be reusable but will still be outlined.
  --iree-hal-indirect-command-buffers                                      - Whether to turn buffer bindings into indirect references when recording command buffers.
  --iree-hal-instrument-dispatches=<power of two byte size>                - Enables dispatch instrumentation with a power-of-two byte size used for storage (16mib, 64mib, 2gib, etc).
  --iree-hal-link-executables                                              - Controls linking of executables. The default is to always link, however disabling linking allows inspecting serialization of each executable in isolation and will dump a single binary per executable when used in conjunction with `--iree-hal-dump-executable-binaries-to`.
  --iree-hal-list-target-backends                                          - Lists all registered target backends for executable compilation.
  --iree-hal-memoization                                                   - Whether to memoize device resources such as command buffers.
  --iree-hal-preprocess-executables-with=<string>                          - Passes each hal.executable to the given command. Multiple commands may be specified and they will be executed in order. A command may either be a pass pipeline available within the IREE compiler specified as `builtin.module(...)` or a shell tool that consumes a hal.executable MLIR file on stdin and produces a modified hal.executable on stdout. Non-zero exit codes will fail compilation.
  --iree-hal-substitute-executable-configuration=<string>                  - A `executable_name=object_file.xxx` pair specifying a hal.executable symbol name that will be substituted with the configured executable file at the given path. Configured execuable paths are relative to those specified on `--iree-hal-executable-object-search-path=`. If a `.mlir` or `.mlirbc` file is specified the entire executable will be replaced with an equivalently named hal.executable in the referenced file and otherwise the executable will be externalized and link the referenced file (`.ptx`/`.spv`/etc).
  --iree-hal-substitute-executable-configurations-from=<string>            - Substitutes any hal.executable with a file in the given path with the same name ala `--iree-hal-substitute-executable-configuration=`.
  --iree-hal-substitute-executable-object=<string>                         - A `executable_name=object_file.xxx` pair specifying a hal.executable symbol name that will be substituted with the object file at the given path. Object paths are relative to those specified on `--iree-hal-executable-object-search-path=`. If a `.mlir` or `.mlirbc` file is specified the entire executable will be replaced with an equivalently named hal.executable in the referenced file and otherwise the executable will be externalized and link the referenced file (`.ptx`/`.spv`/etc).
  --iree-hal-substitute-executable-objects-from=<string>                   - Substitutes any hal.executable with a file in the given path with the same name ala `--iree-hal-substitute-executable-object=`.
  --iree-hal-substitute-executable-source=<string>                         - A `executable_name=object_file.xxx` pair specifying a hal.executable symbol name that will be substituted with the source object file at the given path. Source object paths are relative to those specified on `--iree-hal-executable-object-search-path=`. If a `.mlir` or `.mlirbc` file is specified the entire executable will be replaced with an equivalently named hal.executable in the referenced file and otherwise the executable will be externalized and link the referenced file (`.ptx`/`.spv`/etc).
  --iree-hal-substitute-executable-sources-from=<string>                   - Substitutes any hal.executable with a file in the given path with the same name ala `--iree-hal-substitute-executable-source=`.
  --iree-hip-index-bits=<int>                                              - Set the bit width of indices in ROCm.
  --iree-linalgext-attention-softmax-max=<number>                          - maximum expected value from attention softmax
  --iree-link-bitcode=<string>                                             - Paths of additional bitcode files to load and link. Comma-separated. Any list entry that contains an equals (=) is parsed as `arch=path` and is only linked if `arch` matches the target triple.
  --iree-list-plugins                                                      - Lists all loaded plugins.
  --iree-llvmcpu-check-linalg-vectorization                                - Runs the pass to check if all the Linalg ops are vectorized
  --iree-llvmcpu-disable-arm-sme-tiling                                    - Disables tiling for SME even if it is supported by the target (i.e., when the +sme feature flag is present)
  --iree-llvmcpu-disable-distribution                                      - disable thread distribution in codegen
  --iree-llvmcpu-disable-vector-peeling                                    - Disable peeling as a pre-processing step for vectorization (only relevant when using compiler heuristics to select the strategy).
  --iree-llvmcpu-distribution-size=<int>                                   - default distribution tile size
  --iree-llvmcpu-enable-scalable-vectorization                             - Enable scalable vectorization if it is supported by the target (e.g., +sve, +sve2 and/or +sme feature flags)
  --iree-llvmcpu-enable-vector-contract-custom-kernels                     - Enables vector contract custom kernels for LLVMCPUMmt4dVectorLowering pass.
  --iree-llvmcpu-fail-on-large-vector                                      - fail if there are operations with large vectors
  --iree-llvmcpu-fail-on-out-of-bounds-stack-allocation                    - fail if the upper bound of dynamic stack allocation cannot be solved
  --iree-llvmcpu-force-arm-streaming                                       - Enables Armv9-A streaming SVE mode for any dispatch region that contains supported scalable vector operations (i.e., use SSVE rather than SVE). Requires the +sme feature flag.
  --iree-llvmcpu-general-matmul-tile-bytes=<int>                           - target distribution tile size for matrix operands of general matmuls, expressed in bytes. Currently only used in data-tiled matmuls (mmt4d).
  --iree-llvmcpu-instrument-memory-accesses                                - Instruments memory accesses in dispatches when dispatch instrumentation is enabled.
  --iree-llvmcpu-narrow-matmul-tile-bytes=<int>                            - target distribution tile size for wide matrix operand of narrow matmuls, expressed in bytes. Currently only used in data-tiled matmuls (mmt4d). Since this is only used for narrow matmuls, which traverse their wide matrix operand once, there is no reuse here and this doesn't have to be sized to fit in some CPU cache. This is more about distributing work to threads.
  --iree-llvmcpu-number-of-threads=<int>                                   - number of threads that are used at runtime if codegen thread distribution is enabled
  --iree-llvmcpu-reassociate-fp-reductions                                 - Enables reassociation for FP reductions
  --iree-llvmcpu-riscv-aggressive-distribution                             - Enable aggressive method for distribution tile size. It is only applied for linalg contraction ops now. If distConfig.minTileSizes[i] >= distConfig.maxTileSizes[i], set distConfig.maxTileSizes[i] to 2 * distConfig.minTileSizes[i].
  --iree-llvmcpu-skip-intermediate-roundings                               - Allow skipping intermediate roundings. For example, in f16 matmul kernels on targets with only f32 arithmetic, we have to perform each multiply-accumulate in f32, and if this flag is false, then we have to round those f32 accumulators to the nearest f16 every time, which is slow.
  --iree-llvmcpu-stack-allocation-assumed-vscale=<uint>                    - assumed value of vscale when checking (scalable) stack allocations
  --iree-llvmcpu-stack-allocation-limit=<int>                              - maximum allowed stack allocation size in bytes
  --iree-llvmcpu-tile-dispatch-using-forall                                - Enable tile and distribute to workgroups using scf.forall
  --iree-llvmcpu-use-decompose-softmax-fuse                                - Enables inter-pass fusion for the DecomposeSoftmax pass.
  --iree-llvmcpu-use-fast-min-max-ops                                      - Use `arith.minf/maxf` instead of `arith.minimumf/maximumf` ops
  --iree-llvmcpu-vector-pproc-strategy=<value>                             - Set the strategy for pre-processing Linalg operation before vectorization:
    =peel                                                                  -   Peel iterations from the vector dimensions so that they become multiple of the vector length
    =mask                                                                  -    Compute vector dimensions assuming vector masking support. Vector sizes may be rounded up to the nearest power of two and out-of-bounds elements would be masked-out.
    =none                                                                  -   Do not apply any vectorization pre-processing transformation.
  --iree-llvmgpu-enable-prefetch                                           - Enable prefetch in the vector distribute pipeline
  --iree-llvmgpu-enable-shared-memory-reuse                                - Enable shared memory reuse in the vector distribute pipeline
  --iree-llvmgpu-shared-memory-limit=<long>                                - specify the maximum amount of shared memory allowed to be allocated for the given target
  --iree-spirv-index-bits=<int>                                            - Set the bit width of indices in SPIR-V.
  --iree-stream-annotate-input-affinities                                  - Annotates all tensor/resource affinities on the input to the pipeline for debugging.
  --iree-stream-emulate-memset                                             - Emulate all memset types with dispatches.
  --iree-stream-external-resources-mappable                                - Allocates external resources as host-visible and mappable. This can degrade performance and introduce allocation overhead and staging buffers for readback on the host should be managed by the calling application instead.
  --iree-stream-partitioning-favor=<value>                                 - Default stream partitioning favor configuration.
    =debug                                                                 -   Force debug partitioning (no concurrency or pipelining).
    =min-peak-memory                                                       -   Favor minimizing memory consumption at the cost of additional concurrency.
    =max-concurrency                                                       -   Favor maximizing concurrency at the cost of additional memory consumption.
  --iree-stream-resource-alias-mutable-bindings                            - Fuses bindings that are mutable instead of leaving them split.
  --iree-stream-resource-index-bits=<uint>                                 - Bit width of indices used to reference resource offsets.
  --iree-stream-resource-max-allocation-size=<ulong>                       - Maximum size of an individual memory allocation.
  --iree-stream-resource-max-range=<ulong>                                 - Maximum range of a resource binding; may be less than the max allocation size.
  --iree-stream-resource-memory-model=<value>                              - Memory model used for host-device resource memory access.
    =unified                                                               -   Host and device memory are unified and there's (practically) no performance cost for cross-access.
    =discrete                                                              -   Host and device memory are discrete and cross-access is expensive.
  --iree-stream-resource-min-offset-alignment=<ulong>                      - Minimum required alignment in bytes for resource offsets.
  --iree-util-hoist-into-globals-print-constexpr-dotgraph-to=<filename>    - Prints a dot graph representing the const-expr analysis. The red nodes represent roots and the green nodes represent hoisted values.
  --iree-util-zero-fill-elided-attrs                                       - Fills elided attributes with zeros when serializing.
  --iree-vm-c-module-optimize                                              - Optimizes the VM module with CSE/inlining/etc prior to serialization
  --iree-vm-c-module-output-format=<value>                                 - Output format used to write the C module
    =code                                                                  -   C Code file
    =mlir-text                                                             -   MLIR module file in the VM and EmitC dialects
  --iree-vm-c-module-strip-debug-ops                                       - Strips debug-only ops from the module
  --iree-vmvx-enable-ukernels-decompose-linalg-generic                     - Enables decomposition of linalg.generic ops when ukernels are enabled (experimental)
  --iree-vmvx-skip-intermediate-roundings                                  - Allow skipping intermediate roundings. For example, in f16 matmul kernels on targets with only f32 arithmetic, we have to perform each multiply-accumulate in f32, and if this flag is false, then we have to round those f32 accumulators to the nearest f16 every time, which is slow.
  --iree-vulkan-experimental-indirect-bindings                             - Force indirect bindings for all generated dispatches.
  --iree-vulkan-target=<string>                                            - Vulkan target controlling the SPIR-V environment. Given the wide support of Vulkan, this option supports a few schemes: 1) LLVM CodeGen backend style: e.g., 'gfx*' for AMD GPUs and 'sm_*' for NVIDIA GPUs; 2) architecture code name style: e.g., 'rdna3'/'valhall4'/'ampere'/'adreno' for AMD/ARM/NVIDIA/Qualcomm GPUs; 3) product name style: 'rx7900xtx'/'rtx4090' for AMD/NVIDIA GPUs. See https://iree.dev/guides/deployment-configurations/gpu-vulkan/ for more details.
  --iterative-counter-promotion                                            - Allow counter promotion across the whole loop nest.
  --lint-abort-on-error                                                    - In the Lint pass, abort on errors.
  --load=<pluginfilename>                                                  - Load the specified plugin
  --log-actions-to=<string>                                                - Log action execution to a file, or stderr if  '-' is passed
  --log-mlir-actions-filter=<string>                                       - Comma separated list of locations to filter actions from logging
  --lower-allow-check-percentile-cutoff-hot=<int>                          - Hot percentile cuttoff.
  --lower-allow-check-random-rate=<number>                                 - Probability value in the range [0.0, 1.0] of unconditional pseudo-random checks.
  --lto-embed-bitcode=<value>                                              - Embed LLVM bitcode in object files produced by LTO
    =none                                                                  -   Do not embed
    =optimized                                                             -   Embed after all optimization passes
    =post-merge-pre-opt                                                    -   Embed post merge, but before optimizations
  --matrix-default-layout=<value>                                          - Sets the default matrix layout
    =column-major                                                          -   Use column-major layout
    =row-major                                                             -   Use row-major layout
  --matrix-print-after-transpose-opt                                       - 
  --max-counter-promotions=<int>                                           - Max number of allowed counter promotions
  --max-counter-promotions-per-loop=<uint>                                 - Max number counter promotions per loop to avoid increasing register pressure too much
  --mir-strip-debugify-only                                                - Should mir-strip-debug only strip debug info from debugified modules by default
  --misexpect-tolerance=<uint>                                             - Prevents emitting diagnostics when profile counts are within N% of the threshold..
  --mlir-disable-threading                                                 - Disable multi-threading within MLIR, overrides any further call to MLIRContext::enableMultiThreading()
  --mlir-elide-elementsattrs-if-larger=<uint>                              - Elide ElementsAttrs with "..." that have more elements than the given upper limit
  --mlir-elide-resource-strings-if-larger=<uint>                           - Elide printing value of resources if string is too long in chars.
  --mlir-output-format=<value>                                             - Output format for timing data
    =text                                                                  -   display the results in text format
    =json                                                                  -   display the results in JSON format
  --mlir-pass-pipeline-crash-reproducer=<string>                           - Generate a .mlir reproducer file at the given output path if the pass manager crashes or fails
  --mlir-pass-pipeline-local-reproducer                                    - When generating a crash reproducer, attempt to generated a reproducer with the smallest pipeline.
  --mlir-pass-statistics                                                   - Display the statistics of each pass
  --mlir-pass-statistics-display=<value>                                   - Display method for pass statistics
    =list                                                                  -   display the results in a merged list sorted by pass name
    =pipeline                                                              -   display the results with a nested pipeline view
  --mlir-pretty-debuginfo                                                  - Print pretty debug info in MLIR output
  --mlir-print-debuginfo                                                   - Print debug info in MLIR output
  --mlir-print-elementsattrs-with-hex-if-larger=<long>                     - Print DenseElementsAttrs with a hex string that have more elements than the given upper limit (use -1 to disable)
  --mlir-print-ir-after=<pass-arg>                                         - Print IR after specified passes
  --mlir-print-ir-after-all                                                - Print IR after each pass
  --mlir-print-ir-after-change                                             - When printing the IR after a pass, only print if the IR changed
  --mlir-print-ir-after-failure                                            - When printing the IR after a pass, only print if the pass failed
  --mlir-print-ir-before=<pass-arg>                                        - Print IR before specified passes
  --mlir-print-ir-before-all                                               - Print IR before each pass
  --mlir-print-ir-module-scope                                             - When printing IR for print-ir-[before|after]{-all} always print the top-level operation
  --mlir-print-ir-tree-dir=<string>                                        - When printing the IR before/after a pass, print file tree rooted at this directory. Use in conjunction with mlir-print-ir-* flags
  --mlir-print-local-scope                                                 - Print with local scope and inline information (eliding aliases for attributes, types, and locations
  --mlir-print-op-on-diagnostic                                            - When a diagnostic is emitted on an operation, also print the operation as an attached note
  --mlir-print-skip-regions                                                - Skip regions when printing ops.
  --mlir-print-stacktrace-on-diagnostic                                    - When a diagnostic is emitted, also print the stack trace as an attached note
  --mlir-print-unique-ssa-ids                                              - Print unique SSA ID numbers for values, block arguments and naming conflicts across all regions
  --mlir-print-value-users                                                 - Print users of operation results and block arguments as a comment
  --mlir-timing                                                            - Display execution times
  --mlir-timing-display=<value>                                            - Display method for timing data
    =list                                                                  -   display the results in a list sorted by total time
    =tree                                                                  -   display the results ina with a nested tree view
  --mlir-use-nameloc-as-prefix                                             - Print SSA IDs using NameLocs as prefixes
  --no-discriminators                                                      - Disable generation of discriminator information.
  --nvptx-sched4reg                                                        - NVPTX Specific: schedule for register pressue
  --object-size-offset-visitor-max-visit-instructions=<uint>               - Maximum number of instructions for ObjectSizeOffsetVisitor to look at
  --pgo-block-coverage                                                     - Use this option to enable basic block coverage instrumentation
  --pgo-temporal-instrumentation                                           - Use this option to enable temporal instrumentation
  --pgo-view-block-coverage-graph                                          - Create a dot file of CFGs with block coverage inference information
  --print-pipeline-passes                                                  - Print a '-passes' compatible string describing the pipeline (best-effort only).
  --profile-actions-to=<string>                                            - Profile action execution to a file, or stderr if  '-' is passed
  --profile-correlate=<value>                                              - Use debug info or binary file to correlate profiles.
    =<empty>                                                               -   No profile correlation
    =debug-info                                                            -   Use debug info to correlate
    =binary                                                                -   Use binary to correlate
  --promote-alloca-vector-loop-user-weight=<uint>                          - The bonus weight of users of allocas within loop when sorting profitable allocas
  --r600-ir-structurize                                                    - Use StructurizeCFG IR pass
  --riscv-add-build-attributes                                             - 
  --riscv-use-aa                                                           - Enable the use of AA during codegen.
  --runtime-counter-relocation                                             - Enable relocating counters at runtime.
  --safepoint-ir-verifier-print-only                                       - 
  --sample-profile-check-record-coverage=<N>                               - Emit a warning if less than N% of records in the input profile are matched to the IR.
  --sample-profile-check-sample-coverage=<N>                               - Emit a warning if less than N% of samples in the input profile are matched to the IR.
  --sample-profile-max-propagate-iterations=<uint>                         - Maximum number of iterations to go through when propagating sample block/edge weights through the CFG.
  --sampled-instr-burst-duration=<uint>                                    - Set the profile instrumentation burst duration, which can range from 1 to the value of 'sampled-instr-period' (0 is invalid). This number of samples will be recorded for each 'sampled-instr-period' count update. Setting to 1 enables simple sampling, in which case it is recommended to set 'sampled-instr-period' to a prime number.
  --sampled-instr-period=<uint>                                            - Set the profile instrumentation sample period. A sample period of 0 is invalid. For each sample period, a fixed number of consecutive samples will be recorded. The number is controlled by 'sampled-instr-burst-duration' flag. The default sample period of 65536 is optimized for generating efficient code that leverages unsigned short integer wrapping in overflow, but this is disabled under simple sampling (burst duration = 1).
  --sampled-instrumentation                                                - Do PGO instrumentation sampling
  --skip-ret-exit-block                                                    - Suppress counter promotion if exit blocks contain ret.
  --speculative-counter-promotion-max-exiting=<uint>                       - The max number of exiting blocks of a loop to allow  speculative counter promotion
  --speculative-counter-promotion-to-loop                                  - When the option is false, if the target block is in a loop, the promotion will be disallowed unless the promoted counter  update can be further/iteratively promoted into an acyclic  region.
  --split-input-file                                                       - Split the input file into pieces and process each chunk independently.
  --summary-file=<string>                                                  - The summary file to use for function importing.
  --sve-tail-folding=<string>                                              - Control the use of vectorisation using tail-folding for SVE where the option is specified in the form (Initial)[+(Flag1|Flag2|...)]:
                                                                             disabled      (Initial) No loop types will vectorize using tail-folding
                                                                             default       (Initial) Uses the default tail-folding settings for the target CPU
                                                                             all           (Initial) All legal loop types will vectorize using tail-folding
                                                                             simple        (Initial) Use tail-folding for simple loops (not reductions or recurrences)
                                                                             reductions    Use tail-folding for loops containing reductions
                                                                             noreductions  Inverse of above
                                                                             recurrences   Use tail-folding for loops containing fixed order recurrences
                                                                             norecurrences Inverse of above
                                                                             reverse       Use tail-folding for loops requiring reversed predicates
                                                                             noreverse     Inverse of above
  --tail-predication=<value>                                               - MVE tail-predication pass options
    =disabled                                                              -   Don't tail-predicate loops
    =enabled-no-reductions                                                 -   Enable tail-predication, but not for reduction loops
    =enabled                                                               -   Enable tail-predication, including reduction loops
    =force-enabled-no-reductions                                           -   Enable tail-predication, but not for reduction loops, and force this which might be unsafe
    =force-enabled                                                         -   Enable tail-predication, including reduction loops, and force this which might be unsafe
  --thinlto-assume-merged                                                  - Assume the input has already undergone ThinLTO function importing and the other pre-optimization pipeline changes.
  --type-based-intrinsic-cost                                              - Calculate intrinsics cost based only on argument types
  --verify                                                                 - Verifies the IR for correctness throughout compilation.
  --verify-region-info                                                     - Verify region info (time consuming)
  --vp-counters-per-site=<number>                                          - The average number of profile counters allocated per value profiling site.
  --vp-static-alloc                                                        - Do static counter allocation for value profiler
  --wasm-enable-eh                                                         - WebAssembly exception handling
  --wasm-enable-exnref                                                     - WebAssembly exception handling (exnref)
  --wasm-enable-sjlj                                                       - WebAssembly setjmp/longjmp handling
  --wholeprogramdevirt-cutoff=<uint>                                       - Max number of devirtualizations for devirt module pass
  --write-experimental-debuginfo                                           - Write debug info in the new non-intrinsic format. Has no effect if --preserve-input-debuginfo-format=true.
  --x86-align-branch=<string>                                              - Specify types of branches to align (plus separated list of types):
                                                                             jcc      indicates conditional jumps
                                                                             fused    indicates fused conditional jumps
                                                                             jmp      indicates direct unconditional jumps
                                                                             call     indicates direct and indirect calls
                                                                             ret      indicates rets
                                                                             indirect indicates indirect unconditional jumps
  --x86-align-branch-boundary=<uint>                                       - Control how the assembler should align branches with NOP. If the boundary's size is not 0, it should be a power of 2 and no less than 32. Branches will be aligned to prevent from being across or against the boundary of specified size. The default value 0 does not align branches.
  --x86-branches-within-32B-boundaries                                     - Align selected instructions to mitigate negative performance impact of Intel's micro code update for errata skx102.  May break assumptions about labels corresponding to particular instructions, and should be used with caution.
  --x86-pad-max-prefix-size=<uint>                                         - Maximum number of prefixes to use for padding

Generic Options:

  --help                                                                   - Display available options (--help-hidden for more)
  --help-list                                                              - Display list of available options (--help-list-hidden for more)
  --version                                                                - Display the version of this program

HIP HAL Target:

  --iree-hip-bc-dir=<string>                                               - Directory of HIP Bitcode.
  --iree-hip-enable-ukernels=<string>                                      - Enables microkernels in the HIP compiler backend. May be `default`, `none`, `all`, or a comma-separated list of specific unprefixed microkernels to enable, e.g. `mmt4d`.
  --iree-hip-legacy-sync                                                   - Enables 'legacy-sync' mode, which is required for inline execution.
  --iree-hip-llvm-global-isel                                              - Enable global instruction selection in llvm.
  --iree-hip-llvm-slp-vec                                                  - Enable slp vectorization in llvm opt.
  --iree-hip-pass-plugin-path=<string>                                     - LLVM pass plugins are out of tree libraries that implement LLVM opt passes. The library paths passed in this flag are to be passed to the target backend compiler during HIP executable serialization
  --iree-hip-target=<string>                                               - HIP target as expected by LLVM AMDGPU backend; e.g., 'gfx90a'/'gfx942' for targeting MI250/MI300 GPUs. Additionally this also supports architecture code names like 'cdna3'/'rdna3' or some product names like 'mi300x'/'rtx7900xtx' for a better experience. See https://iree.dev/guides/deployment-configurations/gpu-rocm/ for more details.
  --iree-hip-target-features=<string>                                      - HIP target features as expected by LLVM AMDGPU backend; e.g., '+sramecc,+xnack'.
  --iree-hip-waves-per-eu=<int>                                            - Optimization hint specifying minimum number of waves per execution unit.

IREE HAL executable target options:

  --iree-hal-default-device=<string>                                       - Which device is considered the default when no device affinity is specified. Either the device name when names are specified or the numeric ordinal of the device.
  --iree-hal-dump-executable-benchmarks-to=<string>                        - Path to write standalone hal.executable benchmarks into (- for stdout).
  --iree-hal-dump-executable-binaries-to=<string>                          - Path to write translated and serialized executable binaries into.
  --iree-hal-dump-executable-configurations-to=<string>                    - Path to write individual hal.executable input source listings into, after translation strategy selection and before starting translation (- for stdout).
  --iree-hal-dump-executable-files-to=<string>                             - Meta flag for all iree-hal-dump-executable-* options. Path to write executable files (sources, benchmarks, intermediates, binaries) to.
  --iree-hal-dump-executable-intermediates-to=<string>                     - Path to write translated executable intermediates (.bc, .o, etc) into.
  --iree-hal-dump-executable-sources-to=<string>                           - Path to write individual hal.executable input source listings into (- for stdout).
  --iree-hal-executable-debug-level=<int>                                  - Debug level for executable translation (0-3)
  --iree-hal-target-backends=<string>                                      - Target backends for executable compilation.
  --iree-hal-target-device=<string>                                        - Target device specifications.

IREE HAL local device options:

  --iree-hal-local-host-device-backends=<string>                           - Default host backends for local device executable compilation.
  --iree-hal-local-target-device-backends=<string>                         - Default target backends for local device executable compilation.

IREE Main Options:

  --compile-mode=<value>                                                   - IREE compilation mode
    =std                                                                   -   Standard compilation
    =vm                                                                    -   Compile from VM IR
    =hal-executable                                                        -   Compile an MLIR module containing a single hal.executable into a target-specific binary form (such as an ELF file or a flatbuffer containing a SPIR-V blob)
    =precompile                                                            -   Precompilation pipeline which does input conversion and global optimizations.
  -o <filename>                                                            - Output filename
  --output-format=<value>                                                  - Format of compiled output
    =vm-bytecode                                                           -   IREE VM Bytecode (default)
    =vm-c                                                                  -   C source module
    =vm-asm                                                                -   IREE VM MLIR Assembly

IREE VM bytecode options:

  --iree-vm-bytecode-module-optimize                                       - Optimizes the VM module with CSE/inlining/etc prior to serialization
  --iree-vm-bytecode-module-output-format=<value>                          - Output format the bytecode module is written in
    =flatbuffer-binary                                                     -   Binary FlatBuffer file
    =flatbuffer-text                                                       -   Text FlatBuffer file, debug-only
    =mlir-text                                                             -   MLIR module file in the VM dialect
    =annotated-mlir-text                                                   -   MLIR module file in the VM dialect with annotations
  --iree-vm-bytecode-module-strip-debug-ops                                - Strips debug-only ops from the module
  --iree-vm-bytecode-module-strip-source-map                               - Strips the source map from the module
  --iree-vm-bytecode-source-listing=<string>                               - Dump a VM MLIR file and annotate source locations with it
  --iree-vm-emit-polyglot-zip                                              - Enables output files to be viewed as zip files for debugging (only applies to binary targets)

IREE VM target options:

  --iree-vm-target-extension-f32                                           - Support f32 target opcode extensions.
  --iree-vm-target-extension-f64                                           - Support f64 target opcode extensions.
  --iree-vm-target-index-bits=<int>                                        - Bit width of index types.
  --iree-vm-target-optimize-for-stack-size                                 - Prefer optimizations that reduce VM stack usage over performance.
  --iree-vm-target-truncate-unsupported-floats                             - Truncate f64 to f32 when unsupported.

IREE compiler plugin options:

  --iree-plugin=<string>                                                   - Plugins to activate
  --iree-print-plugin-info                                                 - Prints available and activated plugin info to stderr

IREE options for apply custom preprocessing before normal IREE compilation flow:

  --iree-preprocessing-pass-pipeline=<string>                              - Textual description of the pass pipeline to run before running normal IREE compilation pipelines.
  --iree-preprocessing-pdl-spec-filename=<string>                          - File name of a transform dialect spec to use for preprocessing.
  --iree-preprocessing-transform-spec-filename=<string>                    - File name of a transform dialect spec to use for preprocessing.

IREE options for controlling global optimizations.:

  --iree-opt-aggressively-propagate-transposes                             - Propagates transposes to named ops even when the resulting op will be a linalg.generic
  --iree-opt-const-eval                                                    - Enables eager evaluation of constants using the full compiler and runtime (on by default).
  --iree-opt-const-expr-hoisting                                           - Hoists the results of latent constant expressions into immutable global initializers for evaluation at program load.
  --iree-opt-const-expr-max-size-increase-threshold=<long>                 - Maximum byte size increase allowed for constant expr hoisting policy to allow hoisting.
  --iree-opt-data-tiling                                                   - Enables data tiling path.
  --iree-opt-export-parameter-minimum-size=<long>                          - Minimum size of constants to export to the archive created in `iree-opt-export-parameter-archive-export-file`.
  --iree-opt-export-parameters=<string>                                    - File path to an archive to export parameters to with an optional `scope=` prefix.
  --iree-opt-import-parameter-keys=<string>                                - List of parameter keys to import.
  --iree-opt-import-parameter-maximum-size=<long>                          - Maximum size of parameters to import.
  --iree-opt-import-parameters=<string>                                    - File paths to archives to import parameters from with an optional `scope=` prefix.
  --iree-opt-numeric-precision-reduction                                   - Reduces numeric precision to lower bit depths where possible.
  --iree-opt-outer-dim-concat                                              - Transposes all concatenations to happenalong the outer most dimension.
  --iree-opt-splat-parameters=<string>                                     - File path to create a parameter archive of splat values out of all parameter backed globals.
  --iree-opt-strip-assertions                                              - Strips debug assertions after any useful information has been extracted.

IREE options for controlling host/device scheduling.:

  --iree-execution-model=<value>                                           - Specifies the execution model used for scheduling tensor compute operations.
    =host-only                                                             -   Host-local code only that does not need execution scheduling.
    =async-internal                                                        -   Full HAL using asynchronous host/device execution internally but exporting functions as if synchronous.
    =async-external                                                        -   Full HAL using asynchronous host/device execution both internally and externally.
    =inline-static                                                         -   Inline host-local in-process execution with executable code statically linked into the host program.
    =inline-dynamic                                                        -   Inline host-local in-process execution using dynamic executables.
  --iree-scheduling-dump-statistics-file=<string>                          - File path to write statistics to; or `` for stderr or `-` for stdout.
  --iree-scheduling-dump-statistics-format=<value>                         - Dumps statistics in the specified output format.
    =pretty                                                                -   Human-readable pretty printed output.
    =verbose                                                               -   Pretty printed output with additional IR.
    =csv                                                                   -   Comma separated values.
    =json                                                                  -   JSON output with structures for data exchange
  --iree-scheduling-optimize-bindings                                      - Enables binding fusion and dispatch site specialization.

IREE options for controlling the input transformations to apply.:

  --iree-input-demote-f32-to-f16                                           - Converts all f32 ops and values into f16 counterparts unconditionally before main global optimizations.
  --iree-input-demote-f64-to-f32                                           - Converts all f64 ops and values into f32 counterparts unconditionally before main global optimizations.
  --iree-input-demote-i64-to-i32                                           - Converts all i64 ops and values into i32 counterparts unconditionally before main global optimizations.
  --iree-input-promote-bf16-to-f32                                         - Converts all bf16 ops and values into f32 counterparts unconditionally before main global optimizations.
  --iree-input-promote-f16-to-f32                                          - Converts all f16 ops and values into f32 counterparts unconditionally before main global optimizations.
  --iree-input-type=<string>                                               - Specifies the input program representation:
                                                                               =none          - No input dialect transformation.
                                                                               =auto          - Analyze the input program to choose conversion.
                                                                               =*             - An extensible input type defined in a plugin.

IREE translation binding support options.:

  --iree-native-bindings-support                                           - Include runtime support for native IREE ABI-compatible bindings.
  --iree-tflite-bindings-support                                           - Include runtime support for the IREE TFLite compatibility bindings.

LLVMCPU HAL Target:

  --iree-llvmcpu-debug-symbols                                             - Generate and embed debug information (DWARF, PDB, etc)
  --iree-llvmcpu-embedded-linker-path=<string>                             - Tool used to link embedded ELFs produced by IREE (for --iree-llvmcpu-link-embedded=true).
  --iree-llvmcpu-enable-ukernels=<string>                                  - Enables ukernels in the llvmcpu backend. May be `default`, `none`, `all`, or a comma-separated list of specific unprefixed ukernels to enable, e.g. `mmt4d`.
  --iree-llvmcpu-keep-linker-artifacts                                     - Keep LLVM linker target artifacts (.so/.dll/etc)
  --iree-llvmcpu-link-embedded                                             - Links binaries into a platform-agnostic ELF to be loaded by the embedded IREE ELF loader.
  --iree-llvmcpu-link-static                                               - Links system libraries into binaries statically to isolate them from platform dependencies needed at runtime
  --iree-llvmcpu-link-ukernel-bitcode                                      - Link ukernel bitcode libraries into generated executables
  --iree-llvmcpu-list-targets                                              - Lists all registered targets that the LLVM backend can generate code for.
  --iree-llvmcpu-loop-interleaving                                         - Enable LLVM loop interleaving opt
  --iree-llvmcpu-loop-unrolling                                            - Enable LLVM loop unrolling opt
  --iree-llvmcpu-loop-vectorization                                        - Enable LLVM loop vectorization opt
  --iree-llvmcpu-sanitize=<value>                                          - Apply LLVM sanitize feature
    =address                                                               -   Address sanitizer support
    =thread                                                                -   Thread sanitizer support
  --iree-llvmcpu-slp-vectorization                                         - Enable LLVM SLP Vectorization opt
  --iree-llvmcpu-static-library-output-path=<string>                       - Path to output static object (EX: '/path/to/static-library.o'). This will produce the static library at the specified path along with a similarly named '.h' file for static linking.
  --iree-llvmcpu-system-linker-path=<string>                               - Tool used to link system shared libraries produced by IREE (for --iree-llvmcpu-link-embedded=false).
  --iree-llvmcpu-target-abi=<string>                                       - LLVM target machine ABI; specify for -mabi
  --iree-llvmcpu-target-cpu=<string>                                       - LLVM target machine CPU; use 'host' for your host native CPU.
  --iree-llvmcpu-target-cpu-features=<string>                              - LLVM target machine CPU features; use 'host' for your host native CPU.
  --iree-llvmcpu-target-data-layout=<string>                               - LLVM target machine data layout override.
  --iree-llvmcpu-target-float-abi=<value>                                  - LLVM target codegen enables soft float abi e.g -mfloat-abi=softfp
    =default                                                               -   Default (softfp)
    =soft                                                                  -   Software floating-point emulation
    =hard                                                                  -   Hardware floating-point instructions
  --iree-llvmcpu-target-triple=<string>                                    - LLVM target machine triple.
  --iree-llvmcpu-target-vector-width-in-bytes=<uint>                       - Overrides the native vector register width (in bytes) of the target.
  --iree-llvmcpu-wasm-linker-path=<string>                                 - Tool used to link WebAssembly modules produced by IREE (for --iree-llvmcpu-target-triple=wasm32-*).

MetalSPIRV HAL Target:

  --iree-metal-compile-to-metallib                                         - Compile to .metallib and embed in IREE deployable flatbuffer if true; otherwise stop at and embed MSL source code
  --iree-metal-target-platform=<value>                                     - Apple platform to target
    =macos                                                                 -   macOS platform
    =ios                                                                   -   iOS platform
    =ios-simulator                                                         -   iOS simulator platform

Torch Input:

  --iree-torch-decompose-complex-ops                                       - Decompose complex torch operations.
  --iree-torch-use-strict-symbolic-shapes                                  - Forces dynamic shapes to be treated as strict

VMVX HAL Target:

  --iree-vmvx-enable-microkernels                                          - Enables microkernel lowering for vmvx (experimental)

```
