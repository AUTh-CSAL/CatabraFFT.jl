# Practical Integration: SIMD FFT with Existing Code

using SIMD

# Include the SIMD implementation
include("fft_simd_complete.jl")

"""
Enhanced load_real_imag_gen with SIMD support.
Generates code for loading complex numbers optimally based on access pattern.
"""
function load_real_imag_gen_simd(t; mode, T, ptr_name="px", SIMD_BITS=256)
    n = length(t)
    complex_size_bits = 2 * sizeof(T) * 8
    complexes_per_vec = SIMD_BITS ÷ complex_size_bits
    
    # Extract indices from variable names
    indices = Int[]
    for s in t
        m = match(r"(\d+)\D*$", s)
        if m !== nothing
            push!(indices, parse(Int, m.captures[1]))
        end
    end
    
    # Check if contiguous
    is_contiguous = length(indices) > 1 && 
                    all(i -> indices[i] == indices[1] + i - 1, 2:length(indices))
    
    if mode == :vload_simd
        if is_contiguous && n <= complexes_per_vec
            # Fast path: single vload for contiguous data
            start_idx = 2 * (indices[1] - 1) + 1  # First float index (1-based)
            n_floats = 2 * n
            
            code = """
            # SIMD load: $n contiguous complex numbers
            v_data = vload(Vec{$(n_floats), $T}, $ptr_name, $start_idx)
            """
            
            # Extract individual complex numbers from vector
            for (i, idx) in enumerate(indices)
                real_pos = 2 * (i - 1)
                imag_pos = 2 * (i - 1) + 1
                var = startswith(t[i], "x") ? "x" : startswith(t[i], "y") ? "y" : "t"
                code *= "$(var)$(idx)_r, $(var)$(idx)_i = v_data[$real_pos], v_data[$imag_pos]\n"
            end
            
            return code
            
        elseif n <= complexes_per_vec
            # Gather path for non-contiguous data
            float_indices = Int[]
            for idx in indices
                push!(float_indices, 2*idx - 1)  # real (1-based)
                push!(float_indices, 2*idx)      # imag (1-based)
            end
            
            n_floats = 2 * n
            idx_tuple = "(" * join(float_indices, ", ") * ")"
            
            code = """
            # SIMD gather: $n non-contiguous complex numbers
            gather_idx = Vec{$(n_floats), Int}($idx_tuple)
            v_data = vgather($ptr_name, gather_idx)
            """
            
            # Extract values
            for (i, idx) in enumerate(indices)
                real_pos = 2 * (i - 1)
                imag_pos = 2 * (i - 1) + 1
                var = startswith(t[i], "x") ? "x" : startswith(t[i], "y") ? "y" : "t"
                code *= "$(var)$(idx)_r, $(var)$(idx)_i = v_data[$real_pos], v_data[$imag_pos]\n"
            end
            
            return code
        else
            # Multiple vectors needed - chunk the loads
            code = "# Multi-vector SIMD load\n"
            for chunk_start in 1:complexes_per_vec:n
                chunk_end = min(chunk_start + complexes_per_vec - 1, n)
                chunk_indices = indices[chunk_start:chunk_end]
                chunk_vars = t[chunk_start:chunk_end]
                
                chunk_code = load_real_imag_gen_simd(chunk_vars; 
                    mode=mode, T=T, ptr_name=ptr_name, SIMD_BITS=SIMD_BITS)
                code *= chunk_code * "\n"
            end
            
            return code
        end
        
    elseif mode == :unsafe_load
        # Fall back to scalar loads
        join([
            let
                m = match(r"(\d+)\D*$", s)
                num = parse(Int, m.captures[1])
                var = startswith(s, "x") ? "x" :
                      startswith(s, "y") ? "y" :
                      startswith(s, "D") ? "d" : error("Unknown input: $s")
                prefix = i == 1 ? "" : " "
                idx_r = 2*num - 1
                idx_i = 2*num
                "$(prefix)$(var)$(num)_r, $(prefix)$(var)$(num)_i = unsafe_load($ptr_name, $idx_r), unsafe_load($ptr_name, $idx_i)"
            end
            for (i, s) in enumerate(t)
        ], "; ")
        
    else
        # Default mode
        join([
            let
                m = match(r"(\d+)\D*$", s)
                num = m.captures[1]
                var = startswith(s, "x") ? "x" :
                      startswith(s, "y") ? "y" :
                      startswith(s, "D") ? "d" : error("Unknown input: $s")
                rhs = occursin('[', s) ? replace(s, " " => "") : "$var[$num]"
                prefix = i == 1 ? "" : " "
                "$(prefix)$(var)$(num)_r , $(prefix)$(var)$(num)_i = real($rhs), imag($rhs)"
            end
            for (i, s) in enumerate(t)
        ], "; ")
    end
end

"""
Enhanced store operation with SIMD support.
Generates code for storing complex numbers optimally.
"""
function store_real_imag_gen_simd(y_vars, vals; mode, T, ptr_name="py", SIMD_BITS=256)
    n = length(y_vars)
    complex_size_bits = 2 * sizeof(T) * 8
    complexes_per_vec = SIMD_BITS ÷ complex_size_bits
    
    # Extract indices from y variable names
    indices = Int[]
    for var in y_vars
        m = match(r"\[(\d+)\]", var)
        if m === nothing
            m = match(r"(\d+)", var)
        end
        if m !== nothing
            push!(indices, parse(Int, m.captures[1]))
        end
    end
    
    # Check if contiguous
    is_contiguous = length(indices) > 1 && 
                    all(i -> indices[i] == indices[1] + i - 1, 2:length(indices))
    
    if mode == :vstore_simd
        if is_contiguous && n <= complexes_per_vec
            # Fast path: single vstore for contiguous data
            start_idx = 2 * (indices[1] - 1) + 1  # First float index (1-based)
            n_floats = 2 * n
            
            # Build value tuple from vals (could be variables or expressions)
            val_parts = String[]
            for val in vals
                if contains(val, "_r") || contains(val, "_i")
                    # Already split
                    push!(val_parts, val)
                else
                    # Need to extract real/imag
                    push!(val_parts, "$(val)_r", "$(val)_i")
                end
            end
            
            vals_str = "(" * join(val_parts, ", ") * ")"
            
            return """
            # SIMD store: $n contiguous complex numbers
            v_out = Vec{$(n_floats), $T}($vals_str)
            vstore(v_out, $ptr_name, $start_idx)
            """
            
        elseif n <= complexes_per_vec
            # Scatter path for non-contiguous data
            float_indices = Int[]
            for idx in indices
                push!(float_indices, 2*idx - 1)  # real (1-based)
                push!(float_indices, 2*idx)      # imag (1-based)
            end
            
            n_floats = 2 * n
            idx_tuple = "(" * join(float_indices, ", ") * ")"
            
            val_parts = String[]
            for val in vals
                push!(val_parts, "$(val)_r", "$(val)_i")
            end
            vals_str = "(" * join(val_parts, ", ") * ")"
            
            return """
            # SIMD scatter: $n non-contiguous complex numbers
            scatter_idx = Vec{$(n_floats), Int}($idx_tuple)
            v_out = Vec{$(n_floats), $T}($vals_str)
            vscatter(v_out, $ptr_name, scatter_idx)
            """
        else
            # Multiple vectors
            code = "# Multi-vector SIMD store\n"
            for chunk_start in 1:complexes_per_vec:n
                chunk_end = min(chunk_start + complexes_per_vec - 1, n)
                chunk_y = y_vars[chunk_start:chunk_end]
                chunk_vals = vals[chunk_start:chunk_end]
                
                chunk_code = store_real_imag_gen_simd(chunk_y, chunk_vals;
                    mode=mode, T=T, ptr_name=ptr_name, SIMD_BITS=SIMD_BITS)
                code *= chunk_code * "\n"
            end
            
            return code
        end
    else
        # Fall back to scalar stores
        code_parts = String[]
        for (i, (y_var, val)) in enumerate(zip(y_vars, vals))
            push!(code_parts, "$y_var = $val")
        end
        return join(code_parts, "\n")
    end
end

"""
Complete integration: Enhanced makefftradix with SIMD support.
This replaces the original makefftradix to use SIMD operations.
"""
function makefftradix_simd(n::Int, suffixes, D::AbstractArray{String}, 
                          p::Int, op, SIZE::Int, ::Type{T}, 
                          SIMD_BITS::Int=256) where T <: AbstractFloat
    
    # Determine mode based on size and access pattern
    mode = if n <= 8 && SIMD_BITS >= 256
        :vload_simd  # Use SIMD for small kernels
    else
        :unsafe_load  # Fall back to scalar for complex patterns
    end
    
    has_y = has_flag(suffixes, Y)
    has_vec = has_flag(suffixes, VEC)
    
    input = op.eo ? "y" : "x"
    output = !has_y && op.eo ? "x" : "y"
    
    radix = n
    stride = op.stride
    n_groups = op.n_groups
    input_spacing = SIZE ÷ radix
    
    # Setup pointers for SIMD loads
    input_ptr = mode == :vload_simd ? "$(input)_ptr" : "$(input)_floats"
    output_ptr = "$(output)_floats"
    
    # Input indexing
    x = if mode == :vload_simd
        ["$(input)[$(p + 1 + (i-1)*input_spacing)]" for i in 1:radix]
    else
        ["$(input)[$(2*(p + 1 + (i-1)*input_spacing) - 1 + j)]" for i in 1:radix for j in 0:1]
    end
    
    # Output indexing
    base = (p ÷ stride) * (stride * radix) + (p % stride)
    y = if mode == :vload_simd
        ["$(output)[$(base + 1 + i*stride)]" for i in 0:radix-1]
    else
        ["$(output_ptr)[$(2*(base + 1 + i*stride) - 1 + j)]" for i in 0:radix-1 for j in 0:1]
    end
    
    d = D == String[] ? nothing : D
    
    # Pointer setup
    px = if mode == :vload_simd
        "$(input_ptr) = pointer(reinterpret($T, $(input)));"
    else
        "$(input_ptr) = reinterpret($T, $(input));"
    end
    
    py = "$(output_ptr) = reinterpret($T, $(output));"
    
    # Generate kernel code
    kernel_code = if mode == :vload_simd
        # Use SIMD version
        recfft2_simd(y, x, d, nothing, true, T, 1, SIMD_BITS)
    else
        # Use original version
        recfft2(y, x, d, nothing, true, T, 1, mode, py)
    end
    
    kernel_code = "$px\n$py\n$kernel_code"
    
    if isempty(kernel_code)
        return quote end
    else
        try
            parsed_expr = Meta.parse("begin\n$kernel_code\nend")
            return parsed_expr
        catch e
            @warn "Failed to parse kernel code: $e"
            @warn "Kernel code was: $kernel_code"
            return quote
                copyto!(y, x)
            end
        end
    end
end

"""
Benchmark comparison: SIMD vs Scalar implementations
"""
function benchmark_simd_vs_scalar(::Type{T}=Float64, n::Int=1024) where T
    using BenchmarkTools
    
    # Generate test data
    x = randn(Complex{T}, n)
    y_scalar = similar(x)
    y_simd = similar(x)
    
    println("Benchmarking FFT size $n with element type $T")
    println("="^60)
    
    # Scalar version timing
    println("\nScalar (original) implementation:")
    t_scalar = @benchmark fft_scalar!($y_scalar, $x)
    display(t_scalar)
    
    # SIMD version timing
    println("\n\nSIMD (optimized) implementation:")
    t_simd = @benchmark fft_simd!($y_simd, $x)
    display(t_simd)
    
    # Speedup calculation
    speedup = median(t_scalar).time / median(t_simd).time
    println("\n\nSpeedup: $(round(speedup, digits=2))×")
    
    # Verify correctness
    max_error = maximum(abs.(y_scalar .- y_simd))
    println("Maximum error: $max_error")
    
    return (scalar=t_scalar, simd=t_simd, speedup=speedup)
end

"""
Profile instruction mix to understand SIMD utilization
"""
function profile_simd_instructions(kernel_func, ::Type{T}=Float64) where T
    # This would use perf or similar tools to analyze:
    # - Number of ymm/zmm register instructions
    # - FMA utilization
    # - Gather/scatter vs load/store ratio
    # - Register spills
    
    println("Instruction profile for SIMD kernel:")
    println("="^60)
    println("TODO: Integrate with performance counters")
    println("Metrics to track:")
    println("  - SIMD instruction ratio: (SIMD ops) / (total ops)")
    println("  - FMA utilization: (FMA ops) / (multiply + add ops)")
    println("  - Memory efficiency: (vload/vstore) / (gather/scatter)")
    println("  - Register pressure: Max live registers")
end

"""
Example: Generate and compare multiple kernel variants
"""
function compare_kernel_variants(n::Int, ::Type{T}=Float64) where T
    println("Comparing kernel variants for FFT$n")
    println("="^60)
    
    # Variant 1: All scalar
    kernel1 = makefftradix(n, SuffixFlags(0), String[], 0, 
                          (stride=1, n_groups=1, eo=false), n, T, 128)
    println("\nVariant 1 (SSE/scalar):")
    println(kernel1)
    
    # Variant 2: AVX2 (256-bit)
    kernel2 = makefftradix_simd(n, SuffixFlags(0), String[], 0,
                               (stride=1, n_groups=1, eo=false), n, T, 256)
    println("\nVariant 2 (AVX2/256-bit):")
    println(kernel2)
    
    # Variant 3: AVX512 (512-bit)
    if SIMD_WIDTH >= 512
        kernel3 = makefftradix_simd(n, SuffixFlags(0), String[], 0,
                                   (stride=1, n_groups=1, eo=false), n, T, 512)
        println("\nVariant 3 (AVX512/512-bit):")
        println(kernel3)
    end
    
    return (scalar=kernel1, avx2=kernel2)
end

"""
Auto-tuning: Find best factorization for given size
"""
function autotune_factorization(N::Int, ::Type{T}=Float64; 
                               verbose::Bool=true) where T
    # Find all factorizations of N
    factorizations = find_all_factorizations(N)
    
    println("Auto-tuning FFT$N ($T)")
    println("Found $(length(factorizations)) factorizations")
    println("="^60)
    
    results = []
    
    for (i, factors) in enumerate(factorizations)
        if verbose
            println("\nTesting factorization $i: $(join(factors, " × "))")
        end
        
        # Generate kernel
        # ... code to generate and benchmark kernel ...
        
        # Benchmark
        # time = benchmark_kernel(kernel)
        
        # push!(results, (factors=factors, time=time))
    end
    
    # Select best
    # best = argmin(r -> r.time, results)
    
    println("\nBest factorization: ...")
    
    return results
end

"""
Helper: Find all factorizations of N using allowed radices
"""
function find_all_factorizations(N::Int, radices::Vector{Int}=[2,4,8,16])
    # Implementation of factorization search
    # Returns: Vector of Vector{Int}, each being a valid factorization
    
    results = Vector{Int}[]
    
    function search(n, current_factors)
        if n == 1
            push!(results, copy(current_factors))
            return
        end
        
        for r in radices
            if n % r == 0
                push!(current_factors, r)
                search(n ÷ r, current_factors)
                pop!(current_factors)
            end
        end
    end
    
    search(N, Int[])
    return results
end

# Export integration functions
export load_real_imag_gen_simd, store_real_imag_gen_simd
export makefftradix_simd
export benchmark_simd_vs_scalar, profile_simd_instructions
export compare_kernel_variants, autotune_factorization

println("SIMD integration module loaded.")
println("Use compare_kernel_variants(8) to see generated code comparison.")
compare_kernel_variants(8)