# Helper Tools used throught this project:

    function subpowers_of_two(N::Int)
    # Check if N is a power of two
    @assert N > 1 && (N & (N - 1)) == 0 "N must be a power of two greater than 1"
    
    # Generate the list of subpowers
    subpowers = Vector{Int}()
    while N >= 2
        push!(subpowers, N)
        N = div(N, 2)
    end
    return subpowers
end

function get_radix_family(op_type::Symbol)
    radix = parse(Int, String(op_type)[4:end])
    if ispow2(radix)
        return radix_2_family
    elseif radix ∈ [3, 9]
        return radix_3_family
    elseif radix == 5
        return radix_5_family
    elseif radix == 7
        return radix_7_family
    else
        error("Unsupported radix: $radix")
    end
end

function get_function_reference(radix_family, base_function_name::Symbol)
    func = getfield(radix_family, base_function_name)
    if !isdefined(radix_family, base_function_name)
        error("Function $base_function_name not found in module $(radix_family)")
    end
    return func
end
