include("suffix.jl")

# SOA-based FFT kernel generator for N = 2^q
# Generates expressions directly instead of strings

# Helper to generate variable names for SOA format
function soa_var(prefix::Symbol, idx::Int, component::Symbol)
    Symbol(prefix, idx, "_", component)
end

# Generate load expressions for SOA format
function load_soa_expr(vars, input_sym::Symbol)
    exprs = Expr[]
    for (i, var) in enumerate(vars)
        if occursin('[', var)
            # Direct array access
            idx = parse(Int, match(r"\d+", var).match)
            push!(exprs, :($(soa_var(:x, i, :r)) = real($(Symbol(input_sym))[$(idx)])))
            push!(exprs, :($(soa_var(:x, i, :i)) = imag($(Symbol(input_sym))[$(idx)])))
        else
            # Variable reference
            num = parse(Int, match(r"\d+", var).match)
            push!(exprs, :($(soa_var(:x, i, :r)) = real($(Symbol(input_sym, num)))))
            push!(exprs, :($(soa_var(:x, i, :i)) = imag($(Symbol(input_sym, num)))))
        end
    end
    Expr(:block, exprs...)
end

# Main kernel generation function with SOA
function makefftradix_soa(n::Int, suffixes::SuffixFlags, D::AbstractArray{String}, 
                          p::Int, s::Int, SIZE::Int, ::Type{T}) where T <: AbstractFloat
    
    has_y = has_flag(suffixes, Y)
    has_mat = has_flag(suffixes, MAT)
    has_vec = has_flag(suffixes, VEC)
    input = has_y ? :y : :x
    output = :y
    groups = SIZE ÷ s
    
    # Generate input/output index patterns
    if has_mat
        if has_vec
            x_indices = [i + p*s for i in 1:n]
            y_indices = [i + p*s for i in 1:n]
        else
            x_indices = [p + 1 + (i-1)*groups for i in 1:n]
            y_indices = [i + p*s for i in 1:n]
        end
        d = D == String[] ? nothing : D
    else
        if has_vec
            x_indices = [Symbol("offset + ", i) for i in 1:n]
            y_indices = [Symbol("offset + ", i) for i in 1:n]
        else
            x_indices = collect(1:s:n*s)
            y_indices = collect(1:n)
        end
        d = nothing
    end
    
    # Generate SOA kernel expression
    kernel_expr = recfft2_soa(n, x_indices, y_indices, input, output, d, nothing, true, T)
    
    return kernel_expr
end

# Twiddle factor expression generation with compile-time evaluation
function twiddle_expr_soa(w::String, tmp_r::Expr, tmp_i::Expr, ::Type{T}) where T
    if w == "1"
        return tmp_r, tmp_i
    elseif w == "-im"
        return tmp_i, :(-$(tmp_r))
    elseif w == "INV_SQRT2_Q4"
        return :($(T(1/√2)) * ($(tmp_r) + $(tmp_i))), 
               :($(T(1/√2)) * ($(tmp_i) - $(tmp_r)))
    elseif w == "-INV_SQRT2_Q1"
        return :($(T(1/√2)) * ($(tmp_i) - $(tmp_r))), 
               :($(T(-1/√2)) * ($(tmp_r) + $(tmp_i)))
    else
        # Parse CISPI format
        m = match(r"^([+-]?)(?:im\*)?CISPI_(\d+)_(\d+)_Q([14])$", w)
        if !isnothing(m)
            sign = m[1] == "-" ? -1 : 1
            num = parse(Int, m[2])
            den = parse(Int, m[3])
            is_q1 = m[4] == "1"
            
            # Compute twiddle factors at compile time
            angle = num / den
            c = T(sign * cospi(angle))
            s = T(sign * sinpi(angle))
            
            if startswith(w, "CISPI")
                if is_q1
                    return :(muladd($(c), $(tmp_r), $(-s) * $(tmp_i))),
                           :(muladd($(s), $(tmp_r), $(c) * $(tmp_i)))
                else
                    return :(muladd($(c), $(tmp_r), $(s) * $(tmp_i))),
                           :(muladd($(-s), $(tmp_r), $(c) * $(tmp_i)))
                end
            elseif occursin("im*CISPI", w)
                # Handle im* prefix cases
                if startswith(w, "-im*")
                    if is_q1
                        return :(muladd($(s), $(tmp_r), $(c) * $(tmp_i))),
                               :(muladd($(-c), $(tmp_r), $(s) * $(tmp_i)))
                    else
                        return :(muladd($(-s), $(tmp_r), $(c) * $(tmp_i))),
                               :(muladd($(-c), $(tmp_r), $(-s) * $(tmp_i)))
                    end
                else # im*CISPI
                    if is_q1
                        return :(muladd($(-s), $(tmp_r), $(-c) * $(tmp_i))),
                               :(muladd($(c), $(tmp_r), $(-s) * $(tmp_i)))
                    else
                        return :(muladd($(s), $(tmp_r), $(-c) * $(tmp_i))),
                               :(muladd($(c), $(tmp_r), $(s) * $(tmp_i)))
                    end
                end
            end
        end
        error("Unknown twiddle factor: $w")
    end
end

# Recursive FFT generation with SOA format
function recfft2_soa(n::Int, x_indices, y_indices, input::Symbol, output::Symbol, 
                     d, w, root::Bool, ::Type{T}, tmp_base::Int=1) where T <: AbstractFloat
    
    if n == 1
        # Base case: copy single element
        return quote
            @inbounds $(Symbol(output))[$(y_indices[1])] = $(Symbol(input))[$(x_indices[1])]
        end
        
    elseif n == 2
        # Size-2 butterfly with locals
        x1_idx = x_indices[1]
        x2_idx = x_indices[2]
        y1_idx = y_indices[1]
        y2_idx = y_indices[2]
        
        load_block = quote
            @inbounds begin
                local x1_r = real($(Symbol(input))[$(x1_idx)])
                local x1_i = imag($(Symbol(input))[$(x1_idx)])
                local x2_r = real($(Symbol(input))[$(x2_idx)])
                local x2_i = imag($(Symbol(input))[$(x2_idx)])
            end
        end
        
        if !isnothing(d) && isnothing(w) && root
            # With D matrix
            d1_r, d1_i = twiddle_expr_soa(d[1], :(x1_r - x2_r), :(x1_i - x2_i), T)
            compute_block = quote
                @inbounds begin
                    $(Symbol(output))[$(y1_idx)] = complex(x1_r + x2_r, x1_i + x2_i)
                    $(Symbol(output))[$(y2_idx)] = complex($(d1_r), $(d1_i))
                end
            end
        else
            # Standard butterfly
            if root
                compute_block = quote
                    @inbounds begin
                        $(Symbol(output))[$(y1_idx)] = complex(x1_r + x2_r, x1_i + x2_i)
                        $(Symbol(output))[$(y2_idx)] = complex(x1_r - x2_r, x1_i - x2_i)
                    end
                end
            else
                if isnothing(w)
                    # Store in temporary variables
                    compute_block = quote
                        @inbounds begin
                            $(Symbol(:t, tmp_base, :_r)) = x1_r + x2_r
                            $(Symbol(:t, tmp_base, :_i)) = x1_i + x2_i
                            $(Symbol(:t, tmp_base+1, :_r)) = x1_r - x2_r
                            $(Symbol(:t, tmp_base+1, :_i)) = x1_i - x2_i
                        end
                    end
                else
                    # Apply twiddle factors
                    w1_r, w1_i = twiddle_expr_soa(w[1], :(x1_r + x2_r), :(x1_i + x2_i), T)
                    w2_r, w2_i = twiddle_expr_soa(w[2], :(x1_r - x2_r), :(x1_i - x2_i), T)
                    compute_block = quote
                        @inbounds begin
                            $(Symbol(:t, tmp_base, :_r)) = $(w1_r)
                            $(Symbol(:t, tmp_base, :_i)) = $(w1_i)
                            $(Symbol(:t, tmp_base+1, :_r)) = $(w2_r)
                            $(Symbol(:t, tmp_base+1, :_i)) = $(w2_i)
                        end
                    end
                end
            end
        end
        
        return Expr(:block, load_block, compute_block)
        
    else
        # Recursive Cooley-Tukey decomposition
        n2 = n ÷ 2
        
        # Generate temporary variable symbols
        t_symbols = [Symbol(:t, i) for i in tmp_base:(tmp_base + n - 1)]
        new_tmp_base = tmp_base + n
        
        # Recursive calls for even and odd indices
        even_indices = x_indices[1:2:end]
        odd_indices = x_indices[2:2:end]
        
        # Generate twiddle factors for second half
        twiddles = if n2 > 1
            [get_twiddle_expression_soa(k, n, T) for k in 0:(n2-1)]
        else
            nothing
        end
        
        # Recursive FFT on even elements
        s1 = recfft2_soa(n2, even_indices, t_symbols[1:n2], 
                        input, :t, nothing, nothing, false, T, new_tmp_base)
        
        # Recursive FFT on odd elements with twiddles
        s2 = recfft2_soa(n2, odd_indices, t_symbols[n2+1:n], 
                        input, :t, nothing, twiddles, false, T, new_tmp_base)
        
        # Combine results with butterflies
        combine_exprs = Expr[]
        
        for i in 1:n2
            t_even_r = Symbol(:t, tmp_base + i - 1, :_r)
            t_even_i = Symbol(:t, tmp_base + i - 1, :_i)
            t_odd_r = Symbol(:t, tmp_base + n2 + i - 1, :_r)
            t_odd_i = Symbol(:t, tmp_base + n2 + i - 1, :_i)
            
            if root
                # Final output
                y_upper = y_indices[i]
                y_lower = y_indices[i + n2]
                
                if !isnothing(d) && i > 1
                    # Apply D matrix twiddles
                    d_upper_r, d_upper_i = twiddle_expr_soa(d[i-1], 
                        :($(t_even_r) + $(t_odd_r)), :($(t_even_i) + $(t_odd_i)), T)
                    d_lower_r, d_lower_i = twiddle_expr_soa(d[i+n2-1], 
                        :($(t_even_r) - $(t_odd_r)), :($(t_even_i) - $(t_odd_i)), T)
                    
                    push!(combine_exprs, quote
                        @inbounds begin
                            $(Symbol(output))[$(y_upper)] = complex($(d_upper_r), $(d_upper_i))
                            $(Symbol(output))[$(y_lower)] = complex($(d_lower_r), $(d_lower_i))
                        end
                    end)
                else
                    push!(combine_exprs, quote
                        @inbounds begin
                            $(Symbol(output))[$(y_upper)] = complex(
                                $(t_even_r) + $(t_odd_r),
                                $(t_even_i) + $(t_odd_i)
                            )
                            $(Symbol(output))[$(y_lower)] = complex(
                                $(t_even_r) - $(t_odd_r),
                                $(t_even_i) - $(t_odd_i)
                            )
                        end
                    end)
                end
            else
                # Store in temporaries for next layer
                new_t_upper_r = Symbol(:t, new_tmp_base + i - 1, :_r)
                new_t_upper_i = Symbol(:t, new_tmp_base + i - 1, :_i)
                new_t_lower_r = Symbol(:t, new_tmp_base + n2 + i - 1, :_r)
                new_t_lower_i = Symbol(:t, new_tmp_base + n2 + i - 1, :_i)
                
                push!(combine_exprs, quote
                    @inbounds begin
                        local $(new_t_upper_r) = $(t_even_r) + $(t_odd_r)
                        local $(new_t_upper_i) = $(t_even_i) + $(t_odd_i)
                        local $(new_t_lower_r) = $(t_even_r) - $(t_odd_r)
                        local $(new_t_lower_i) = $(t_even_i) - $(t_odd_i)
                    end
                end)
            end
        end
        
        return Expr(:block, s1, s2, combine_exprs...)
    end
end

# Generate twiddle factor at compile time
function get_twiddle_expression_soa(k::Int, n::Int, ::Type{T}) where T
    angle = -2π * k / n
    c = T(cos(angle))
    s = T(sin(angle))
    
    if abs(c - 1) < eps(T) && abs(s) < eps(T)
        return "1"
    elseif abs(c) < eps(T) && abs(s + 1) < eps(T)
        return "-im"
    elseif abs(c - 1/√2) < eps(T) && abs(s - 1/√2) < eps(T)
        return "INV_SQRT2_Q1"
    elseif abs(c - 1/√2) < eps(T) && abs(s + 1/√2) < eps(T)
        return "INV_SQRT2_Q4"
    else
        # Return as CISPI format for consistency
        # Find the fraction representation
        frac = k / (n/2)
        num = k
        den = n ÷ 2
        gcd_val = gcd(num, den)
        num ÷= gcd_val
        den ÷= gcd_val
        
        return "CISPI_$(num)_$(den)_Q4"
    end
end

# Counter for temporary variables
function inccounter()
    let counter = 0
        return () -> (counter += 1)
    end
end

inc = inccounter()

# Export the new SOA-based kernel generator
export makefftradix_soa, recfft2_soa