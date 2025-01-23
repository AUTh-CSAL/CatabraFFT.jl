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