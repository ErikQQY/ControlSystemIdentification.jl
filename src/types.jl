abstract type AbstractIdData end

const AnyInput = Union{AbstractArray,AbstractIdData}

struct InputOutputData{Y,U,T} <: AbstractIdData
    y::Y
    u::U
    Ts::T
end

struct OutputData{Y,T} <: AbstractIdData
    y::Y
    Ts::T
end

struct InputOutputStateData{Y,U,X,T} <: AbstractIdData
    y::Y
    u::U
    x::X
    Ts::T
end

autodim(x::Vector{<:AbstractVector}) = x
autodim(x::AbstractVector) = x'
function autodim(x)
    r = size(x, 1)
    c = size(x, 2)
    if (c < 5 && c < r) || (r > 4c)
        @info "Transposing input. The convention used in ControlSystemIdentification is that input-output data is made out of either of 1) Vectors with scalars, 2) vectors of vectors or 3) matrices with time along the second dimension. The supplied input appears to be multidimensional and have time in the first dimension." maxlog =
            3
        return copy(x')
    end
    x
end

function Base.show(io::IO, d::OutputData)
    write(io, "Output data of length $(length(d)) with $(noutputs(d)) outputs")
end
function Base.show(io::IO, d::InputOutputData)
    write(
        io,
        "InputOutput data of length $(length(d)) with $(noutputs(d)) outputs and $(ninputs(d)) inputs",
    )
end


iddata(y::AbstractArray, Ts::Union{Real,Nothing} = nothing) = OutputData(autodim(y), Ts)
iddata(y::AbstractArray, u::AbstractArray, Ts::Union{Real,Nothing} = nothing) =
    InputOutputData(autodim(y), autodim(u), Ts)

"""
    iddata(y, u, x, Ts = nothing)

Returns the appropriate IdData object, depending on the input.

# Arguments
- `y::AbstractArray`: output data
- `u::AbstractArray`: input data
- `x::AbstractArray`: state data
- `Ts::Union{Real,Nothing} = nothing`: optional sample time

# Examples
```jldoctest
julia> iddata(randn(10))
Output data of length 10 with 1 outputs

julia> iddata(randn(10), randn(10), 1)
InputOutput data of length 10 with 1 outputs and 1 inputs
```
"""
iddata(
    y::AbstractArray,
    u::AbstractArray,
    x::AbstractArray,
    Ts::Union{Real,Nothing} = nothing,
) = InputOutputStateData(autodim(y), autodim(u), x, Ts)


output(d::AbstractIdData)                        = d.y
input(d::AbstractIdData)                         = d.u
LowLevelParticleFilters.state(d::AbstractIdData) = d.x
output(d::AbstractArray)                         = d
input(d::AbstractArray)                          = d
LowLevelParticleFilters.state(d::AbstractArray)  = d
hasinput(::OutputData)                           = false
hasinput(::AbstractIdData)                       = true
hasinput(::AbstractArray)                        = true
hasinput(::ControlSystems.LTISystem)             = true
ControlSystems.noutputs(d::AbstractIdData)       = obslength(d.y)
ControlSystems.ninputs(d::AbstractIdData)        = hasinput(d) ? obslength(d.u) : 0
ControlSystems.nstates(d::AbstractIdData)        = 0
ControlSystems.nstates(d::InputOutputStateData)  = obslength(d.x)
obslength(d::AbstractIdData)                     = ControlSystems.noutputs(d)
sampletime(d::AbstractIdData)                    = d.Ts === nothing ? 1.0 : d.Ts
function Base.length(d::AbstractIdData)
    y = output(d)
    y isa AbstractMatrix && return size(y, 2)
    return length(y)
end

Base.lastindex(d::AbstractIdData) = length(d)

function Base.getproperty(d::AbstractIdData, s::Symbol)
    if s === :fs || s === :Fs
        return 1 / d.Ts
    end
    return getfield(d, s)
end

function Base.:(==)(d1::T, d2::T) where {T<:AbstractIdData}
    all(fieldnames(T)) do field
        getfield(d1, field) == getfield(d2, field)
    end
end


timevec(d::AbstractIdData) = range(0, step = sampletime(d), length = length(d))
timevec(d::AbstractVector, h::Real) = range(0, step = h, length = length(d))
timevec(d::AbstractMatrix, h::Real) = range(0, step = h, length = maximum(size(d)))

function apply_fun(fun, d::OutputData, Ts = d.Ts)
    iddata(fun(d.y), Ts)
end

"""
	apply_fun(fun, d::InputOutputData)

Apply `fun(y)` to all time series `y[,u,[x]] ∈ d` and return a new `iddata` with the transformed series.
"""
function apply_fun(fun, d::InputOutputData, Ts = d.Ts)
    iddata(fun(d.y), fun(d.u), Ts)
end

function apply_fun(fun, d::InputOutputStateData, Ts = d.Ts)
    iddata(fun(d.y), fun(d.u), fun(d.x), Ts)
end

torange(x::Number) = x:x
torange(x) = x

function Base.getindex(d::Union{InputOutputData,InputOutputStateData}, i, j)
    iddata(d.y[torange(i), :], d.u[torange(j), :], d.Ts)
end


function Base.getindex(d::AbstractIdData, i)
    apply_fun(d) do y
        y[:, i]
    end
end

"""
dr = resample(d::InputOutputData, f)

Resample iddata `d` with fraction `f`, e.g., `f = fs_new / fs_original`.
"""
function DSP.resample(d::AbstractIdData, f)
    apply_fun(d, d.Ts / f) do y
        yr = mapslices(y, dims = 2) do y
            resample(y, f)
        end
        yr
    end
end

function DSP.resample(M::AbstractMatrix, f)
    mapslices(M, dims = 1) do y
        resample(y, f)
    end
end

function Base.hcat(d1::InputOutputData, d2::InputOutputData)
    @assert d1.Ts == d2.Ts
    iddata([d1.y d2.y], [d1.u d2.u], d1.Ts)
end


@recipe function plot(d::AbstractIdData)
    y = time1(output(d))
    n = noutputs(d)
    if hasinput(d)
        u = time1(input(d))
        n += ninputs(d)
    end
    layout --> (n, 1)
    legend --> false
    xguide --> "Time"
    link --> :x
    xvec = range(0, step = sampletime(d), length = length(d))

    for i = 1:size(y, 2)
        @series begin
            title --> "Output $i"
            label --> "Output $i"
            xvec, y[:, i]
        end
    end
    if hasinput(d)
        for i = 1:size(u, 2)
            @series begin
                title --> "Input $i"
                label --> "Input $i"
                xvec, u[:, i]
            end
        end
    end
end
