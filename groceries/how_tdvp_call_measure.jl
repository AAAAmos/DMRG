function sub_sweep_update(
        updater,
        updater_kwargs,
    )
    update_observer!(
        observer!;
        state,
        reduced_operator,
        bond = b,
        sweep,
        half_sweep = isforward(direction) ? 1 : 2,
        spec,
        outputlevel,
        half_sweep_is_done = is_half_sweep_done(direction, b, N; ncenter = nsite),
        current_time,
        info,
    )

end

function update_observer!(observer::AbstractObserver; kwargs...)
    return measure!(observer; kwargs...)
end