from core.delay_chain_pipeline import DelayChainPipeline

pipeline = DelayChainPipeline()
real = pipeline.analyze(real_df, planned_df, label="real")

simulated_events = pipeline.prepare_simulated_events(simulated_df, planned_df)
simulated = pipeline.analyze(simulated_events, planned_df, label="simulated")

comparison = pipeline.compare({
    "real": real,
    "simulated": simulated,
})