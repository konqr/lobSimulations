from gymnasium.envs.registration import register
register(
    id="HawkesRLTradingEnv-v0", entry_point="HawkesRLTrading.src.Envs:HawkesRLTradingEnv"
)