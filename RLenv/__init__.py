from gymnasium.envs.registration import register
register(
    id="HawkesRLTradingEnv-v0", entry_point="RLenv.Envs:HawkesRLTradingEnv"
)