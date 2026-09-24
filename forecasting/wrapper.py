from pettingzoo.utils.wrappers import BaseParallelWrapper
import numpy as np
import gymnasium.spaces as spaces

class DemandForecastWrapper(BaseParallelWrapper):
    """
    Wrapper that adds demand forecasts to the observation of each agent.
    If comm is disabled, replaces Box(5) with Box(5 + horizon).
    If comm is enabled, replaces Dict['state'] Box(5) with Box(5 + horizon).
    """
    def __init__(self, env, forecaster, forecast_horizon=1, history_window=5):
        super().__init__(env)
        self.forecaster = forecaster
        self.forecast_horizon = forecast_horizon
        self.history_window = history_window
        self.demand_history = []
        
        # Modify observation spaces
        self.observation_spaces = dict()
        for agent in self.possible_agents:
            base_space = env.observation_space(agent)
            
            if isinstance(base_space, spaces.Dict):
                # Comm enabled
                old_state = base_space.spaces["state"]
                new_state = spaces.Box(
                    low=0.0, 
                    high=np.inf, 
                    shape=(old_state.shape[0] + forecast_horizon,), 
                    dtype=np.float32
                )
                
                new_dict = dict(base_space.spaces)
                new_dict["state"] = new_state
                self.observation_spaces[agent] = spaces.Dict(new_dict)
            else:
                # Comm disabled
                self.observation_spaces[agent] = spaces.Box(
                    low=0.0, 
                    high=np.inf, 
                    shape=(base_space.shape[0] + forecast_horizon,), 
                    dtype=np.float32
                )
                
    def observation_space(self, agent):
        return self.observation_spaces[agent]
        
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.demand_history = []
        
        # We need the initial demand. The simulator has it.
        # But wait, at reset, last_demand is 0.
        initial_demand = 0 # Actually we could peek at simulator
        self.demand_history.append(initial_demand)
        
        return self._inject_forecast(obs), info
        
    def step(self, action):
        obs, rewards, terminations, truncations, infos = super().step(action)
        
        # After step, the simulator has processed new demand.
        # Retailer's observation has the latest demand at index 3 (last_demand)
        # We can extract it from the raw obs before injection.
        
        retailer_obs = obs["retailer"]
        if isinstance(retailer_obs, dict):
            last_demand = retailer_obs["state"][3]
        else:
            last_demand = retailer_obs[3]
            
        self.demand_history.append(last_demand)
        
        # Inject forecast
        return self._inject_forecast(obs), rewards, terminations, truncations, infos
        
    def _inject_forecast(self, obs):
        forecast = self.forecaster.predict(self.demand_history, horizon=self.forecast_horizon)
        forecast_array = np.array(forecast, dtype=np.float32)
        
        new_obs = {}
        for agent, o in obs.items():
            if isinstance(o, dict):
                state = o["state"]
                new_state = np.concatenate([state, forecast_array])
                new_o = dict(o)
                new_o["state"] = new_state
                new_obs[agent] = new_o
            else:
                new_obs[agent] = np.concatenate([o, forecast_array])
                
        return new_obs
