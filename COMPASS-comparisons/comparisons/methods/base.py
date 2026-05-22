from typing import Protocol,Any

class Method(Protocol):
    name: str

    def init_worker(self, log,scenario) -> None:
        """Initialize any global state needed for the method. 
        This will be called once per worker process, not once per rep."""
        ...
    
    def run_rep(self,rep_id, **args) -> Any:
        """Run a single replication of the method."""
        ...

    def save_scenario(self,out_path) -> None:
        """Save results for a single scenario. This will be called after all reps for the scenario are done."""
        ...