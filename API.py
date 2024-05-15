# Autor: Gabriel Antonio Cavichioli | Data: 15/05/24.

from coppeliasim_zmqremoteapi_client import RemoteAPIClient # type: ignore

client = RemoteAPIClient()
sim = client.require('sim')

sim.setStepping(True)

sim.startSimulation()
while (t := sim.getSimulationTime()) < 50:
    print(f'Simulation time: {t:.2f} [s]')
    sim.step()
sim.stopSimulation()