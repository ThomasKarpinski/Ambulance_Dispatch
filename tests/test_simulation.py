import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation import DispatchSimulator, Ambulance, Emergency

class TestSimulation(unittest.TestCase):
    def setUp(self):
        """Set up a new simulator for each test."""
        self.simulator = DispatchSimulator(num_ambulances_per_base=1)

    def test_simulator_initialization(self):
        """Test that the simulator initializes with the correct number of ambulances."""
        self.assertEqual(len(self.simulator.ambulances), 1)
        self.assertEqual(self.simulator.ambulances[0].id, 0)
        self.assertEqual(self.simulator.ambulances[0].status, 'available')

    def test_spawn_emergency(self):
        """Test that an emergency is spawned and added to the active list."""
        initial_emergency_count = len(self.simulator.active_emergencies)
        self.simulator.spawn_emergency()
        self.assertEqual(len(self.simulator.active_emergencies), initial_emergency_count + 1)

    def test_reassign_emergency(self):
        """Test that an available ambulance is reassigned to a new emergency."""
        self.simulator.spawn_emergency()
        self.simulator.reassign_emergencies()
        dispatched_ambulance_found = any(e.dispatched_ambulance is not None for e in self.simulator.active_emergencies)
        self.assertTrue(dispatched_ambulance_found)
    
    def test_full_simulation_step(self):
        """Test a full simulation step to see if ambulance state changes."""
        self.simulator.spawn_emergency()
        self.simulator.run_simulation_step()
        
        ambulance = self.simulator.ambulances[0]
        # Depending on the random emergency location, the ambulance might be responding or transporting
        # So we check if the status is no longer 'available'.
        self.assertNotEqual(ambulance.status, 'available')


if __name__ == '__main__':
    unittest.main()
