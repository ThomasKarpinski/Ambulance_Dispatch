import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genetic_algorithm import GA_Generator

class TestGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        """Set up a new GA_Generator for each test."""
        ga_config = {
            "emergency_spawn_range": (0, 1),
            "mutation_rate": 0.1,
            "mutation_rate_change": 0.0,
            "crosbreed_rate": 0.7,
            "crosbreed_rate_change": 0.0,
            "fuzzy": False
        }
        self.ga_generator = GA_Generator(**ga_config)

    def test_ga_initialization(self):
        """Test that the GA_Generator initializes correctly."""
        self.assertEqual(self.ga_generator.mutation_rate, 0.1)
        self.assertFalse(self.ga_generator.fuzzy)

    def test_genetic_algorithm_run(self):
        """Test that the geneticAlgorithm method runs without errors."""
        try:
            self.ga_generator.geneticAlgorithm(epochs=2, generations=2, ambulances_per_base=1)
        except Exception as e:
            self.fail(f"geneticAlgorithm raised an exception: {e}")

    def test_genetic_algorithm_run_with_fuzzy(self):
        """Test that the geneticAlgorithm method runs with fuzzy=True."""
        self.ga_generator.fuzzy = True
        try:
            self.ga_generator.geneticAlgorithm(epochs=2, generations=2, ambulances_per_base=1)
        except Exception as e:
            self.fail(f"geneticAlgorithm with fuzzy=True raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
