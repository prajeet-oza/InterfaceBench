import argparse
import unittest
from interfacebench.cli import parse_models

class TestCLI(unittest.TestCase):
    def test_parse_models_comma_separated(self):
        """Ensure simple comma-separated tags are expanded to valid dictionaries"""
        res = parse_models("mace-mpa, mattersim-5m")
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]['tag'], 'mace-mpa')
        self.assertEqual(res[0]['model_code'], 'None')
        self.assertEqual(res[0]['model_import'], '# Define your calculator import here')
        
    def test_parse_models_json_array(self):
        """Ensure a raw string array of dictionaries is safely evaluated"""
        res = parse_models("[{'tag': 'mattersim', 'model_code': 'MatterSimCalculator()'}]")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]['model_code'], 'MatterSimCalculator()')
        
    def test_parse_models_invalid_syntax(self):
        """Ensure malformed model strings raise an ArgumentTypeError instead of a raw Python traceback"""
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_models("[{'tag': 'mattersim'") # Missing closing bracket

if __name__ == '__main__':
    unittest.main()