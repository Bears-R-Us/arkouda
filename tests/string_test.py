from context import arkouda as ak
from base_test import ArkoudaTest


class StringTest(ArkoudaTest):
    
    def test_random_strings_uniform(self):
        r_strings = ak.random_strings_uniform(minlen=5, maxlen=6, size=5, characters='uppercase')
        self.assertEqual(5, len(r_strings))
        for r_string in r_strings:
            self.assertFalse(r_string)
