import unittest

class TestInference(unittest.TestCase):
    def test_inference_success(self):
        actual = ''
        expected = ''
        self.assertEqual(actual, expected)

    def test_inference_fail(self):
        actual = ''
        expected = ' '
        self.assertNotEqual(actual, expected)

if __name__ == '__main__' : 
    unittest.main()