from context import arkouda as ak
from base_test import ArkoudaTest


class RegexTest(ArkoudaTest):

    def test_regex_contains(self):
        digit_strings = ak.array(['{} string {}'.format(i, i) for i in range(1, 6)])
        self.assertTrue(digit_strings.contains('\\d str', regex=True).all())
        self.assertTrue(digit_strings.contains('ing \\d', regex=True).all())

        aaa_strings = ak.array(['{} string {}'.format('a'*i, 'a'*i) for i in range(1, 6)])
        self.assertTrue(aaa_strings.contains('a+ str', regex=True).all())
        self.assertTrue(aaa_strings.contains('ing a+', regex=True).all())

    def test_regex_startswith(self):
        digit_strings = ak.array(['{} string {}'.format(i, i) for i in range(1, 6)])
        self.assertTrue(digit_strings.startswith('\\d str', regex=True).all())
        self.assertFalse(digit_strings.startswith('ing \\d', regex=True).any())

        aaa_strings = ak.array(['{} string {}'.format('a'*i, 'a'*i) for i in range(1, 6)])
        self.assertTrue(aaa_strings.startswith('a+ str', regex=True).all())
        self.assertFalse(aaa_strings.startswith('ing a+', regex=True).any())

    def test_regex_endswith(self):
        digit_strings = ak.array(['{} string {}'.format(i, i) for i in range(1, 6)])
        self.assertTrue(digit_strings.endswith('ing \\d', regex=True).all())
        self.assertFalse(digit_strings.endswith('\\d str', regex=True).any())

        aaa_strings = ak.array(['{} string {}'.format('a'*i, 'a'*i) for i in range(1, 6)])
        self.assertTrue(aaa_strings.endswith('ing a+', regex=True).all())
        self.assertFalse(aaa_strings.endswith('a+ str', regex=True).any())

    def test_regex_match(self):
        digit_strings = ak.array(['{} string {}'.format(i, i) for i in range(1, 6)])
        self.assertTrue(digit_strings.match('\\d string \\d').all())
        # the following are false because the regex is not an exact match of any digit_strings
        self.assertFalse(digit_strings.match('ing \\d').any())
        self.assertFalse(digit_strings.match('\\d str').any())

        aaa_strings = ak.array(['{} string {}'.format('a'*i, 'a'*i) for i in range(1, 6)])
        self.assertTrue(aaa_strings.match('a+ string a+').all())
        # the following are false because the regex is not an exact match of any aaa_strings
        self.assertFalse(aaa_strings.match('ing a+').any())
        self.assertFalse(aaa_strings.match('a+ str').any())

    def test_regex_peel(self):
        orig = ak.array(['a.b', 'c.d', 'e.f.g'])
        digit = ak.array(['a1b', 'c1d', 'e1f2g'])
        under = ak.array(['a_b', 'c___d', 'e__f____g'])

        o_left, o_right = orig.peel('.')
        d_left, d_right = digit.peel('\\d', regex=True)
        u_left, u_right = under.peel('_+', regex=True)
        self.assertListEqual(['a', 'c', 'e'], o_left.to_ndarray().tolist())
        self.assertListEqual(['a', 'c', 'e'], d_left.to_ndarray().tolist())
        self.assertListEqual(['a', 'c', 'e'], u_left.to_ndarray().tolist())
        self.assertListEqual(['b', 'd', 'f.g'], o_right.to_ndarray().tolist())
        self.assertListEqual(['b', 'd', 'f2g'], d_right.to_ndarray().tolist())
        self.assertListEqual(['b', 'd', 'f____g'], u_right.to_ndarray().tolist())

        o_left, o_right = orig.peel('.', includeDelimiter=True)
        d_left, d_right = digit.peel('\\d', includeDelimiter=True, regex=True)
        u_left, u_right = under.peel('_+', includeDelimiter=True, regex=True)
        self.assertListEqual(['a.', 'c.', 'e.'], o_left.to_ndarray().tolist())
        self.assertListEqual(['a1', 'c1', 'e1'], d_left.to_ndarray().tolist())
        self.assertListEqual(['a_', 'c___', 'e__'], u_left.to_ndarray().tolist())
        self.assertListEqual(['b', 'd', 'f.g'], o_right.to_ndarray().tolist())
        self.assertListEqual(['b', 'd', 'f2g'], d_right.to_ndarray().tolist())
        self.assertListEqual(['b', 'd', 'f____g'], u_right.to_ndarray().tolist())

        o_left, o_right = orig.peel('.', times=2, keepPartial=True)
        d_left, d_right = digit.peel('\\d', times=2, keepPartial=True, regex=True)
        u_left, u_right = under.peel('_+', times=2, keepPartial=True, regex=True)
        self.assertListEqual(['a.b', 'c.d', 'e.f'], o_left.to_ndarray().tolist())
        self.assertListEqual(['a1b', 'c1d', 'e1f'], d_left.to_ndarray().tolist())
        self.assertListEqual(['a_b', 'c___d', 'e__f'], u_left.to_ndarray().tolist())
        self.assertListEqual(['', '', 'g'], o_right.to_ndarray().tolist())
        self.assertListEqual(['', '', 'g'], d_right.to_ndarray().tolist())
        self.assertListEqual(['', '', 'g'], u_right.to_ndarray().tolist())

        # rpeel / fromRight: digit is testing fromRight and under is testing rpeel
        o_left, o_right = orig.peel('.', times=2, includeDelimiter=True, fromRight=True)
        d_left, d_right = digit.peel('\\d', times=2, includeDelimiter=True, fromRight=True, regex=True)
        u_left, u_right = under.rpeel('_+', times=2, includeDelimiter=True, regex=True)
        self.assertListEqual(['a.b', 'c.d', 'e'], o_left.to_ndarray().tolist())
        self.assertListEqual(['a1b', 'c1d', 'e'], d_left.to_ndarray().tolist())
        self.assertListEqual(['a_b', 'c___d', 'e'], u_left.to_ndarray().tolist())
        self.assertListEqual(['', '', '.f.g'], o_right.to_ndarray().tolist())
        self.assertListEqual(['', '', '1f2g'], d_right.to_ndarray().tolist())
        self.assertListEqual(['', '', '__f____g'], u_right.to_ndarray().tolist())

    def test_regex_flatten(self):
        orig = ak.array(['one|two', 'three|four|five', 'six', 'seven|eight|nine|ten|', 'eleven'])
        digit = ak.array(['one1two', 'three2four3five', 'six', 'seven4eight5nine6ten7', 'eleven'])
        under = ak.array(['one_two', 'three_four__five', 'six', 'seven_____eight__nine____ten_', 'eleven'])

        answer_flat = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', '', 'eleven']
        answer_map = [0, 2, 5, 6, 11]

        orig_flat, orig_map = orig.flatten('|', return_segments=True)
        digit_flat, digit_map = digit.flatten('\\d', return_segments=True, regex=True)
        under_flat, under_map = under.flatten('_+', return_segments=True, regex=True)

        self.assertListEqual(answer_flat, orig_flat.to_ndarray().tolist())
        self.assertListEqual(answer_flat, digit_flat.to_ndarray().tolist())
        self.assertListEqual(answer_flat, under_flat.to_ndarray().tolist())

        self.assertListEqual(answer_map, orig_map.to_ndarray().tolist())
        self.assertListEqual(answer_map, digit_map.to_ndarray().tolist())
        self.assertListEqual(answer_map, under_map.to_ndarray().tolist())

        # empty string, start with delim, end with delim, and only delim cases
        orig = ak.array(['', '|', '|1|2', '3|4|', '5'])
        regex = ak.array(['', '____', '_1_2', '3___4___', '5'])

        answer_flat = ['', '', '', '', '1', '2', '3', '4', '', '5']
        answer_map = [0, 1, 3, 6, 9]

        orig_flat, orig_map = orig.flatten('|', return_segments=True)
        regex_flat, regex_map = regex.flatten('_+', return_segments=True, regex=True)

        self.assertListEqual(answer_flat, orig_flat.to_ndarray().tolist())
        self.assertListEqual(answer_flat, regex_flat.to_ndarray().tolist())

        self.assertListEqual(answer_map, orig_map.to_ndarray().tolist())
        self.assertListEqual(answer_map, regex_map.to_ndarray().tolist())
