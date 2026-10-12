import unittest
from lattice_basis_reduction import lll_reduce


class TestLLReduce(unittest.TestCase):
    def test_empty_basis(self):
        result = lll_reduce([])
        self.assertEqual(result, [])

    def test_single_vector(self):
        result = lll_reduce([[3, 4]])
        self.assertEqual(result, [[3, 4]])

    def test_already_reduced(self):
        basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        result = lll_reduce(basis)
        self.assertEqual(result, basis)

    def test_simple_reduction(self):
        basis = [[1, 1, 1], [-1, 0, 2], [3, 4, 5]]
        result = lll_reduce(basis)
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                dot = sum(a * b for a, b in zip(result[i], result[j]))
                norm_i = sum(x * x for x in result[i])
                norm_j = sum(x * x for x in result[j])
                if norm_i > 0 and norm_j > 0:
                    ratio = abs(dot) * abs(dot)
                    self.assertLess(
                        ratio,
                        norm_i * norm_j // 2 + 1,
                        "Vectors not sufficiently orthogonal",
                    )

    def test_output_dimensions_preserved(self):
        basis = [[5, 2, 1], [1, 0, -3], [0, 0, 1]]
        result = lll_reduce(basis)
        self.assertEqual(len(result), 3)
        for row in result:
            self.assertEqual(len(row), 3)

    def test_invalid_delta(self):
        with self.assertRaises(ValueError):
            lll_reduce([[1, 0], [0, 1]], delta=10)
        with self.assertRaises(ValueError):
            lll_reduce([[1, 0], [0, 1]], delta=150)

    def test_ragged_basis_raises(self):
        with self.assertRaises(ValueError):
            lll_reduce([[1, 0], [0, 1, 0]])

    def test_lattice_span_preserved(self):
        import itertools

        basis = [[2, 1, 0], [0, 1, 2], [1, 0, 1]]
        result = lll_reduce(basis)

        def det3(m):
            return (
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            )

        d1 = det3(basis)
        d2 = det3(result)
        self.assertEqual(abs(d1), abs(d2))

    def test_two_vectors(self):
        basis = [[10, 0], [0, 10]]
        result = lll_reduce(basis)
        self.assertEqual(result, basis)

    def test_near_parallel_vectors(self):
        basis = [[1000, 1], [1001, 0]]
        result = lll_reduce(basis)
        norm0 = sum(x * x for x in result[0])
        self.assertLess(norm0, 1000 * 1000 + 1)

    def test_input_not_mutated(self):
        basis = [[1, 1, 1], [-1, 0, 2], [3, 4, 5]]
        import copy

        original = copy.deepcopy(basis)
        lll_reduce(basis)
        self.assertEqual(basis, original)

    def test_large_coefficients(self):
        basis = [[1000000, 500000], [500001, 250000]]
        result = lll_reduce(basis)
        self.assertEqual(len(result), 2)
        n0 = sum(x * x for x in result[0])
        n1 = sum(x * x for x in result[1])
        self.assertLessEqual(n0, n1)


if __name__ == "__main__":
    unittest.main()
