import numpy as np
import pytest
import arkouda as ak

class TestSparse:

    def test_utils(self):
        csc = ak.random_sparse_matrix(10, 0.2, "CSC")
        csr = ak.random_sparse_matrix(10, 0.2, "CSR")
        vals_csc = csc.to_pdarray()[2].to_ndarray()
        vals_csr = csr.to_pdarray()[2].to_ndarray()
        assert np.all(vals_csc == 0)
        assert np.all(vals_csr == 0)
        fill_vals_csc = ak.randint(0, 10, csc.nnz)
        fill_vals_csr = ak.randint(0, 10, csr.nnz)
        csc.fill_vals(fill_vals_csc)
        csr.fill_vals(fill_vals_csr)
        vals_csc = csc.to_pdarray()[2].to_ndarray()
        vals_csr = csr.to_pdarray()[2].to_ndarray()
        assert np.all(vals_csc == fill_vals_csc.to_ndarray())
        assert np.all(vals_csr == fill_vals_csr.to_ndarray())



    def test_matmatmult(self):
        # Create a reference for matrix multiplication in python:
        def matmatmult(rowsA, colsA, valsA, rowsB, colsB, valsB):
            """
            Perform matrix-matrix multiplication of two sparse matrices represented by
            3 arrays of rows, cols, and vals.
            A . B
            Parameters
            ----------
            rowsA : list or array-like
                Row indices of non-zero elements in matrix A.
            colsA : list or array-like
                Column indices of non-zero elements in matrix A.
            valsA : list or array-like
                Values of non-zero elements in matrix A.
            rowsB : list or array-like
                Row indices of non-zero elements in matrix B.
            colsB : list or array-like
                Column indices of non-zero elements in matrix B.
            valsB : list or array-like
                Values of non-zero elements in matrix B.

            Returns
            -------
            result_rows : list
                Row indices of non-zero elements in the result matrix.
            result_cols : list
                Column indices of non-zero elements in the result matrix.
            result_vals : list
                Values of non-zero elements in the result matrix.
            """
            from collections import defaultdict

            # Dictionary to accumulate the results
            result = defaultdict(float)

            # Create a dictionary for quick lookup of matrix B elements
            B_dict = defaultdict(list)
            for r, c, v in zip(rowsB, colsB, valsB):
                B_dict[r].append((c, v))

            # Perform the multiplication
            for rA, cA, vA in zip(rowsA, colsA, valsA):
                if cA in B_dict:
                    for cB, vB in B_dict[cA]:
                        result[(rA, cB)] += vA * vB

            # Extract the results into separate lists
            result_rows, result_cols, result_vals = zip(*[(r, c, v) for (r, c), v in result.items()])

            return list(result_rows), list(result_cols), list(result_vals)
        matA = ak.random_sparse_matrix(100, 1, 'CSC') # Make it fully dense to make testing easy
        matB = ak.random_sparse_matrix(100, 1, 'CSR') # Make it fully dense to make testing easy
        fill_vals_a = ak.randint(0, 10, matA.nnz)
        fill_vals_b = ak.randint(0, 10, matB.nnz)
        matA.fill_vals(fill_vals_a)
        matB.fill_vals(fill_vals_b)
        rowsA, colsA, valsA = (arr.to_ndarray() for arr in matA.to_pdarray())
        rowsB, colsB, valsB = (arr.to_ndarray() for arr in matB.to_pdarray())
        assert np.all(valsA == fill_vals_a.to_ndarray())
        assert np.all(valsB == fill_vals_b.to_ndarray())
        ans_rows, ans_cols, ans_vals = matmatmult(rowsA, colsA, valsA, rowsB, colsB, valsB)

        result = ak.sparse_matrix_matrix_mult(matA, matB)
        result_rows, result_cols, result_vals = (arr.to_ndarray() for arr in result.to_pdarray())

        # Check the result is correct
        assert np.all(result_rows == ans_rows)
        assert np.all(result_cols == ans_cols)
        assert np.all(result_vals == ans_vals)
