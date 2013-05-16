using System;

namespace HmmDotNet.Mathematic.MatrixDecomposition
{
    public class LU : IDecomposition<double>
    {
        #region Members

        private double[,] _decomposition = null;
        private double[,] _upper = null;
        private double[,] _lower = null;
        private int _pivotSign;
        private int[] _pivot;

        #endregion Members

        #region Properties


        public double[,] Decomposition
        {
            get { return _decomposition; }
        }

        public double[,] Upper
        {
            get { return _upper; }
        }

        public double[,] Lower
        {
            get { return _lower; }
        }

        #endregion Properties

        /// <summary>
        ///     Returns matrix decomposition of m, when the decomposition is formed from two
        ///     triangular matrixes L and U. m can be not positive definite matrix and must be square.
        ///     m must be nonsingular and m transpose diagonally dominant
        /// <para>
        ///   References:
        ///   <list type="bullet">
        ///     <item><description><a href="http://en.wikipedia.org/wiki/LU_decomposition">LU Decomposition</a></description></item>
        ///   </list></para>
        /// </summary>
        /// <param name="m"></param>
        public void Calculate(double[,] m)
        {
            _decomposition = (double[,])m.Clone();

            var rows = _decomposition.GetLength(0);
            var cols = _decomposition.GetLength(1);
            _pivotSign = 1;

            _pivot = new int[rows];
            for (int i = 0; i < rows; i++)
                _pivot[i] = i;

            var LUcolj = new double[rows];

            unsafe
            {
                fixed (double* LU = _decomposition)
                {
                    // Outer loop.
                    for (int j = 0; j < cols; j++)
                    {
                        // Make a copy of the j-th column to localize references.
                        for (int i = 0; i < rows; i++)
                            LUcolj[i] = _decomposition[i, j];

                        // Apply previous transformations.
                        for (int i = 0; i < rows; i++)
                        {
                            double s = 0;

                            // Most of the time is spent in
                            // the following dot product:
                            int kmax = Math.Min(i, j);
                            double* LUrowi = &LU[i * cols];
                            for (int k = 0; k < kmax; k++)
                                s += LUrowi[k] * LUcolj[k];

                            LUrowi[j] = LUcolj[i] -= s;
                        }

                        // Find pivot and exchange if necessary.
                        int p = j;
                        for (int i = j + 1; i < rows; i++)
                        {
                            if (Math.Abs(LUcolj[i]) > Math.Abs(LUcolj[p]))
                                p = i;
                        }

                        if (p != j)
                        {
                            for (int k = 0; k < cols; k++)
                            {
                                var t = _decomposition[p, k];
                                _decomposition[p, k] = _decomposition[j, k];
                                _decomposition[j, k] = t;
                            }

                            int v = _pivot[p];
                            _pivot[p] = _pivot[j];
                            _pivot[j] = v;

                            _pivotSign = -_pivotSign;
                        }

                        // Compute multipliers.
                        if (j < rows && _decomposition[j, j] != 0)
                        {
                            for (int i = j + 1; i < rows; i++)
                                _decomposition[i, j] /= _decomposition[j, j];
                        }
                    }
                }
            }
        }
        /// <summary>
        ///     Calculates determinant of the matrix using precalculated decomposition.
        /// <para>
        ///   References:
        ///   <list type="bullet">
        ///     <item><description><a href="http://www.proofwiki.org/wiki/Determinant_of_a_Triangular_Matrix">Determinant of a Triangular Matrix</a></description></item>
        ///   </list></para>
        /// </summary>
        /// <returns></returns>
        public double Determinant()
        {
            if (_decomposition == null)
            {
                throw new ApplicationException("Decomposition matrix not calculated");
            }
            var n = _decomposition.GetLength(0);
            var product = (double)_pivotSign;
            for (int i = 0; i < n; i++)
            {
                product = product * _decomposition[i, i];
            }
            return product;
        }


        public double InvertDeterminant()
        {
            return 1 / Determinant();
        }

        public double[,] Inverse(double[,] m)
        {
            var rows = m.GetLength(0);

            if (_decomposition == null)
            {
                Calculate(m);
            }
            int count = m.GetLength(0);

            // Copy right hand side with pivoting
            var inverse = new double[rows, rows];
            for (int i = 0; i < rows; i++)
            {
                int k = _pivot[i];
                inverse[i, k] = 1;
            }

            // Solve L*Y = B(piv,:)
            for (int k = 0; k < rows; k++)
                for (int i = k + 1; i < rows; i++)
                    for (int j = 0; j < count; j++)
                        inverse[i, j] -= inverse[k, j] * _decomposition[i, k];

            // Solve U*X = I;
            for (int k = rows - 1; k >= 0; k--)
            {
                for (int j = 0; j < count; j++)
                    inverse[k, j] /= _decomposition[k, k];

                for (int i = 0; i < k; i++)
                    for (int j = 0; j < count; j++)
                        inverse[i, j] -= inverse[k, j] * _decomposition[i, k];
            }

            return inverse;
        }
    }
}
