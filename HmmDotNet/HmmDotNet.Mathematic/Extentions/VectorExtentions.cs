using System;

namespace HmmDotNet.Mathematic.Extentions
{
    public static class VectorExtentions
    {
        public static double[] Round(this double[] v, int digits)
        {
            var result = new double[v.Length];
            for (var i = 0; i < v.Length; i++)
            {
                result[i] = Math.Round(v[i], digits);
            }
            return result;
        }

        public static bool In(this int[] v, int value)
        {
            for (int i = 0; i < v.Length; i++)
            {
                if (v[i] == value)
                {
                    return true;
                }
            }

            return false;
        }

        #region Product

        /// <summary>
        /// 
        /// </summary>
        /// <param name="v">Vector</param>
        /// <param name="x">Scalar</param>
        /// <returns></returns>
        public static double[] Product(this double[] v, double x)
        {
            var N = v.Length;
            var res = new double[N];
            for (int i = 0; i < N; i++)
            {
                res[i] = v[i] * x;
            }
            return res;
        }

        /// <summary>
        ///     Product 1xN * Nx1 = a
        /// </summary>
        /// <param name="v">Vector</param>
        /// <param name="x">Vector</param>
        /// <returns></returns>
        public static double Product(this double[] v, double[] x)
        {
            if (v.Length != x.Length)
            {
                throw new ApplicationException("Array must be of the same length");
            }

            var N = v.Length;
            var res = 0d;
            for (var i = 0; i < N; i++)
            {
                res += v[i] * x[i];
            }

            return res;
        }

        /// <summary>
        ///     Outer Product Nx1 * 1xN = NxN
        /// </summary>
        /// <param name="v">Vector</param>
        /// <param name="x">Vector</param>
        /// <returns></returns>
        public static double[][] OuterProductJagged(this double[] v, double[] x)
        {
            if (v.Length != x.Length)
            {
                throw new ApplicationException("Array must be of the same length");
            }

            var N = v.Length;
            var res = new double[N][];
            for (var i = 0; i < N; i++)
            {
                res[i] = new double[N];
                for (var j = 0; j < N; j++)
                {
                    res[i][j] = v[i] * x[j];
                }
            }

            return res;
        }

        /// <summary>
        ///     Outer Product Nx1 * 1xN = NxN
        /// </summary>
        /// <param name="v">Vector</param>
        /// <param name="x">Vector</param>
        /// <returns></returns>
        public static double[,] OuterProduct(this double[] v, double[] x)
        {
            var N = v.Length;
            var K = x.Length;
            var res = new double[N, K];
            for (var i = 0; i < N; i++)
            {
                for (var j = 0; j < K; j++)
                {
                    res[i, j] = v[i] * x[j];
                }
            }

            return res;
        }

        #endregion Product

        #region Add

        public static double[] Add(this double[] v, double[] w)
        {
            if (v.Length != w.Length)
            {
                throw new ApplicationException("Array must be of the same size");
            }
            var N = v.Length;
            var res = new double[N];
            for (var i = 0; i < N; i++)
            {
                res[i] = v[i] + w[i];
            }
            return res;
        }

        public static double[] Add(this double[] v, double x)
        {
            var N = v.Length;
            var res = new double[N];
            for (var i = 0; i < N; i++)
            {
                res[i] = v[i] + x;
            }
            return res;
        }

        #endregion Add

        #region Substruct

        public static double[] Substruct(this double[] v, double a)
        {
            var N = v.Length;
            var res = new double[N];
            for (var i = 0; i < N; i++)
            {
                res[i] = v[i] - a;
            }
            return res;
        }

        public static double[] Substruct(this double[] v, double[] a)
        {
            var N = v.Length;
            var res = new double[N];
            for (var i = 0; i < N; i++)
            {
                res[i] = v[i] - a[i];
            }
            return res;
        }

        #endregion Substruct

        #region Sum

        public static double Sum(this double[] v)
        {
            var N = v.Length;
            var res = 0d;
            for (int i = 0; i < N; i++)
            {
                res = res + v[i];
            }
            return res;
        }

        #endregion Sum

        #region Equals

        public static bool EqualsTo(this double[] a, double[] b)
        {
            for (var i = 0; i < a.Length; i++)
            {
                if (!a[i].EqualsTo(b[i]))
                {
                    return false;
                }
            }
            return true;
        }

        #endregion Equals

        #region Initialize

        public static double[] Init(this double[] a, double value)
        {
            for (var i = 0; i < a.Length; i++)
            {
                a[i] = a[i] / value;
            }

            return a;
        }

        #endregion Initialize

        #region Mean

        public static double Mean(this double[] vector)
        {
            return vector.Sum() / vector.Length;
        }

        #endregion Mean
    }
}
