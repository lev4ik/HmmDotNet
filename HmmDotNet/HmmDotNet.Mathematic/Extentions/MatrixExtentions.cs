using System;

namespace HmmDotNet.Mathematic.Extentions
{
    public static class MatrixExtentions
    {
        public static bool IsSemetric(this double[,] m)
        {
            var rows = m.GetLength(0);
            var cols = m.GetLength(1);

            var res = (rows == cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (i != j)
                    {
                        res = res & Math.Round(m[i, j], 10).EqualsTo(Math.Round(m[j, i], 10));
                    }
                }
            }

            return res;
        }

        public static bool IsPositiveDefinite(this double[,] m)
        {
            var rows = m.GetLength(0);
            var cols = m.GetLength(1);

            var A = m.Convert();
            var L = new double[rows][];
            var res = (rows == cols);

            for (var j = 0; j < rows; j++)
            {
                L[j] = new double[cols];
                var d = 0.0;
                for (var k = 0; k < j; k++)
                {
                    var s = 0.0;
                    for (var i = 0; i < k; i++)
                    {
                        s += L[k][i] * L[j][i];
                    }
                    L[j][k] = s = (A[j][k] - s) / L[k][k];
                    d += s * s;                    
                }
                d = A[j][j] - d;
                res = res & (d > 0.0);
                L[j][j] = Math.Sqrt(Math.Max(d, 0.0));
                for (var k = j + 1; k < rows; k++)
                {
                    L[j][k] = 0.0;
                }
            }

            return res;
        }

        public static bool IsSemetricAndPositiveDefinite(this double[,] m)
        {
            var rows = m.GetLength(0);
            var cols = m.GetLength(1);

            var A = m.Convert();
            var L = new double[rows][];
            var res = (rows == cols);

            for (var j = 0; j < rows; j++)
            {
                L[j] = new double[cols];
                var d = 0.0;
                for (var k = 0; k < j; k++)
                {
                    var s = 0.0;
                    for (var i = 0; i < k; i++)
                    {
                        s += L[k][i] * L[j][i];
                    }
                    L[j][k] = s = (A[j][k] - s) / L[k][k];
                    d += s * s;
                    res = res & (A[k][j] == A[j][k]);
                }
                d = A[j][j] - d;
                res = res & (d > 0.0);
                L[j][j] = Math.Sqrt(Math.Max(d, 0.0));
                for (var k = j + 1; k < rows; k++)
                {
                    L[j][k] = 0.0;
                }
            }

            return res;
        }

        public static bool In(this double[][] m, double[] value)
        {
            var rows = m.Length;

            for (int i = 0; i < rows; i++)
            {
                if (m[i].EqualsTo(value))
                {
                    return true;
                }
            }

            return false;
        }

        public static double[] Mean(this double[][] m)
        {
            var rows = m.Length;
            var columns = m[0].Length;

            var mean = new double[columns];

            for (int i = 0; i < columns; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    mean[i] += m[j][i];
                }                
            }
            
            return mean.Init(m.Length);
        }

        public static double[][] Append(this double[][] m, double[] value)
        {
            var N = m.Length;
            var arr = new double[N + 1][];
            for (int i = 0; i < N; i++)
            {
                arr[i] = m[i];
            }
            arr[N] = value;

            return arr;
        }

        #region Duplicate

        public static double[,] Duplicate(this double[,] m)
        {
            var result = new double[m.GetLength(0), m.GetLength(1)];
            for (var i = 0; i < m.GetLength(0); ++i)
            {
                for (var j = 0; j < m.GetLength(1); ++j)
                {
                    result[i, j] = m[i, j];
                }
            }
            return result;
        }

        #endregion Duplicate

        #region General
        
        public static double[][] Convert(this double[,] m)
        {
            var rows = m.GetLength(0);
            var cols = m.GetLength(1);
            var result = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                result[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    result[i][j] = m[i, j];
                }
            }
            return result;
        }

        public static double[,] Convert(this double[][] m)
        {
            var rows = m.Length;
            var cols = m[0].Length;
            var result = new double[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m[i][j];
                }
            }
            return result;
        }

        public static void Init<T>(this T[,] m, T value)
        {
            var rows = m.GetLength(0);
            var cols = m.GetLength(1);
            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < cols; j++)
                {
                    m[i, j] = value;
                }
            }
        }
        /// <summary>
        ///     Exchange row i with j
        /// </summary>
        /// <param name="m"></param>
        /// <param name="i">Row</param>
        /// <param name="j">Row</param>
        public static void ExchangeRows(this double[,] m, int i, int j)
        {
            var columns = m.GetLength(1);
            for (var k = 0; k < columns; k++)
            {
                var tmp = m[i, k];
                m[i, k] = m[j, k];
                m[j, k] = tmp;
            }
        }

        public static double[] GetColumn(this double[][] m, int column)
        {
            var columns = m[0].Length;
            var result = new double[m.Length];

            for (int i = 0; i < columns; i++)
            {
                result[i] = m[i][column];
            }

            return result;
        }

        public static double[] GetColumn(this double[,] m, int column)
        {
            var rows = m.GetLength(0);
            var result = new double[m.GetLength(0)];

            for (int i = 0; i < rows; i++)
            {
                result[i] = m[i, column];
            }

            return result;
        }

        public static double[] GetRow(this double[][] a, int row)
        {
            var rows = a.Length;
            var result = new double[a[0].Length];

            for (var i = 0; i < rows; i++)
            {
                result[i] = a[i][row];
            }
            return result;
        }

        public static double[] GetRow(this double[,] a, int row)
        {
            var rows = a.GetLength(0);
            var result = new double[a.GetLength(1)];

            for (var i = 0; i < rows; i++)
            {
                result[i] = a[i,row];
            }
            return result;
        }

        #endregion General

        #region Equals

        public static bool EqualsTo(this double[,] a, double[,] b)
        {
            var rows = a.GetLength(0);
            var cols = a.GetLength(1);


            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < cols; j++)
                {
                    if (!a[i, j].EqualsTo(b[i, j]))
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        public static bool EqualsTo(this double[][] a, double[][] b)
        {
            var rows = a.Length;
            var cols = a[0].Length;
            var result = true;

            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < cols; j++)
                {
                    if (!a[i][j].EqualsTo(b[i][j]))
                    {
                        return false;
                    }
                }
            }
            return result;
        }

        #endregion Equals

        #region Sum

        public static double Sum(this double[,] a)
        {
            var rows = a.GetLength(0);
            var cols = a.GetLength(1);
            var sum = 0.0d;
            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < cols; j++)
                {
                    sum = sum + a[i, j];
                }
            }

            return sum;
        }

        #endregion Sum

        #region Add
        /// <summary>
        ///     Add two matrixes
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double[,] Add(this double[,] a, double[,] b)
        {
            if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
            {
                throw new ApplicationException("Dimensions of matrix A must be equal to dimensions of matrix B");
            }
            
            var rows = a.GetLength(0);
            var cols = a.GetLength(1);
            var c = new double[rows, cols];

            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < cols; j++)
                {
                    c[i, j] = a[i, j] + b[i, j];
                }
            }

            return c;
        }

        #endregion Add

        #region Substract
        /// <summary>
        ///     Substructs two matrixes
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double[,] Substruct(this double[,] a, double[,] b)
        {
            if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
            {
                throw new ApplicationException("Dimensions of matrix A must be equal to dimensions of matrix B");
            }

            var rows = a.GetLength(0);
            var cols = a.GetLength(1);
            var c = new double[rows, cols];

            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < cols; j++)
                {
                    c[i, j] = a[i, j] - b[i, j];
                }
            }

            return c;
        }

        #endregion Substract

        #region Product
        /// <summary>
        ///     Matrix with matrix product
        /// </summary>
        /// <param name="m">Matrix</param>
        /// <param name="B">Matrix</param>
        /// <returns>Matrix</returns>
        public static double[,] Product(this double[,] m, double[,] B)
        {
            var aRows = m.GetLength(0);
            var aCols = m.GetLength(1);
            var bRows = B.GetLength(0);
            var bCols = B.GetLength(1);
            if (aCols != bRows)
                throw new ArgumentException("Non-conformable matrices");

            var result = new double[aRows, bCols];

            for (var i = 0; i < aRows; ++i)
                for (var j = 0; j < bCols; ++j)
                    for (var k = 0; k < aCols; ++k)
                        result[i,j] += m[i,k] * B[k,j];

            return result;
        }

        /// <summary>
        ///     Matrix with scalar product
        /// </summary>
        /// <param name="m">Matrix</param>
        /// <param name="x">Scalar</param>
        /// <returns>Matrix</returns>
        public static double[,] Product(this double[,] m, double x)
        {
            var rows = m.GetLength(1);
            var columns = m.GetLength(0);

            var product = new double[rows, columns];

            for (var row = 0; row < rows; row++)
            {
                for (var column = 0; column < columns; column++)
                {
                    product[row, column] = m[row, column] * x;
                }
            }

            return product;
        }

        /// <summary>
        ///     Matrix with vector product
        /// </summary>
        /// <param name="m">Matrix</param>
        /// <param name="b">Vector</param>
        /// <returns>Vector</returns>
        public static double[] Product(this double[,] m, double[] b)
        {
            if (m.GetLength(0) != b.Length)
                throw new ArgumentException("Non-conformable matrix and vector");

            var columns = b.Length;
            var product = new double[columns];
            for (var i = 0; i < columns; i++)
            {
                for (var j = 0; j < columns; j++)
                {
                    product[i] += b[i] * m[i, j];
                }               
            }

            return product;
        }

        #endregion Product
    }
}
