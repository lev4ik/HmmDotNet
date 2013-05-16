using System;
using System.Diagnostics;
using System.Text;
using HmmDotNet.Mathematic.MatrixDecomposition;
using HmmDotNet.Mathematic.Extentions;

namespace HmmDotNet.Mathematic
{
    public class Matrix : IMatrix<double>
    {
        #region Members

        private double[,] _m;
        private double? _determinant;
        private IDecomposition<double> _decomposition;
        private bool? _isSemetricAndPositiviDefinite;
        private bool? _isPositiviDefinite;
        private bool? _isSemetric;

        #endregion Members

        #region Constructors

        public Matrix(double[,] m)
        {
            _m = m.Duplicate();
            _decomposition = new LU();
        }

        public Matrix(double[][] m)
        {
            var rows = m.Length;
            var columns = m[0].Length;
            _m = new double[rows,columns];
            for (int i = 0; i < m.Length; i++)
            {
                for (int j = 0; j < m[0].Length; j++)
                {
                    _m[i, j] = m[i][j];
                }
            }
            _decomposition = new LU();
        }

        #endregion Constructors

        #region Properties

        public int Rows
        {
            get { return _m.GetLength(0); }
        }

        public int Columns
        {
            get { return _m.GetLength(1); }
        }

        public bool Semetric
        {
            get
            {
                if (!_isSemetric.HasValue)
                {
                    _isSemetric = _m.IsSemetric();
                }
                return _isSemetric.Value;
            }
        }

        public bool PositiviDefinite
        {
            get
            {
                if (!_isPositiviDefinite.HasValue)
                {
                    _isPositiviDefinite = _m.IsPositiveDefinite();
                }
                return _isPositiviDefinite.Value;
            }
        }

        public bool SemetricAndPositiviDefinite
        {
            get
            {
                if (!_isSemetricAndPositiviDefinite.HasValue)
                {
                    _isSemetricAndPositiviDefinite = _m.IsSemetricAndPositiveDefinite();
                }
                return _isSemetricAndPositiviDefinite.Value;
            }
        }

        public double Determinant
        {
            get
            {
                if (!PositiviDefinite)
                {
                    Debug.WriteLine(new Matrix(_m));
                    throw new ApplicationException("Matrix is not Semetric And Positive Definite");
                }
                if (_determinant.HasValue)
                {
                    return _determinant.Value;
                }
                if (Rows != Columns)
                {
                    throw new ApplicationException("Matrix must be n x n");
                }
                // Sum over all left to right with plus                
                _decomposition.Calculate(_m);
                _determinant = _decomposition.Determinant();

                return _determinant.Value;
            }
        }

        public double DeterminantInverse
        {
            get
            {
                if (Rows != Columns)
                {
                    throw new ApplicationException("Matrix must be n x n");
                } 
                if (_determinant.HasValue)
                {
                    return 1 / _determinant.Value;
                }

                _decomposition.Calculate(_m);
                return _decomposition.InvertDeterminant();
            }
        }

        public double[,] M
        {
            get { return _m; }
        }

        #endregion Properties

        #region Helper Methods

        public override string ToString()
        {
            var builder = new StringBuilder();
            builder.Append("{");
            for (int i = 0; i < Rows; i++)
            {
                builder.Append("{");
                for (int j = 0; j < Columns; j++)
                {
                    builder.Append(_m[i, j]);
                    if (j < Columns - 1)
                    {
                        builder.Append(",");
                    }
                }
                builder.Append("}");
                if (i < Rows - 1)
                {
                    builder.Append(",");
                }
            }
            builder.Append("}");
            return builder.ToString();
        }

        #endregion Helper Methods

        #region Methods

        public double[,] ConvertToPositiveDefinite()
        {
            var esp = Math.Pow(10.0, -6);
            var zero = Math.Pow(10.0, -10);
            var decomp = new EigenvalueDecomposition();
            decomp.Calculate(_m);
            var eiginvalues = decomp.Eigenvalues.Convert();//new double[,] { { 4333.79, 0, 0, 0 }, 
                                               //{ 0, 149.91, 0, 0 }, 
                                               //{ 0, 0, 3.71021 * 10e-12, 0 }, 
                                               //{ 0, 0, 0, -3.41178 * 10e-12 } };//decomp.Eigenvalues;
            var eigenvector = decomp.Eigenvectors.Convert(); //new double[,]{{-0.501785, -0.483579, -0.441634, -0.565086}, 
                                            //{-0.664266, 0.150482, -0.277177, 0.677701}, 
                                            //{0.532041, -0.476915, -0.581008, 0.389762},
                                            //{0.15458, 0.718372, -0.62495, -0.263599}};//decomp.Eigenvectors.Convert();

            for (var i = 0; i < Columns; i++)
            {
                if (eiginvalues[i, i] <= zero)
                {
                    eiginvalues[i, i] = esp;
                }
            }

            var newMatrix = eigenvector.Product(eiginvalues);
            var transpose = new Matrix(eigenvector).Transpose();
            var result = newMatrix.Product(transpose);

            return result;
        }

        public double[] Diagonal()
        {
            var diagonal = new double[Rows];
            for (int i = 0; i < Rows; i++)
            {
                diagonal[i] = _m[i, i];
            }
            return diagonal;
        }

        public double[,] Identity()
        {
            var m = new double[Rows, Columns];
            for (int i = 0; i < Rows; i++)
            {
                m[i, i] = 1;
            }
            return m;
        }

        #region Product
  
        public double[,] Product(double[,] B)
        {
            return _m.Product(B);
        }

        public double[] Product(double[] b)
        {
            return _m.Product(b);
        }

        public double[,] Product(double x)
        {
            return _m.Product(x);
        }

        #endregion Product

        public double[,] Transpose()
        {
            var transpose = new double[Columns, Rows];
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    transpose[j, i] = _m[i, j];
                }
            }
            return transpose;
        }

        public double[,] Inverse()
        {
            return _decomposition.Inverse(_m);
        }

        #endregion Methods
    }
}
