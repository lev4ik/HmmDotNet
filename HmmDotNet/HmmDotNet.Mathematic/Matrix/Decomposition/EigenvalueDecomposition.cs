using System;
using System.Diagnostics;
using HmmDotNet.Mathematic.Extentions;

namespace HmmDotNet.Mathematic.MatrixDecomposition
{
    public class EigenvalueDecomposition : IDecomposition<double>
    {
        #region Private Variables

        private double[][] _hessenbergForm; //H
        private double[] _workingStorage; //ort
        private double[] _workingEigenVector1; //d
        private double[] _workingEigenVector2; //e
        private double[][] _eigenVectors; // V
        private int _rows;
        private int _cols;

        #endregion Private Variables
        
        public double[][] Eigenvectors
        {
            get
            {
                if (_eigenVectors != null)
                {
                    Debug.WriteLine(new Matrix(_eigenVectors));
                }
                return _eigenVectors;
            }
        }

        public double[][] Eigenvalues
        {
            get
            {
                if (_workingEigenVector1 != null)
                {
                    Debug.WriteLine(new Vector(_workingEigenVector1));
                }
                var D = new double[_rows][];
                for (var i = 0; i < _cols; i++)
                {
                    D[i] = new double[_cols];
                    for (var j = 0; j < _cols; j++)
                    {
                        D[i][j] = 0.0;
                    }
                    D[i][i] = _workingEigenVector1[i];
                    if (_workingEigenVector2[i] > 0)
                    {
                        D[i][i + 1] = _workingEigenVector2[i];
                    }
                    else if (_workingEigenVector2[i] < 0)
                    {
                        D[i][i - 1] = _workingEigenVector2[i];
                    }
                }
                
                
                return D;
            }
        }

        public double[,] Decomposition { get; private set; }
        public double[,] Upper { get; private set; }
        public double[,] Lower { get; private set; }
        
        public void Calculate(double[,] m)
        {
            _rows = m.GetLength(0);
            _cols = m.GetLength(1);

            double[][] A = m.Convert();
            _eigenVectors = new double[_cols][];
            for (int i = 0; i < _cols; i++)
            {
                _eigenVectors[i] = new double[_cols];
            }
            _workingEigenVector1 = new double[_cols];
            _workingEigenVector2 = new double[_cols];

            var issymmetric = true;
            for (int j = 0; (j < _cols) & issymmetric; j++)
            {
                for (int i = 0; (i < _cols) & issymmetric; i++)
                {
                    issymmetric = (A[i][j] == A[j][i]);
                }
            }

            if (issymmetric)
            {
                for (int i = 0; i < _cols; i++)
                {
                    for (int j = 0; j < _cols; j++)
                    {
                        _eigenVectors[i][j] = A[i][j];
                    }
                }

                // Tridiagonalize.
                tred2();

                // Diagonalize.
                tql2();
            }
            else
            {
                _hessenbergForm = new double[_cols][];
                for (int i2 = 0; i2 < _cols; i2++)
                {
                    _hessenbergForm[i2] = new double[_cols];
                }
                _workingStorage = new double[_cols];

                for (int j = 0; j < _cols; j++)
                {
                    for (int i = 0; i < _cols; i++)
                    {
                        _hessenbergForm[i][j] = A[i][j];
                    }
                }

                // Reduce to Hessenberg form.
                orthes();

                // Reduce Hessenberg to real Schur form.
                hqr2();
            }
        }

        public double Determinant()
        {
            throw new System.NotImplementedException();
        }

        public double InvertDeterminant()
        {
            throw new System.NotImplementedException();
        }

        public double[,] Inverse(double[,] m)
        {
            throw new System.NotImplementedException();
        }

        #region Private Methods

        private void tred2()
        {
            //  This is derived from the Algol procedures tred2 by
            //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
            //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
            //  Fortran subroutine in EISPACK.

            for (int j = 0; j < _cols; j++)
            {
                _workingEigenVector1[j] = _eigenVectors[_cols - 1][j];
            }

            // Householder reduction to tridiagonal form.

            for (int i = _cols - 1; i > 0; i--)
            {
                // Scale to avoid under/overflow.

                double scale = 0.0;
                double h = 0.0;
                for (int k = 0; k < i; k++)
                {
                    scale = scale + Math.Abs(_workingEigenVector1[k]);
                }
                if (scale == 0.0)
                {
                    _workingEigenVector2[i] = _workingEigenVector1[i - 1];
                    for (int j = 0; j < i; j++)
                    {
                        _workingEigenVector1[j] = _eigenVectors[i - 1][j];
                        _eigenVectors[i][j] = 0.0;
                        _eigenVectors[j][i] = 0.0;
                    }
                }
                else
                {
                    // Generate Householder vector.

                    for (int k = 0; k < i; k++)
                    {
                        _workingEigenVector1[k] /= scale;
                        h += _workingEigenVector1[k] * _workingEigenVector1[k];
                    }
                    double f = _workingEigenVector1[i - 1];
                    double g = Math.Sqrt(h);
                    if (f > 0)
                    {
                        g = -g;
                    }
                    _workingEigenVector2[i] = scale * g;
                    h = h - f * g;
                    _workingEigenVector1[i - 1] = f - g;
                    for (int j = 0; j < i; j++)
                    {
                        _workingEigenVector2[j] = 0.0;
                    }

                    // Apply similarity transformation to remaining columns.

                    for (int j = 0; j < i; j++)
                    {
                        f = _workingEigenVector1[j];
                        _eigenVectors[j][i] = f;
                        g = _workingEigenVector2[j] + _eigenVectors[j][j] * f;
                        for (int k = j + 1; k <= i - 1; k++)
                        {
                            g += _eigenVectors[k][j] * _workingEigenVector1[k];
                            _workingEigenVector2[k] += _eigenVectors[k][j] * f;
                        }
                        _workingEigenVector2[j] = g;
                    }
                    f = 0.0;
                    for (int j = 0; j < i; j++)
                    {
                        _workingEigenVector2[j] /= h;
                        f += _workingEigenVector2[j] * _workingEigenVector1[j];
                    }
                    double hh = f / (h + h);
                    for (int j = 0; j < i; j++)
                    {
                        _workingEigenVector2[j] -= hh * _workingEigenVector1[j];
                    }
                    for (int j = 0; j < i; j++)
                    {
                        f = _workingEigenVector1[j];
                        g = _workingEigenVector2[j];
                        for (int k = j; k <= i - 1; k++)
                        {
                            _eigenVectors[k][j] -= (f * _workingEigenVector2[k] + g * _workingEigenVector1[k]);
                        }
                        _workingEigenVector1[j] = _eigenVectors[i - 1][j];
                        _eigenVectors[i][j] = 0.0;
                    }
                }
                _workingEigenVector1[i] = h;
            }

            // Accumulate transformations.

            for (int i = 0; i < _cols - 1; i++)
            {
                _eigenVectors[_cols - 1][i] = _eigenVectors[i][i];
                _eigenVectors[i][i] = 1.0;
                double h = _workingEigenVector1[i + 1];
                if (h != 0.0)
                {
                    for (int k = 0; k <= i; k++)
                    {
                        _workingEigenVector1[k] = _eigenVectors[k][i + 1] / h;
                    }
                    for (int j = 0; j <= i; j++)
                    {
                        double g = 0.0;
                        for (int k = 0; k <= i; k++)
                        {
                            g += _eigenVectors[k][i + 1] * _eigenVectors[k][j];
                        }
                        for (int k = 0; k <= i; k++)
                        {
                            _eigenVectors[k][j] -= g * _workingEigenVector1[k];
                        }
                    }
                }
                for (int k = 0; k <= i; k++)
                {
                    _eigenVectors[k][i + 1] = 0.0;
                }
            }
            for (int j = 0; j < _cols; j++)
            {
                _workingEigenVector1[j] = _eigenVectors[_cols - 1][j];
                _eigenVectors[_cols - 1][j] = 0.0;
            }
            _eigenVectors[_cols - 1][_cols - 1] = 1.0;
            _workingEigenVector2[0] = 0.0;
        }

        private void tql2()
        {
            //  This is derived from the Algol procedures tql2, by
            //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
            //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
            //  Fortran subroutine in EISPACK.

            for (int i = 1; i < _cols; i++)
            {
                _workingEigenVector2[i - 1] = _workingEigenVector2[i];
            }
            _workingEigenVector2[_cols - 1] = 0.0;

            double f = 0.0;
            double tst1 = 0.0;
            double eps = Math.Pow(2.0, -52.0);
            for (int l = 0; l < _cols; l++)
            {
                // Find small subdiagonal element

                tst1 = Math.Max(tst1, Math.Abs(_workingEigenVector1[l]) + Math.Abs(_workingEigenVector2[l]));
                int m = l;
                while (m < _cols)
                {
                    if (Math.Abs(_workingEigenVector2[m]) <= eps * tst1)
                    {
                        break;
                    }
                    m++;
                }

                // If m == l, d[l] is an eigenvalue,
                // otherwise, iterate.

                if (m > l)
                {
                    int iter = 0;
                    do
                    {
                        iter = iter + 1; // (Could check iteration count here.)

                        // Compute implicit shift

                        double g = _workingEigenVector1[l];
                        double p = (_workingEigenVector1[l + 1] - g) / (2.0 * _workingEigenVector2[l]);
                        double r = MathExtention.Hypot(p, 1.0);
                        if (p < 0)
                        {
                            r = -r;
                        }
                        _workingEigenVector1[l] = _workingEigenVector2[l] / (p + r);
                        _workingEigenVector1[l + 1] = _workingEigenVector2[l] * (p + r);
                        double dl1 = _workingEigenVector1[l + 1];
                        double h = g - _workingEigenVector1[l];
                        for (int i = l + 2; i < _cols; i++)
                        {
                            _workingEigenVector1[i] -= h;
                        }
                        f = f + h;

                        // Implicit QL transformation.

                        p = _workingEigenVector1[m];
                        double c = 1.0;
                        double c2 = c;
                        double c3 = c;
                        double el1 = _workingEigenVector2[l + 1];
                        double s = 0.0;
                        double s2 = 0.0;
                        for (int i = m - 1; i >= l; i--)
                        {
                            c3 = c2;
                            c2 = c;
                            s2 = s;
                            g = c * _workingEigenVector2[i];
                            h = c * p;
                            r = MathExtention.Hypot(p, _workingEigenVector2[i]);
                            _workingEigenVector2[i + 1] = s * r;
                            s = _workingEigenVector2[i] / r;
                            c = p / r;
                            p = c * _workingEigenVector1[i] - s * g;
                            _workingEigenVector1[i + 1] = h + s * (c * g + s * _workingEigenVector1[i]);

                            // Accumulate transformation.

                            for (int k = 0; k < _cols; k++)
                            {
                                h = _eigenVectors[k][i + 1];
                                _eigenVectors[k][i + 1] = s * _eigenVectors[k][i] + c * h;
                                _eigenVectors[k][i] = c * _eigenVectors[k][i] - s * h;
                            }
                        }
                        p = (-s) * s2 * c3 * el1 * _workingEigenVector2[l] / dl1;
                        _workingEigenVector2[l] = s * p;
                        _workingEigenVector1[l] = c * p;

                        // Check for convergence.
                    }
                    while (Math.Abs(_workingEigenVector2[l]) > eps * tst1);
                }
                _workingEigenVector1[l] = _workingEigenVector1[l] + f;
                _workingEigenVector2[l] = 0.0;
            }

            // Sort eigenvalues and corresponding vectors.

            for (int i = 0; i < _cols - 1; i++)
            {
                int k = i;
                double p = _workingEigenVector1[i];
                for (int j = i + 1; j < _cols; j++)
                {
                    if (_workingEigenVector1[j] < p)
                    {
                        k = j;
                        p = _workingEigenVector1[j];
                    }
                }
                if (k != i)
                {
                    _workingEigenVector1[k] = _workingEigenVector1[i];
                    _workingEigenVector1[i] = p;
                    for (int j = 0; j < _cols; j++)
                    {
                        p = _eigenVectors[j][i];
                        _eigenVectors[j][i] = _eigenVectors[j][k];
                        _eigenVectors[j][k] = p;
                    }
                }
            }
        }

        private void orthes()
        {
            //  This is derived from the Algol procedures orthes and ortran,
            //  by Martin and Wilkinson, Handbook for Auto. Comp.,
            //  Vol.ii-Linear Algebra, and the corresponding
            //  Fortran subroutines in EISPACK.

            int low = 0;
            int high = _cols - 1;

            for (int m = low + 1; m <= high - 1; m++)
            {

                // Scale column.

                double scale = 0.0;
                for (int i = m; i <= high; i++)
                {
                    scale = scale + Math.Abs(_hessenbergForm[i][m - 1]);
                }
                if (scale != 0.0)
                {

                    // Compute Householder transformation.

                    double h = 0.0;
                    for (int i = high; i >= m; i--)
                    {
                        _workingStorage[i] = _hessenbergForm[i][m - 1] / scale;
                        h += _workingStorage[i] * _workingStorage[i];
                    }
                    double g = System.Math.Sqrt(h);
                    if (_workingStorage[m] > 0)
                    {
                        g = -g;
                    }
                    h = h - _workingStorage[m] * g;
                    _workingStorage[m] = _workingStorage[m] - g;

                    // Apply Householder similarity transformation
                    // H = (I-u*u'/h)*H*(I-u*u')/h)

                    for (int j = m; j < _cols; j++)
                    {
                        double f = 0.0;
                        for (int i = high; i >= m; i--)
                        {
                            f += _workingStorage[i] * _hessenbergForm[i][j];
                        }
                        f = f / h;
                        for (int i = m; i <= high; i++)
                        {
                            _hessenbergForm[i][j] -= f * _workingStorage[i];
                        }
                    }

                    for (int i = 0; i <= high; i++)
                    {
                        double f = 0.0;
                        for (int j = high; j >= m; j--)
                        {
                            f += _workingStorage[j] * _hessenbergForm[i][j];
                        }
                        f = f / h;
                        for (int j = m; j <= high; j++)
                        {
                            _hessenbergForm[i][j] -= f * _workingStorage[j];
                        }
                    }
                    _workingStorage[m] = scale * _workingStorage[m];
                    _hessenbergForm[m][m - 1] = scale * g;
                }
            }

            // Accumulate transformations (Algol's ortran).

            for (int i = 0; i < _cols; i++)
            {
                for (int j = 0; j < _cols; j++)
                {
                    _eigenVectors[i][j] = (i == j ? 1.0 : 0.0);
                }
            }

            for (int m = high - 1; m >= low + 1; m--)
            {
                if (_hessenbergForm[m][m - 1] != 0.0)
                {
                    for (int i = m + 1; i <= high; i++)
                    {
                        _workingStorage[i] = _hessenbergForm[i][m - 1];
                    }
                    for (int j = m; j <= high; j++)
                    {
                        double g = 0.0;
                        for (int i = m; i <= high; i++)
                        {
                            g += _workingStorage[i] * _eigenVectors[i][j];
                        }
                        // Double division avoids possible underflow
                        g = (g / _workingStorage[m]) / _hessenbergForm[m][m - 1];
                        for (int i = m; i <= high; i++)
                        {
                            _eigenVectors[i][j] += g * _workingStorage[i];
                        }
                    }
                }
            }
        }

        private void hqr2()
        {
            //  This is derived from the Algol procedure hqr2,
            //  by Martin and Wilkinson, Handbook for Auto. Comp.,
            //  Vol.ii-Linear Algebra, and the corresponding
            //  Fortran subroutine in EISPACK.

            // Initialize

            int nn = _cols;
            int n = nn - 1;
            int low = 0;
            int high = nn - 1;
            double eps = Math.Pow(2.0, -52.0);
            double exshift = 0.0;
            double p = 0, q = 0, r = 0, s = 0, z = 0, t, w, x, y;

            // Store roots isolated by balanc and compute matrix norm

            double norm = 0.0;
            for (int i = 0; i < nn; i++)
            {
                if (i < low | i > high)
                {
                    _workingEigenVector1[i] = _hessenbergForm[i][i];
                    _workingEigenVector2[i] = 0.0;
                }
                for (int j = Math.Max(i - 1, 0); j < nn; j++)
                {
                    norm = norm + Math.Abs(_hessenbergForm[i][j]);
                }
            }

            // Outer loop over eigenvalue index

            int iter = 0;
            while (n >= low)
            {

                // Look for single small sub-diagonal element

                int l = n;
                while (l > low)
                {
                    s = Math.Abs(_hessenbergForm[l - 1][l - 1]) + Math.Abs(_hessenbergForm[l][l]);
                    if (s == 0.0)
                    {
                        s = norm;
                    }
                    if (Math.Abs(_hessenbergForm[l][l - 1]) < eps * s)
                    {
                        break;
                    }
                    l--;
                }

                // Check for convergence
                // One root found

                if (l == n)
                {
                    _hessenbergForm[n][n] = _hessenbergForm[n][n] + exshift;
                    _workingEigenVector1[n] = _hessenbergForm[n][n];
                    _workingEigenVector2[n] = 0.0;
                    n--;
                    iter = 0;

                    // Two roots found
                }
                else if (l == n - 1)
                {
                    w = _hessenbergForm[n][n - 1] * _hessenbergForm[n - 1][n];
                    p = (_hessenbergForm[n - 1][n - 1] - _hessenbergForm[n][n]) / 2.0;
                    q = p * p + w;
                    z = Math.Sqrt(Math.Abs(q));
                    _hessenbergForm[n][n] = _hessenbergForm[n][n] + exshift;
                    _hessenbergForm[n - 1][n - 1] = _hessenbergForm[n - 1][n - 1] + exshift;
                    x = _hessenbergForm[n][n];

                    // Real pair

                    if (q >= 0)
                    {
                        if (p >= 0)
                        {
                            z = p + z;
                        }
                        else
                        {
                            z = p - z;
                        }
                        _workingEigenVector1[n - 1] = x + z;
                        _workingEigenVector1[n] = _workingEigenVector1[n - 1];
                        if (z != 0.0)
                        {
                            _workingEigenVector1[n] = x - w / z;
                        }
                        _workingEigenVector2[n - 1] = 0.0;
                        _workingEigenVector2[n] = 0.0;
                        x = _hessenbergForm[n][n - 1];
                        s = Math.Abs(x) + Math.Abs(z);
                        p = x / s;
                        q = z / s;
                        r = Math.Sqrt(p * p + q * q);
                        p = p / r;
                        q = q / r;

                        // Row modification

                        for (int j = n - 1; j < nn; j++)
                        {
                            z = _hessenbergForm[n - 1][j];
                            _hessenbergForm[n - 1][j] = q * z + p * _hessenbergForm[n][j];
                            _hessenbergForm[n][j] = q * _hessenbergForm[n][j] - p * z;
                        }

                        // Column modification

                        for (int i = 0; i <= n; i++)
                        {
                            z = _hessenbergForm[i][n - 1];
                            _hessenbergForm[i][n - 1] = q * z + p * _hessenbergForm[i][n];
                            _hessenbergForm[i][n] = q * _hessenbergForm[i][n] - p * z;
                        }

                        // Accumulate transformations

                        for (int i = low; i <= high; i++)
                        {
                            z = _eigenVectors[i][n - 1];
                            _eigenVectors[i][n - 1] = q * z + p * _eigenVectors[i][n];
                            _eigenVectors[i][n] = q * _eigenVectors[i][n] - p * z;
                        }

                        // Complex pair
                    }
                    else
                    {
                        _workingEigenVector1[n - 1] = x + p;
                        _workingEigenVector1[n] = x + p;
                        _workingEigenVector2[n - 1] = z;
                        _workingEigenVector2[n] = -z;
                    }
                    n = n - 2;
                    iter = 0;

                    // No convergence yet
                }
                else
                {

                    // Form shift

                    x = _hessenbergForm[n][n];
                    y = 0.0;
                    w = 0.0;
                    if (l < n)
                    {
                        y = _hessenbergForm[n - 1][n - 1];
                        w = _hessenbergForm[n][n - 1] * _hessenbergForm[n - 1][n];
                    }

                    // Wilkinson's original ad hoc shift

                    if (iter == 10)
                    {
                        exshift += x;
                        for (int i = low; i <= n; i++)
                        {
                            _hessenbergForm[i][i] -= x;
                        }
                        s = Math.Abs(_hessenbergForm[n][n - 1]) + Math.Abs(_hessenbergForm[n - 1][n - 2]);
                        x = y = 0.75 * s;
                        w = (-0.4375) * s * s;
                    }

                    // MATLAB's new ad hoc shift

                    if (iter == 30)
                    {
                        s = (y - x) / 2.0;
                        s = s * s + w;
                        if (s > 0)
                        {
                            s = Math.Sqrt(s);
                            if (y < x)
                            {
                                s = -s;
                            }
                            s = x - w / ((y - x) / 2.0 + s);
                            for (int i = low; i <= n; i++)
                            {
                                _hessenbergForm[i][i] -= s;
                            }
                            exshift += s;
                            x = y = w = 0.964;
                        }
                    }

                    iter = iter + 1; // (Could check iteration count here.)

                    // Look for two consecutive small sub-diagonal elements

                    int m = n - 2;
                    while (m >= l)
                    {
                        z = _hessenbergForm[m][m];
                        r = x - z;
                        s = y - z;
                        p = (r * s - w) / _hessenbergForm[m + 1][m] + _hessenbergForm[m][m + 1];
                        q = _hessenbergForm[m + 1][m + 1] - z - r - s;
                        r = _hessenbergForm[m + 2][m + 1];
                        s = Math.Abs(p) + Math.Abs(q) + Math.Abs(r);
                        p = p / s;
                        q = q / s;
                        r = r / s;
                        if (m == l)
                        {
                            break;
                        }
                        if (Math.Abs(_hessenbergForm[m][m - 1]) * (Math.Abs(q) + Math.Abs(r)) < eps * (Math.Abs(p) * (Math.Abs(_hessenbergForm[m - 1][m - 1]) + Math.Abs(z) + Math.Abs(_hessenbergForm[m + 1][m + 1]))))
                        {
                            break;
                        }
                        m--;
                    }

                    for (int i = m + 2; i <= n; i++)
                    {
                        _hessenbergForm[i][i - 2] = 0.0;
                        if (i > m + 2)
                        {
                            _hessenbergForm[i][i - 3] = 0.0;
                        }
                    }

                    // Double QR step involving rows l:n and columns m:n

                    for (int k = m; k <= n - 1; k++)
                    {
                        bool notlast = (k != n - 1);
                        if (k != m)
                        {
                            p = _hessenbergForm[k][k - 1];
                            q = _hessenbergForm[k + 1][k - 1];
                            r = (notlast ? _hessenbergForm[k + 2][k - 1] : 0.0);
                            x = Math.Abs(p) + Math.Abs(q) + Math.Abs(r);
                            if (x != 0.0)
                            {
                                p = p / x;
                                q = q / x;
                                r = r / x;
                            }
                        }
                        if (x == 0.0)
                        {
                            break;
                        }
                        s = Math.Sqrt(p * p + q * q + r * r);
                        if (p < 0)
                        {
                            s = -s;
                        }
                        if (s != 0)
                        {
                            if (k != m)
                            {
                                _hessenbergForm[k][k - 1] = (-s) * x;
                            }
                            else if (l != m)
                            {
                                _hessenbergForm[k][k - 1] = -_hessenbergForm[k][k - 1];
                            }
                            p = p + s;
                            x = p / s;
                            y = q / s;
                            z = r / s;
                            q = q / p;
                            r = r / p;

                            // Row modification

                            for (int j = k; j < nn; j++)
                            {
                                p = _hessenbergForm[k][j] + q * _hessenbergForm[k + 1][j];
                                if (notlast)
                                {
                                    p = p + r * _hessenbergForm[k + 2][j];
                                    _hessenbergForm[k + 2][j] = _hessenbergForm[k + 2][j] - p * z;
                                }
                                _hessenbergForm[k][j] = _hessenbergForm[k][j] - p * x;
                                _hessenbergForm[k + 1][j] = _hessenbergForm[k + 1][j] - p * y;
                            }

                            // Column modification

                            for (int i = 0; i <= Math.Min(n, k + 3); i++)
                            {
                                p = x * _hessenbergForm[i][k] + y * _hessenbergForm[i][k + 1];
                                if (notlast)
                                {
                                    p = p + z * _hessenbergForm[i][k + 2];
                                    _hessenbergForm[i][k + 2] = _hessenbergForm[i][k + 2] - p * r;
                                }
                                _hessenbergForm[i][k] = _hessenbergForm[i][k] - p;
                                _hessenbergForm[i][k + 1] = _hessenbergForm[i][k + 1] - p * q;
                            }

                            // Accumulate transformations

                            for (int i = low; i <= high; i++)
                            {
                                p = x * _eigenVectors[i][k] + y * _eigenVectors[i][k + 1];
                                if (notlast)
                                {
                                    p = p + z * _eigenVectors[i][k + 2];
                                    _eigenVectors[i][k + 2] = _eigenVectors[i][k + 2] - p * r;
                                }
                                _eigenVectors[i][k] = _eigenVectors[i][k] - p;
                                _eigenVectors[i][k + 1] = _eigenVectors[i][k + 1] - p * q;
                            }
                        } // (s != 0)
                    } // k loop
                } // check convergence
            } // while (n >= low)

            // Backsubstitute to find vectors of upper triangular form

            if (norm == 0.0)
            {
                return;
            }

            for (n = nn - 1; n >= 0; n--)
            {
                p = _workingEigenVector1[n];
                q = _workingEigenVector2[n];

                // Real vector

                if (q == 0)
                {
                    int l = n;
                    _hessenbergForm[n][n] = 1.0;
                    for (int i = n - 1; i >= 0; i--)
                    {
                        w = _hessenbergForm[i][i] - p;
                        r = 0.0;
                        for (int j = l; j <= n; j++)
                        {
                            r = r + _hessenbergForm[i][j] * _hessenbergForm[j][n];
                        }
                        if (_workingEigenVector2[i] < 0.0)
                        {
                            z = w;
                            s = r;
                        }
                        else
                        {
                            l = i;
                            if (_workingEigenVector2[i] == 0.0)
                            {
                                if (w != 0.0)
                                {
                                    _hessenbergForm[i][n] = (-r) / w;
                                }
                                else
                                {
                                    _hessenbergForm[i][n] = (-r) / (eps * norm);
                                }

                                // Solve real equations
                            }
                            else
                            {
                                x = _hessenbergForm[i][i + 1];
                                y = _hessenbergForm[i + 1][i];
                                q = (_workingEigenVector1[i] - p) * (_workingEigenVector1[i] - p) + _workingEigenVector2[i] * _workingEigenVector2[i];
                                t = (x * s - z * r) / q;
                                _hessenbergForm[i][n] = t;
                                if (Math.Abs(x) > Math.Abs(z))
                                {
                                    _hessenbergForm[i + 1][n] = (-r - w * t) / x;
                                }
                                else
                                {
                                    _hessenbergForm[i + 1][n] = (-s - y * t) / z;
                                }
                            }

                            // Overflow control

                            t = Math.Abs(_hessenbergForm[i][n]);
                            if ((eps * t) * t > 1)
                            {
                                for (int j = i; j <= n; j++)
                                {
                                    _hessenbergForm[j][n] = _hessenbergForm[j][n] / t;
                                }
                            }
                        }
                    }

                    // Complex vector
                }
                else if (q < 0)
                {
                    int l = n - 1;

                    // Last vector component imaginary so matrix is triangular

                    if (Math.Abs(_hessenbergForm[n][n - 1]) > Math.Abs(_hessenbergForm[n - 1][n]))
                    {
                        _hessenbergForm[n - 1][n - 1] = q / _hessenbergForm[n][n - 1];
                        _hessenbergForm[n - 1][n] = (-(_hessenbergForm[n][n] - p)) / _hessenbergForm[n][n - 1];
                    }
                    else
                    {
                        cdiv(0.0, -_hessenbergForm[n - 1][n], _hessenbergForm[n - 1][n - 1] - p, q);
                        _hessenbergForm[n - 1][n - 1] = cdivr;
                        _hessenbergForm[n - 1][n] = cdivi;
                    }
                    _hessenbergForm[n][n - 1] = 0.0;
                    _hessenbergForm[n][n] = 1.0;
                    for (int i = n - 2; i >= 0; i--)
                    {
                        double ra, sa, vr, vi;
                        ra = 0.0;
                        sa = 0.0;
                        for (int j = l; j <= n; j++)
                        {
                            ra = ra + _hessenbergForm[i][j] * _hessenbergForm[j][n - 1];
                            sa = sa + _hessenbergForm[i][j] * _hessenbergForm[j][n];
                        }
                        w = _hessenbergForm[i][i] - p;

                        if (_workingEigenVector2[i] < 0.0)
                        {
                            z = w;
                            r = ra;
                            s = sa;
                        }
                        else
                        {
                            l = i;
                            if (_workingEigenVector2[i] == 0)
                            {
                                cdiv(-ra, -sa, w, q);
                                _hessenbergForm[i][n - 1] = cdivr;
                                _hessenbergForm[i][n] = cdivi;
                            }
                            else
                            {

                                // Solve complex equations

                                x = _hessenbergForm[i][i + 1];
                                y = _hessenbergForm[i + 1][i];
                                vr = (_workingEigenVector1[i] - p) * (_workingEigenVector1[i] - p) + _workingEigenVector2[i] * _workingEigenVector2[i] - q * q;
                                vi = (_workingEigenVector1[i] - p) * 2.0 * q;
                                if (vr == 0.0 & vi == 0.0)
                                {
                                    vr = eps * norm * (Math.Abs(w) + Math.Abs(q) + Math.Abs(x) + Math.Abs(y) + Math.Abs(z));
                                }
                                cdiv(x * r - z * ra + q * sa, x * s - z * sa - q * ra, vr, vi);
                                _hessenbergForm[i][n - 1] = cdivr;
                                _hessenbergForm[i][n] = cdivi;
                                if (Math.Abs(x) > (Math.Abs(z) + Math.Abs(q)))
                                {
                                    _hessenbergForm[i + 1][n - 1] = (-ra - w * _hessenbergForm[i][n - 1] + q * _hessenbergForm[i][n]) / x;
                                    _hessenbergForm[i + 1][n] = (-sa - w * _hessenbergForm[i][n] - q * _hessenbergForm[i][n - 1]) / x;
                                }
                                else
                                {
                                    cdiv(-r - y * _hessenbergForm[i][n - 1], -s - y * _hessenbergForm[i][n], z, q);
                                    _hessenbergForm[i + 1][n - 1] = cdivr;
                                    _hessenbergForm[i + 1][n] = cdivi;
                                }
                            }

                            // Overflow control

                            t = System.Math.Max(System.Math.Abs(_hessenbergForm[i][n - 1]), System.Math.Abs(_hessenbergForm[i][n]));
                            if ((eps * t) * t > 1)
                            {
                                for (int j = i; j <= n; j++)
                                {
                                    _hessenbergForm[j][n - 1] = _hessenbergForm[j][n - 1] / t;
                                    _hessenbergForm[j][n] = _hessenbergForm[j][n] / t;
                                }
                            }
                        }
                    }
                }
            }

            // Vectors of isolated roots

            for (int i = 0; i < nn; i++)
            {
                if (i < low | i > high)
                {
                    for (int j = i; j < nn; j++)
                    {
                        _eigenVectors[i][j] = _hessenbergForm[i][j];
                    }
                }
            }

            // Back transformation to get eigenvectors of original matrix

            for (int j = nn - 1; j >= low; j--)
            {
                for (int i = low; i <= high; i++)
                {
                    z = 0.0;
                    for (int k = low; k <= Math.Min(j, high); k++)
                    {
                        z = z + _eigenVectors[i][k] * _hessenbergForm[k][j];
                    }
                    _eigenVectors[i][j] = z;
                }
            }
        }

        private double cdivr, cdivi;

        private void cdiv(double xr, double xi, double yr, double yi)
        {
            double r, d;
            if (Math.Abs(yr) > Math.Abs(yi))
            {
                r = yi / yr;
                d = yr + r * yi;
                cdivr = (xr + r * xi) / d;
                cdivi = (xi - r * xr) / d;
            }
            else
            {
                r = yr / yi;
                d = yi + r * yr;
                cdivr = (r * xr + xi) / d;
                cdivi = (r * xi - xr) / d;
            }
        }

        #endregion Private Methods
    }
}