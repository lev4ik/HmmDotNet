using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Mathematic;

namespace HmmDotNet.Logic.Test.Mathematic
{
    [TestClass]
    public class MatrixTest
    {
        private Matrix _matrix;

        [TestInitialize]
        public void TestInitializer()
        {
            var matrix = new double[4, 4];
            matrix[0, 0] = 1196.756;
            matrix[0, 1] = 1214.831;
            matrix[0, 2] = 1177.636;
            matrix[0, 3] = 1197.112;
            matrix[1, 0] = 1214.831;
            matrix[1, 1] = 1234.175;
            matrix[1, 2] = 1195.806;
            matrix[1, 3] = 1216.329;
            matrix[2, 0] = 1177.636;
            matrix[2, 1] = 1195.806;
            matrix[2, 2] = 1159.749;
            matrix[2, 3] = 1179.056;
            matrix[3, 0] = 1197.112;
            matrix[3, 1] = 1216.329;
            matrix[3, 2] = 1179.056;
            matrix[3, 3] = 1199.651;

            _matrix = new Matrix(matrix);
        }

        [TestMethod]
        public void Constructor_JaggedArray_MatrixInitialized()
        {
            var matrix = new double[4][];
            matrix[0] = new double[] {1196.756,1214.831,1177.636,1197.112};
            matrix[1] = new double[] {1214.831,1234.175,1195.806,1216.329};
            matrix[2] = new double[] {1177.636,1195.806,1159.749,1179.056};
            matrix[3] = new double[] {1197.112,1216.329,1179.056,1199.651};

            var m = new Matrix(matrix);

            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Columns; j++)
                {
                    Assert.AreEqual(_matrix.M[i,j], m.M[i,j]);
                }
            }
        }

        [TestMethod]
        public void Constructor_4x4Matrix_RowsEquals4()
        {
            Assert.AreEqual(4, _matrix.Rows);
        }
        
        [TestMethod]
        public void Constructor_4x4Matrix_ColumnsEquals4()
        {
            Assert.AreEqual(4, _matrix.Columns);
        }

        [TestMethod]
        public void Constructor_4x4Matrix_MatrixEquals4x4Matrix()
        {
            _matrix.M[0, 0] = 1196.756;
            _matrix.M[0, 1] = 1214.831;
            _matrix.M[0, 2] = 1177.636;
            _matrix.M[0, 3] = 1197.112;
            _matrix.M[1, 0] = 1214.831;
            _matrix.M[1, 1] = 1234.175;
            _matrix.M[1, 2] = 1195.806;
            _matrix.M[1, 3] = 1216.329;
            _matrix.M[2, 0] = 1177.636;
            _matrix.M[2, 1] = 1195.806;
            _matrix.M[2, 2] = 1159.749;
            _matrix.M[2, 3] = 1179.056;
            _matrix.M[3, 0] = 1197.112;
            _matrix.M[3, 1] = 1216.329;
            _matrix.M[3, 2] = 1179.056;
            _matrix.M[3, 3] = 1199.651;            
        }

        [TestMethod]
        public void Determinant_4x4Matrix_PositiveDeterminant()
        {
            Assert.AreEqual(347.505, Math.Round(_matrix.Determinant, 3));
        }

        [TestMethod]
        public void InverseDeterminant_4x4Matix_PositiveInverseDeterminant()
        {
            Assert.AreEqual(Math.Round(1 / 347.505, 3), Math.Round(_matrix.DeterminantInverse, 3));
        }

        [TestMethod]
        public void Inverse_4x4Matrix_InverseMatrixCalculated()
        {
            var inverse = _matrix.Inverse();

            Assert.AreEqual(2.577, Math.Round(inverse[0, 0], 3));
            Assert.AreEqual(-2.142, Math.Round(inverse[0, 1], 3));
            Assert.AreEqual(-2.144, Math.Round(inverse[0, 2], 3));
            Assert.AreEqual(1.708, Math.Round(inverse[0, 3], 3));
            Assert.AreEqual(-2.142, Math.Round(inverse[1, 0], 3));
            Assert.AreEqual(3.032, Math.Round(inverse[1, 1], 3));
            Assert.AreEqual(1.302, Math.Round(inverse[1, 2], 3));
            Assert.AreEqual(-2.217, Math.Round(inverse[1, 3], 3));
            Assert.AreEqual(-2.144, Math.Round(inverse[2, 0], 3));
            Assert.AreEqual(1.302, Math.Round(inverse[2, 1], 3));
            Assert.AreEqual(3.038, Math.Round(inverse[2, 2], 3));
            Assert.AreEqual(-2.167, Math.Round(inverse[2, 3], 3));
            Assert.AreEqual(1.708, Math.Round(inverse[3, 0], 3));
            Assert.AreEqual(-2.217, Math.Round(inverse[3, 1], 3));
            Assert.AreEqual(-2.167, Math.Round(inverse[3, 2], 3));
            Assert.AreEqual(2.674, Math.Round(inverse[3, 3], 3));
        }

        [TestMethod]
        public void Diagonal_3x3Matrix_DiagonalVectorReturned()
        {
            var m = new double[3, 3];
            m[0, 0] = 3;
            m[0, 1] = 6;
            m[0, 2] = -9;
            m[1, 0] = 2;
            m[1, 1] = 5;
            m[1, 2] = -3;
            m[2, 0] = -4;
            m[2, 1] = 1;
            m[2, 2] = 10;
            var matrix = new Matrix(m);
            var diagonal = matrix.Diagonal();
            for (int i = 0; i < matrix.Rows; i++)
            {
                Assert.AreEqual(diagonal[i], m[i,i]);
            }            
        }

        [TestMethod]
        public void Determinant_4x4Matrix_DeterminantReturned()
        {
            var m = new double[4, 4];
            m[0, 0] = 1196.756;
            m[0, 1] = 1214.831;
            m[0, 2] = 1177.636;
            m[0, 3] = 1197.112;
            m[1, 0] = 1214.831;
            m[1, 1] = 1234.175;
            m[1, 2] = 1195.806;
            m[1, 3] = 1216.329;
            m[2, 0] = 1177.636;
            m[2, 1] = 1195.806;
            m[2, 2] = 1159.749;
            m[2, 3] = 1179.056;
            m[3, 0] = 1197.112;
            m[3, 1] = 1216.329;
            m[3, 2] = 1179.056;
            m[3, 3] = 1199.651;
            var matrix = new Matrix(m);
            var det = matrix.Determinant;
            Assert.AreEqual(347.50528495071779, det);
        }

        [TestMethod]
        public void Identity_Size3Matrix_Identity3x3Matrix()
        {
            var size = 3;
            var matrix = new Matrix(new double[size, size]);
            var m = matrix.Identity();
            for (int i = 0; i < size; i++)
            {
                Assert.AreEqual(m[i, i], 1);
            }
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Product_3x4MatrixAnd5x5Matrix_ExceptionTrown()
        {
            var A = new double[3,4];
            var B = new double[5,5];
            var matrix = new Matrix(A);
            matrix.Product(B);
        }

        [TestMethod]
        public void Product_3x4MatrixAnd2x3Matrix_2x4Matrix()
        {
            var A = new double[3, 4];
            A[0, 0] = 14;
            A[0, 1] = 2;
            A[0, 2] = 0;
            A[0, 3] = 5;
            A[1, 0] = 9;
            A[1, 1] = 11;
            A[1, 2] = 12;
            A[1, 3] = 2;            
            A[2, 0] = 3;
            A[2, 1] = 15;
            A[2, 2] = 17;
            A[2, 3] = 3;
            var B = new double[2, 3];
            B[0, 0] = 12;
            B[0, 1] = 9;
            B[0, 2] = 8;
            B[1, 0] = 25;
            B[1, 1] = 10;
            B[1, 2] = 5;

            var matrix = new Matrix(B);
            var product = matrix.Product(A);
            Assert.AreEqual(product[0, 0], 273);
            Assert.AreEqual(product[0, 1], 243);
            Assert.AreEqual(product[0, 2], 244);
            Assert.AreEqual(product[0, 3], 102);
            Assert.AreEqual(product[1, 0], 455);
            Assert.AreEqual(product[1, 1], 235);
            Assert.AreEqual(product[1, 2], 205);
            Assert.AreEqual(product[1, 3], 160);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Product_4x5MatrixAnd6Scalar_ExceptionThrown()
        {
            var A = new double[4, 5];
            var b = new double[6];
            var matrix = new Matrix(A);
            matrix.Product(b);
        }

        [TestMethod]
        public void Product_2x2MatrixAndScalar_2x2Matrix()
        {
            var A = new double[2, 2];
            A[0, 0] = 1;
            A[0, 1] = 2;
            A[1, 0] = 3;
            A[1, 1] = 4;
            var x = 2;
            var matrix = new Matrix(A);
            var product = matrix.Product(x);
            Assert.AreEqual(product[0, 0], 2);
            Assert.AreEqual(product[0, 1], 4);
            Assert.AreEqual(product[1, 0], 6);
            Assert.AreEqual(product[1, 1], 8);
        }

        [TestMethod]
        public void Transpose_4x4MirrorMatrix_TransposeEqualsOriginalMatrix()
        {
            var m = new double[4, 4];
            m[0, 0] = 1196.756;
            m[0, 1] = 1214.831;
            m[0, 2] = 1177.636;
            m[0, 3] = 1197.112;
            m[1, 0] = 1214.831;
            m[1, 1] = 1234.175;
            m[1, 2] = 1195.806;
            m[1, 3] = 1216.329;
            m[2, 0] = 1177.636;
            m[2, 1] = 1195.806;
            m[2, 2] = 1159.749;
            m[2, 3] = 1179.056;
            m[3, 0] = 1197.112;
            m[3, 1] = 1216.329;
            m[3, 2] = 1179.056;
            m[3, 3] = 1199.651;

            var matrix = new Matrix(m);
            var transpose = matrix.Transpose();

            Assert.AreEqual(transpose[0, 0], 1196.756);
            Assert.AreEqual(transpose[0, 1], 1214.831);
            Assert.AreEqual(transpose[0, 2], 1177.636);
            Assert.AreEqual(transpose[0, 3], 1197.112);
            Assert.AreEqual(transpose[1, 0], 1214.831);
            Assert.AreEqual(transpose[1, 1], 1234.175);
            Assert.AreEqual(transpose[1, 2], 1195.806);
            Assert.AreEqual(transpose[1, 3], 1216.329);
            Assert.AreEqual(transpose[2, 0], 1177.636);
            Assert.AreEqual(transpose[2, 1], 1195.806);
            Assert.AreEqual(transpose[2, 2], 1159.749);
            Assert.AreEqual(transpose[2, 3], 1179.056);
            Assert.AreEqual(transpose[3, 0], 1197.112);
            Assert.AreEqual(transpose[3, 1], 1216.329);
            Assert.AreEqual(transpose[3, 2], 1179.056);
            Assert.AreEqual(transpose[3, 3], 1199.651);
        }

        [TestMethod]
        public void Transpose_NonSquareMirrorMatrix_TransposeCalculated()
        {
            var m = new double[3, 2];
            m[0, 0] = 1;
            m[0, 1] = 2;
            m[1, 0] = 3;
            m[1, 1] = 4;
            m[2, 0] = 5;
            m[2, 1] = 6;

            var matrix = new Matrix(m);
            var transpose = matrix.Transpose();
            Assert.AreEqual(transpose[0, 0], 1);
            Assert.AreEqual(transpose[0, 1], 3);
            Assert.AreEqual(transpose[0, 2], 5);
            Assert.AreEqual(transpose[1, 0], 2);
            Assert.AreEqual(transpose[1, 1], 4);
            Assert.AreEqual(transpose[1, 2], 6);
        }

        [TestMethod]
        public void Inverse_3x3Matrix_3x3InverseMatrix()
        {
            var m = new double[3, 3];
            m[0, 0] = 3;
            m[0, 1] = 6;
            m[0, 2] = -9;
            m[1, 0] = 2;
            m[1, 1] = 5;
            m[1, 2] = -3;
            m[2, 0] = -4;
            m[2, 1] = 1;
            m[2, 2] = 10;

            var matrix = new Matrix(m);
            var inverse = matrix.Inverse();
            Assert.AreEqual(Math.Round(inverse[0, 0], 4), Math.Round(-53d / 87d, 4));
            Assert.AreEqual(Math.Round(inverse[0, 1], 4), Math.Round(23d / 29d, 4));
            Assert.AreEqual(Math.Round(inverse[0, 2], 4), Math.Round(-9d / 29d, 4));
            Assert.AreEqual(Math.Round(inverse[1, 0], 4), Math.Round(8d / 87d, 4));
            Assert.AreEqual(Math.Round(inverse[1, 1], 4), Math.Round(2d / 29d, 4));
            Assert.AreEqual(Math.Round(inverse[1, 2], 4), Math.Round(3d / 29d, 4));
            Assert.AreEqual(Math.Round(inverse[2, 0], 4), Math.Round(-22d / 87d, 4));
            Assert.AreEqual(Math.Round(inverse[2, 1], 4), Math.Round(9d / 29d, 4));
            Assert.AreEqual(Math.Round(inverse[2, 2], 4), Math.Round(-1d / 29d, 4));
        }

        [TestMethod]
        public void ConvertToPositiveDefinite_4x4NotPositiveDefinite_PositiveDefinite()
        {
            var m = new double[,]{{1157.34222212636, 1036.61999992791, 987.991111034635, 1161.36666659532}, 
                                  {1036.61999992791, 1016.84666662175, 919.293333279443, 1199.55666662978}, 
                                  {987.991111034635, 919.293333279443, 856.782222162621, 1053.38666661627}, 
                                  {1161.36666659532, 1199.55666662978, 1053.38666661627, 1452.72666664381}};

            var matrix = new Matrix(m);

            var positiveDefinite = matrix.ConvertToPositiveDefinite();

            var positiveDefiniteMatrix = new Matrix(positiveDefinite);

            Assert.IsTrue(positiveDefiniteMatrix.PositiviDefinite);
            Assert.IsTrue(matrix.Semetric);
            Assert.IsTrue(positiveDefiniteMatrix.Semetric);
        }
    }
}