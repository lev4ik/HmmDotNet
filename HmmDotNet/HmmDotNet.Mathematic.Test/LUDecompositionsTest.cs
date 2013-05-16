using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Mathematic.MatrixDecomposition;

namespace HmmDotNet.Logic.Test.Mathematic
{
    [TestClass]
    public class LUDecompositionsTest
    {
        private double[,] _m;

        [TestInitialize]
        public void TestInitialize()
        {
            _m = new double[4, 4];
            _m[0, 0] = 1196.756;
            _m[0, 1] = 1214.831;
            _m[0, 2] = 1177.636;
            _m[0, 3] = 1197.112;
            _m[1, 0] = 1214.831;
            _m[1, 1] = 1234.175;
            _m[1, 2] = 1195.806;
            _m[1, 3] = 1216.329;
            _m[2, 0] = 1177.636;
            _m[2, 1] = 1195.806;
            _m[2, 2] = 1159.749;
            _m[2, 3] = 1179.056;
            _m[3, 0] = 1197.112;
            _m[3, 1] = 1216.329;
            _m[3, 2] = 1179.056;
            _m[3, 3] = 1199.651;            
        }

        [TestMethod]
        public void LUDecomposition_4x4Matrix_DeterminantCalculated()
        {
            var d = new LU();
            d.Calculate(_m);
            
            Assert.AreEqual(347.505, Math.Round(d.Determinant(), 3));
        }

        [TestMethod]
        public void LUDecomposition_4x4Matrix_InvertDeterminantCalculated()
        {
            var d = new LU();
            d.Calculate(_m);

            Assert.AreEqual(Math.Round(1 / 347.505, 5), Math.Round(d.InvertDeterminant(), 5));
        }

        [TestMethod]
        public void LUDecomposition_4x4Matrix_4x4Decomposition()
        {
            var d = new LU();
            d.Calculate(_m);

            Assert.AreEqual(1214.831, Math.Round(d.Decomposition[0, 0], 3));
            Assert.AreEqual(1234.175, Math.Round(d.Decomposition[0, 1], 3));
            Assert.AreEqual(1195.806, Math.Round(d.Decomposition[0, 2], 3));
            Assert.AreEqual(1216.329, Math.Round(d.Decomposition[0, 3], 3));
            Assert.AreEqual(0.985, Math.Round(d.Decomposition[1, 0], 3));
            Assert.AreEqual(-0.981, Math.Round(d.Decomposition[1, 1], 3));
            Assert.AreEqual(-0.378, Math.Round(d.Decomposition[1, 2], 3));
            Assert.AreEqual(-1.120, Math.Round(d.Decomposition[1, 3], 3));
            Assert.AreEqual(0.969, Math.Round(d.Decomposition[2, 0], 3));
            Assert.AreEqual(0.593, Math.Round(d.Decomposition[2, 1], 3));
            Assert.AreEqual(0.780, Math.Round(d.Decomposition[2, 2], 3));
            Assert.AreEqual(0.632, Math.Round(d.Decomposition[2, 3], 3));
            Assert.AreEqual(0.985, Math.Round(d.Decomposition[3, 0], 3));
            Assert.AreEqual(-0.158, Math.Round(d.Decomposition[3, 1], 3));
            Assert.AreEqual(0.810, Math.Round(d.Decomposition[3, 2], 3));
            Assert.AreEqual(0.374, Math.Round(d.Decomposition[3, 3], 3));
        }

        [TestMethod]
        public void LUDecomposition_4x4Matrix_InverseMatrixCalculated()
        {
            var d = new LU();
            var inverse = d.Inverse(_m);

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
    }
}
