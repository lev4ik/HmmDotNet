using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Mathematic.Extentions;

namespace HmmDotNet.Logic.Test.Mathematic
{
    [TestClass]
    public class VectorExtentionsTest
    {
        private double[] _vector;

        [TestInitialize]
        public void TestInitializer()
        {
            _vector = new double[4];
            _vector[0] = 61.4820;
            _vector[1] = 62.3977;
            _vector[2] = 60.5985;
            _vector[3] = 61.5268;
        }

        [TestMethod]
        public void Product_4LengthVectorAndx_4LengthVectorMultipliedByX()
        {
            var prod = _vector.Product(10d);

            Assert.AreEqual(614.820, Math.Round(prod[0], 3));
            Assert.AreEqual(623.977, Math.Round(prod[1], 3));
            Assert.AreEqual(605.985, Math.Round(prod[2], 3));
            Assert.AreEqual(615.268, Math.Round(prod[3], 3));
        }

        [TestMethod]
        public void Product_4LengthVectorAnd4LengthVector_4LengthVector()
        {
            var v = new double[] {10d, 10d, 10d, 10d};
            var prod = _vector.Product(v);

            Assert.AreEqual(2460.05, prod);
        }

        [TestMethod]
        public void OuterProductJagged_4LengthVectorAnd4LengthVector_4x4Matrix()
        {
            var v = new double[] { 10d, 10d, 10d, 10d };
            var prod = _vector.OuterProductJagged(v);

            Assert.AreEqual(614.820, Math.Round(prod[0][0], 3));
            Assert.AreEqual(614.820, Math.Round(prod[0][1], 3));
            Assert.AreEqual(614.820, Math.Round(prod[0][2], 3));
            Assert.AreEqual(614.820, Math.Round(prod[0][3], 3));
            Assert.AreEqual(623.977, Math.Round(prod[1][0], 3));
            Assert.AreEqual(623.977, Math.Round(prod[1][1], 3));
            Assert.AreEqual(623.977, Math.Round(prod[1][2], 3));
            Assert.AreEqual(623.977, Math.Round(prod[1][3], 3));
            Assert.AreEqual(605.985, Math.Round(prod[2][0], 3));
            Assert.AreEqual(605.985, Math.Round(prod[2][1], 3));
            Assert.AreEqual(605.985, Math.Round(prod[2][2], 3));
            Assert.AreEqual(605.985, Math.Round(prod[2][3], 3));
            Assert.AreEqual(615.268, Math.Round(prod[3][0], 3));
            Assert.AreEqual(615.268, Math.Round(prod[3][1], 3));
            Assert.AreEqual(615.268, Math.Round(prod[3][2], 3));
            Assert.AreEqual(615.268, Math.Round(prod[3][3], 3));
        }

        [TestMethod]
        public void OuterProduct_4LengthVectorAnd4LengthVector_4x4Matrix()
        {
            var v = new double[] { 10d, 10d, 10d, 10d };
            var prod = _vector.OuterProduct(v);

            Assert.AreEqual(614.820, Math.Round(prod[0,0], 3));
            Assert.AreEqual(614.820, Math.Round(prod[0,1], 3));
            Assert.AreEqual(614.820, Math.Round(prod[0,2], 3));
            Assert.AreEqual(614.820, Math.Round(prod[0,3], 3));
            Assert.AreEqual(623.977, Math.Round(prod[1,0], 3));
            Assert.AreEqual(623.977, Math.Round(prod[1,1], 3));
            Assert.AreEqual(623.977, Math.Round(prod[1,2], 3));
            Assert.AreEqual(623.977, Math.Round(prod[1,3], 3));
            Assert.AreEqual(605.985, Math.Round(prod[2,0], 3));
            Assert.AreEqual(605.985, Math.Round(prod[2,1], 3));
            Assert.AreEqual(605.985, Math.Round(prod[2,2], 3));
            Assert.AreEqual(605.985, Math.Round(prod[2,3], 3));
            Assert.AreEqual(615.268, Math.Round(prod[3,0], 3));
            Assert.AreEqual(615.268, Math.Round(prod[3,1], 3));
            Assert.AreEqual(615.268, Math.Round(prod[3,2], 3));
            Assert.AreEqual(615.268, Math.Round(prod[3,3], 3));
        }

        [TestMethod]
        public void Add_4LengthVectorAnd4LengthVector_4Vector()
        {
            var v = new double[] { 10d, 10d, 10d, 10d };
            var prod = _vector.Add(v);

            Assert.AreEqual(71.4820, Math.Round(prod[0], 4));
            Assert.AreEqual(72.3977, Math.Round(prod[1], 4));
            Assert.AreEqual(70.5985, Math.Round(prod[2], 4));
            Assert.AreEqual(71.5268, Math.Round(prod[3], 4));
        }

        [TestMethod]
        public void Add_4LengthVectorAndScalar_4Vector()
        {
            var prod = _vector.Add(10d);

            Assert.AreEqual(71.4820, Math.Round(prod[0], 4));
            Assert.AreEqual(72.3977, Math.Round(prod[1], 4));
            Assert.AreEqual(70.5985, Math.Round(prod[2], 4));
            Assert.AreEqual(71.5268, Math.Round(prod[3], 4));
        }

        [TestMethod]
        public void Substruct_4LengthVectorAnd4LengthVector_4Vector()
        {
            var v = new double[] { 10d, 10d, 10d, 10d };
            var prod = _vector.Substruct(v);

            Assert.AreEqual(51.4820, Math.Round(prod[0], 4));
            Assert.AreEqual(52.3977, Math.Round(prod[1], 4));
            Assert.AreEqual(50.5985, Math.Round(prod[2], 4));
            Assert.AreEqual(51.5268, Math.Round(prod[3], 4));
        }

        [TestMethod]
        public void Substruct_4LengthVectorAndScalar_4Vector()
        {
            var prod = _vector.Substruct(10d);

            Assert.AreEqual(51.4820, Math.Round(prod[0], 4));
            Assert.AreEqual(52.3977, Math.Round(prod[1], 4));
            Assert.AreEqual(50.5985, Math.Round(prod[2], 4));
            Assert.AreEqual(51.5268, Math.Round(prod[3], 4));
        }

        [TestMethod]
        public void Sum_4LengthVector_4VectorMembersSum()
        {
            var prod = _vector.Sum();

            Assert.AreEqual(246.005, Math.Round(prod, 3));
        }

        [TestMethod]
        public void EqualTo_Two4LengthEqualVectors_True()
        {
            Assert.IsTrue(_vector.EqualsTo(_vector));
        }

        [TestMethod]
        public void EqualTo_Two4LengthNotEqualVectors_False()
        {
            var v = new double[] { 10d, 10d, 10d, 10d };
            Assert.IsFalse(_vector.EqualsTo(v));
        }

        [TestMethod]
        public void Init_ValueAnd4LengthVector_4LengthVector()
        {
            var prod = _vector.Init(10d);

            Assert.AreEqual(6.14820, Math.Round(prod[0], 5));
            Assert.AreEqual(6.23977, Math.Round(prod[1], 5));
            Assert.AreEqual(6.05985, Math.Round(prod[2], 5));
            Assert.AreEqual(6.15268, Math.Round(prod[3], 5));
        }

        [TestMethod]
        public void Mean_4LengthVector_4LengthVectorMean()
        {
            var prod = _vector.Mean();

            Assert.AreEqual(61.50125, Math.Round(prod, 5));
        }
    }
}
