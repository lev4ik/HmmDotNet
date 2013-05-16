using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Mathematic;

namespace HmmDotNet.Logic.Test.Mathematic
{
    [TestClass]
    public class VectorTest
    {
        private Vector _vector;
 
        [TestInitialize]
        public void TestInitializer()
        {
            var vector = new double[4];
            vector[0] = 61.4820;
            vector[1] = 62.3977;
            vector[2] = 60.5985;
            vector[3] = 61.5268;

            _vector = new Vector(vector);
        }

        [TestMethod]
        public void Constructor_4LengthVector_DimentionIs4()
        {
            Assert.AreEqual(4, _vector.Dimention);
        }

        [TestMethod]
        public void Constructor_4LengthVector_4LengthVectorEqualToV()
        {
            Assert.AreEqual(61.4820, Math.Round(_vector.V[0], 4));
            Assert.AreEqual(62.3977, Math.Round(_vector.V[1], 4));
            Assert.AreEqual(60.5985, Math.Round(_vector.V[2], 4));
            Assert.AreEqual(61.5268, Math.Round(_vector.V[3], 4));
        }

        [TestMethod]
        public void Product_Two4LengthVectors_Scalar()
        {
            var v = new double[] { 10d, 10d, 10d, 10d };
            var prod = _vector.Product(v);

            Assert.AreEqual(2460.05, prod);
        }

        [TestMethod]
        public void Product_4LengthVectorAndScalar_4LengthVectorMultipliedByScalar()
        {
            var prod = _vector.Product(10d);

            Assert.AreEqual(614.820, Math.Round(prod[0], 3));
            Assert.AreEqual(623.977, Math.Round(prod[1], 3));
            Assert.AreEqual(605.985, Math.Round(prod[2], 3));
            Assert.AreEqual(615.268, Math.Round(prod[3], 3));            
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
            var prod = _vector.Substract(v);

            Assert.AreEqual(51.4820, Math.Round(prod[0], 4));
            Assert.AreEqual(52.3977, Math.Round(prod[1], 4));
            Assert.AreEqual(50.5985, Math.Round(prod[2], 4));
            Assert.AreEqual(51.5268, Math.Round(prod[3], 4));
        }

        [TestMethod]
        public void Substruct_4LengthVectorAndScalar_4Vector()
        {
            var prod = _vector.Substract(10d);

            Assert.AreEqual(51.4820, Math.Round(prod[0], 4));
            Assert.AreEqual(52.3977, Math.Round(prod[1], 4));
            Assert.AreEqual(50.5985, Math.Round(prod[2], 4));
            Assert.AreEqual(51.5268, Math.Round(prod[3], 4));
        }

        [TestMethod]
        public void OuterProduct_RowVectorAndColumnVector_Matrix()
        {
            var a = new double[2];
            a[0] = 1;
            a[1] = 2;
            var b = new double[2];
            b[0] = 3;
            b[1] = 4;
            var v = new Vector(a);
            var m = v.OuterProduct(b);
            Assert.AreEqual(m[0, 0], 3);
            Assert.AreEqual(m[0, 1], 4);
            Assert.AreEqual(m[1, 0], 6);
            Assert.AreEqual(m[1, 1], 8);
        }

        [TestMethod]
        public void CrossProduct_RowVectorAndRowVector_Scalar()
        {
            var a = new double[2];
            a[0] = 1;
            a[1] = 2;
            var b = new double[2];
            b[0] = 3;
            b[1] = 4;
            var v = new Vector(a);
            var res = v.Product(b);
            Assert.AreEqual(res, 11);
        }
    }
}
