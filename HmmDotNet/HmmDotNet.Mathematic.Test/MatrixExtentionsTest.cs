using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Mathematic.Extentions;

namespace HmmDotNet.Logic.Test.Mathematic
{
    [TestClass]
    public class MatrixExtentionsTest
    {
        private double[,] _m;
        private double[][] _mj;

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

            _mj = new double[4][];
            _mj[0] = new double[] { 1196.756, 1214.831, 1177.636, 1197.112 };
            _mj[1] = new double[] { 1214.831, 1234.175, 1195.806, 1216.329 };
            _mj[2] = new double[] { 1177.636, 1195.806, 1159.749, 1179.056 };
            _mj[3] = new double[] { 1197.112, 1216.329, 1179.056, 1199.651 };
        }

        [TestMethod]
        public void GetColumn_Jagged3x3MatrixColumn2_Column2()
        {
            var matrix = new double[3][];
            matrix[0] = new double[] { 1, 2, 3 };
            matrix[1] = new double[] { 4, 5, 6 };
            matrix[2] = new double[] { 7, 8, 9 };

            var column = matrix.GetColumn(2);

            Assert.AreEqual(column[0], 3);
            Assert.AreEqual(column[1], 6);
            Assert.AreEqual(column[2], 9);
        }

        [TestMethod]
        public void GetRows_Jagged3x3MatrixRow2_Row2()
        {
            var matrix = new double[3][];
            matrix[0] = new double[] { 1, 2, 3 };
            matrix[1] = new double[] { 4, 5, 6 };
            matrix[2] = new double[] { 7, 8, 9 };

            var row = matrix.GetRow(2);

            Assert.AreEqual(row[0], 3);
            Assert.AreEqual(row[1], 6);
            Assert.AreEqual(row[2], 9);
        }

        [TestMethod]
        public void Substruct_Two4x4MatrixesWithSameValues_AllZero4x4Matrix()
        {
            var sub = _m.Substruct(_m);

            Assert.AreEqual(sub[0, 0], 0d);
            Assert.AreEqual(sub[0, 1], 0d);
            Assert.AreEqual(sub[0, 2], 0d);
            Assert.AreEqual(sub[0, 3], 0d);
            Assert.AreEqual(sub[1, 0], 0d);
            Assert.AreEqual(sub[1, 1], 0d);
            Assert.AreEqual(sub[1, 2], 0d);
            Assert.AreEqual(sub[1, 3], 0d);
            Assert.AreEqual(sub[2, 0], 0d);
            Assert.AreEqual(sub[2, 1], 0d);
            Assert.AreEqual(sub[2, 2], 0d);
            Assert.AreEqual(sub[2, 3], 0d);
            Assert.AreEqual(sub[3, 0], 0d);
            Assert.AreEqual(sub[3, 1], 0d);
            Assert.AreEqual(sub[3, 2], 0d);
            Assert.AreEqual(sub[3, 3], 0d);
        }

        [TestMethod]
        public void Sum_4x4MultidimentionalMatrix_SumCalculated()
        {
            var sum = _m.Sum();

            Assert.AreEqual(19151.871000000006, sum);
        }

        [TestMethod]
        public void EqualsTo_TwoNotEqual4x4MultidimentionalMatrix_NotEquals()
        {
            var m2 = _m.Duplicate();
            m2.ExchangeRows(1, 2);

            var equals = _m.EqualsTo(m2);

            Assert.IsFalse(equals);
        }

        [TestMethod]
        public void EqualsTo_TwoEqual4x4JaggedArrays_Equals()
        {
            var equals = _mj.EqualsTo(_mj);

            Assert.IsTrue(equals);
        }

        [TestMethod]
        public void EqualsTo_TwoEqual4x4MultidimentionalMatrix_Equals()
        {
            var equals = _m.EqualsTo(_m);

            Assert.IsTrue(equals);
        }

        [TestMethod]
        public void GetColumn_4x4MultidimentionalMatrixAndColumn1_Column1()
        {
            var column = _m.GetColumn(1);

            Assert.AreEqual(1214.831, column[0]);
            Assert.AreEqual(1234.175, column[1]);
            Assert.AreEqual(1195.806, column[2]);
            Assert.AreEqual(1216.329, column[3]);
        }

        [TestMethod]
        public void GetRow_4x4MultidimentionalMatrixAndRow1_Row1()
        {
            var row = _m.GetRow(1);

            Assert.AreEqual(1214.831, row[0]);
            Assert.AreEqual(1234.175, row[1]);
            Assert.AreEqual(1195.806, row[2]);
            Assert.AreEqual(1216.329, row[3]);
        }

        [TestMethod]
        public void ExchangeRows_4x4Matrix_Rows2ExchngedWith1()
        {
            _m.ExchangeRows(1, 2);

            Assert.AreEqual(1177.636, _m[1,0]);
            Assert.AreEqual(1195.806, _m[1,1]);
            Assert.AreEqual(1159.749, _m[1,2]);
            Assert.AreEqual(1179.056, _m[1,3]);

            Assert.AreEqual(1214.831, _m[2,0]);
            Assert.AreEqual(1234.175, _m[2,1]);
            Assert.AreEqual(1195.806, _m[2,2]);
            Assert.AreEqual(1216.329, _m[2,3]);

            _m.ExchangeRows(2, 1);
        }

        [TestMethod]
        public void IsSemetricAndPositiveDefinite_4x4PositiveDefiniteMatrix_True()
        {
            var res = _m.IsSemetricAndPositiveDefinite();

            Assert.IsTrue(res);        
        }

        [TestMethod]
        public void IsSemetricAndPositiveDefinite_4x4NotPositiveDefiniteMatrix_False()
        {
            var m = new double[,]{{1157.34222212636, 1036.61999992791, 987.991111034635, 1161.36666659532}, 
                                  {1036.61999992791, 1016.84666662175, 919.293333279443, 1199.55666662978}, 
                                  {987.991111034635, 919.293333279443, 856.782222162621, 1053.38666661627}, 
                                  {1161.36666659532, 1199.55666662978, 1053.38666661627, 1452.72666664381}};
            
            var res = m.IsSemetricAndPositiveDefinite();

            Assert.IsFalse(res);
        }

        [TestMethod]
        public void Convert_4x4Matrix_4x4JaggedArray()
        {
            var a = _m.Convert();

            Assert.AreEqual(a[0][0], 1196.756);
            Assert.AreEqual(a[0][1], 1214.831);
            Assert.AreEqual(a[0][2], 1177.636);
            Assert.AreEqual(a[0][3], 1197.112);
            Assert.AreEqual(a[1][0], 1214.831);
            Assert.AreEqual(a[1][1], 1234.175);
            Assert.AreEqual(a[1][2], 1195.806);
            Assert.AreEqual(a[1][3], 1216.329);
            Assert.AreEqual(a[2][0], 1177.636);
            Assert.AreEqual(a[2][1], 1195.806);
            Assert.AreEqual(a[2][2], 1159.749);
            Assert.AreEqual(a[2][3], 1179.056);
            Assert.AreEqual(a[3][0], 1197.112);
            Assert.AreEqual(a[3][1], 1216.329);
            Assert.AreEqual(a[3][2], 1179.056);
            Assert.AreEqual(a[3][3], 1199.651);
        }

        [TestMethod]
        public void Duplicate_4x4Matrix_ExactlyTheSame4x4MatrixCreated()
        {
            var a = _m.Duplicate();

            Assert.AreEqual(a[0, 0] , 1196.756);
            Assert.AreEqual(a[0, 1] , 1214.831);
            Assert.AreEqual(a[0, 2] , 1177.636);
            Assert.AreEqual(a[0, 3] , 1197.112);
            Assert.AreEqual(a[1, 0] , 1214.831);
            Assert.AreEqual(a[1, 1] , 1234.175);
            Assert.AreEqual(a[1, 2] , 1195.806);
            Assert.AreEqual(a[1, 3] , 1216.329);
            Assert.AreEqual(a[2, 0] , 1177.636);
            Assert.AreEqual(a[2, 1] , 1195.806);
            Assert.AreEqual(a[2, 2] , 1159.749);
            Assert.AreEqual(a[2, 3] , 1179.056);
            Assert.AreEqual(a[3, 0] , 1197.112);
            Assert.AreEqual(a[3, 1] , 1216.329);
            Assert.AreEqual(a[3, 2] , 1179.056);
            Assert.AreEqual(a[3, 3] , 1199.651);
        }

        [TestMethod]
        public void Append_4x4MatrixAndVector_5x4Matrix()
        {
            var m = new double[4][];
            m[0] = new []{1196.756,1214.831,1177.636,1197.112};
            m[1] = new []{1214.831,1234.175,1195.806,1216.329};
            m[2] = new []{1177.636,1195.806,1159.749,1179.056};
            m[3] = new []{1197.112,1216.329,1179.056,1199.651};

            var vector = new double[] { 1116.756, 1296.756, 2196.756, 1596.756 };

            var result = m.Append(vector);

            Assert.AreEqual(5, result.Length);
            Assert.AreEqual(1116.756, result[4][0]);
            Assert.AreEqual(1296.756, result[4][1]);
            Assert.AreEqual(2196.756, result[4][2]);
            Assert.AreEqual(1596.756, result[4][3]);
        }

        [TestMethod]
        public void Mean_4x4Matrix_MeanVector()
        {
            var m = new double[4][];
            m[0] = new[] { 1196.756, 1214.831, 1177.636, 1197.112 };
            m[1] = new[] { 1214.831, 1234.175, 1195.806, 1216.329 };
            m[2] = new[] { 1177.636, 1195.806, 1159.749, 1179.056 };
            m[3] = new[] { 1197.112, 1216.329, 1179.056, 1199.651 };

            var result = m.Mean();

            Assert.AreEqual(1196.58375, Math.Round(result[0], 5));
            Assert.AreEqual(1215.28525, Math.Round(result[1], 5));
            Assert.AreEqual(1178.06175, Math.Round(result[2], 5));
            Assert.AreEqual(1198.037, Math.Round(result[3], 5));
        }
    }
}
