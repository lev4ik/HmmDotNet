using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.Extentions.Test
{
    [TestClass]
    public class ChangeRatioFinderTest
    {
        [TestMethod]
        public void GetMaximumChangeRatios_10ElementsArray_RatiosFound()
        {
            var array = new double[10][]
                {
                    new []{1353.36,1362.03,1343.35,1359.88},
                    new []{1355.41,1360.62,1348.05,1353.33},
                    new []{1374.64,1380.13,1352.5,1355.49},
                    new []{1380.03,1388.81,1371.39,1374.53},
                    new []{1379.86,1384.87,1377.19,1380},
                    new []{1377.55,1391.39,1373.03,1379.85},
                    new []{1394.53,1401.23,1377.51,1377.51},
                    new []{1428.27,1428.27,1388.14,1394.53},
                    new []{1417.26,1433.38,1417.26,1428.39},
                    new []{1414.02,1419.9,1408.13,1417.26}
                };

            var crf = new ChangeRatioFinder();

            var result = crf.GetMaximumChangeRatios(array);

            Assert.AreEqual(2.43, Math.Round(result.Up, 2));
            Assert.AreEqual(0.78, Math.Round(result.Down, 2));
        }
    }
}
