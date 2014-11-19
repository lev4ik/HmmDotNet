using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace HmmDotNet.MachineLearning.Test.Data
{
    public interface ITestDataUtils
    {
        /// <summary>
        ///     Retrieves S&P 500 data from CSV file and transform it to multidimentional array
        /// </summary>
        /// <param name="fromDate"></param>
        /// <param name="toDate"></param>
        /// <param name="filePath"></param>
        /// <returns></returns>
        double[][] GetSvcData(string filePath, DateTime fromDate, DateTime toDate);
    }
}
