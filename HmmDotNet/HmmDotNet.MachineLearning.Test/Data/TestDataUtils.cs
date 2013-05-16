using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using HmmDotNet.Mathematic;

namespace HmmDotNet.Logic.Test.MachineLearning.Data
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

    public class TestDataUtils : ITestDataUtils
    {
        private const string username = @"lev";
        public string INTLFilePath = string.Format(@"C:\Users\{0}\Dropbox\Trading Automation Project\TA.Logic.Test\MachineLearning\Data\INTL\intl.csv", username);
        public string Sp500FilePath = string.Format(@"C:\Users\{0}\Dropbox\Trading Automation Project\TA.Logic.Test\MachineLearning\Data\SP500\SP500.csv", username);
        public string MSFTFilePath = string.Format(@"C:\Users\{0}\Dropbox\Trading Automation Project\TA.Logic.Test\MachineLearning\Data\MSFT\msft.csv", username);
        public string FTSEFilePath = string.Format(@"C:\Users\{0}\Dropbox\Trading Automation Project\TA.Logic.Test\MachineLearning\Data\FTSE\FTSE.csv", username);
        public string GOOGFilePath = string.Format(@"C:\Users\{0}\Dropbox\Trading Automation Project\TA.Logic.Test\MachineLearning\Data\GOOG\goog.csv", username);
        public string EURUSDFilePath = string.Format(@"C:\Users\{0}\Dropbox\Trading Automation Project\TA.Logic.Test\MachineLearning\Data\EURUSD\EURUSD.csv", username);

        private string OutputDirectory = string.Format(@"C:\Users\{0}\Dropbox\Trading Automation Project\TA.Logic.Test\MachineLearning\Data\", username);

        public string GetMathematicaArray(string filePath, DateTime fromDate, DateTime toDate)
        {
            var array = GetSvcData(filePath, fromDate, toDate);
            var builder = new StringBuilder();
            builder.Append("{");
            for (int i = 0; i < array.Length; i++)
            {
                builder.Append(new Vector(array[i]));
                if (i < array.Length - 1)
                {
                    builder.Append(",");
                }
            }
            builder.Append("}");
            return builder.ToString();
        }

        public double[][] GetSvcData(string filePath, DateTime fromDate, DateTime toDate)
        {
            var fileContent = new List<string>();
            using (var reader = new StreamReader(File.OpenRead(filePath)))
            {
                var skipRows = 1;
                while (!reader.EndOfStream)
                {
                    if (skipRows > 0)
                    {
                        skipRows--;
                        continue;
                    }
                    fileContent.Add(reader.ReadLine());
                }
            }

            var result = new List<DataRecord>();
            for (var i = 1; i < fileContent.Count; i++)
            {
                var arr = fileContent[i].Split(',');
                if (Convert.ToDateTime(arr[0]) >= fromDate &&  Convert.ToDateTime(arr[0]) <= toDate)
                {

                    result.Add(new DataRecord()
                        {
                            RecordDate = Convert.ToDateTime(arr[0]),
                            Open = Convert.ToDouble(arr[1]), 
                            High = Convert.ToDouble(arr[2]), 
                            Low = Convert.ToDouble(arr[3]),
                            Close = Convert.ToDouble(arr[4])
                        });
                }
            }
            var sorted = result.OrderBy(t => t.RecordDate).ToArray();
            var array = new double[result.Count][];
            for (var i = 0; i < result.Count; i++)
            {
                array[i] = new []
                    {
                        sorted[i].Open,
                        sorted[i].High,
                        sorted[i].Low,
                        sorted[i].Close,
                    };
            }

            return array;
        }

        public void SaveToFile(double[][] observations, string filePath, DateTime fromDate, DateTime toDate)
        {
            var fileContent = new List<string>();
            using (var reader = new StreamReader(File.OpenRead(filePath)))
            {
                var skipRows = 1;
                while (!reader.EndOfStream)
                {
                    if (skipRows > 0)
                    {
                        skipRows--;
                        continue;
                    }
                    fileContent.Add(reader.ReadLine());
                }
            }

            var result = new List<PredictionFileRecord>();
            for (var i = 1; i < fileContent.Count; i++)
            {
                var arr = fileContent[i].Split(',');
                if (Convert.ToDateTime(arr[0]) >= fromDate && Convert.ToDateTime(arr[0]) <= toDate)
                {
                    result.Add(new PredictionFileRecord
                        {
                            RecordDate = Convert.ToDateTime(arr[0]),
                            Open = Convert.ToDouble(arr[1]), 
                            High = Convert.ToDouble(arr[2]), 
                            Low = Convert.ToDouble(arr[3]),
                            Close = Convert.ToDouble(arr[4])
                        });
                }
            }

            for (int j = observations.Length - 1; j >= 0; j--)
            {
                result[j].PredictedOpen = observations[j][0];
                result[j].PredictedLow = observations[j][1];
                result[j].PredictedHigh = observations[j][2];
                result[j].PredictedClose = observations[j][3];
            }

            var builder = new StringBuilder();
            for (int k = 0; k < result.Count; k++)
            {
                builder.Append(result[k].RecordDate).Append(",");
                builder.Append(result[k].Open).Append(",");
                builder.Append(result[k].High).Append(",");
                builder.Append(result[k].Low).Append(",");
                builder.Append(result[k].Close).Append(",");
                builder.Append(result[k].PredictedOpen).Append(",");
                builder.Append(result[k].PredictedHigh).Append(",");
                builder.Append(result[k].PredictedLow).Append(",");
                builder.Append(result[k].PredictedClose).Append(Environment.NewLine);
            }
            using (var outfile = new StreamWriter(OutputDirectory + @"\data.csv"))
            {
                outfile.Write(builder.ToString());
            }
        }
    }
}
