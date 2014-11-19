using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace HmmDotNet.MachineLearning.Test.Data
{
    public class TestPipedDataUtils : ITestDataUtils
    {
        private const string username = @"lev";
        public string VideoViewsFilePath = string.Format(@"C:\Users\{0}\Dropbox\Trading Automation Project\TA.Logic.Test\MachineLearning\Data\VideoViews\video_views.txt", username);
        public string OutputDirectory = string.Format(@"C:\Users\{0}\Documents", username);

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

            var result = new List<VideoViewDataRecord>();
            for (var i = 1; i < fileContent.Count; i++)
            {
                var arr = fileContent[i].Split('|');
                if (Convert.ToDateTime(arr[0]) >= fromDate && Convert.ToDateTime(arr[0]) <= toDate)
                {

                    result.Add(new VideoViewDataRecord()
                    {
                        RecordDate = Convert.ToDateTime(arr[0]),
                        VideoViews = Convert.ToDouble(arr[1])
                    });
                }
            }
            var sorted = result.OrderBy(t => t.RecordDate).ToArray();
            var array = new double[result.Count][];
            for (var i = 0; i < result.Count; i++)
            {
                array[i] = new[]
                    {
                        sorted[i].VideoViews
                    };
            }

            return array;
        }

        public void SaveToFile(double[][] test, double[][] predicted)
        {
            var builder = new StringBuilder();
            for (int k = 0; k < test.Length; k++)
            {
                builder.Append(test[k][0]).Append(",");
                builder.Append(predicted[k][0]).Append(",").Append(Environment.NewLine);
            }
            using (var outfile = new StreamWriter(OutputDirectory + @"\data.csv"))
            {
                outfile.Write(builder.ToString());
            }
        }
    }
}
