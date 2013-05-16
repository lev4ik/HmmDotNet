using System;

namespace HmmDotNet.Logic.Test.MachineLearning.Data
{
    public class PredictionFileRecord : DataRecord
    {
        public double PredictedOpen { get; set; }
        public double PredictedHigh { get; set; }
        public double PredictedLow { get; set; }
        public double PredictedClose { get; set; }
    }
}
