using System.Collections.Generic;

namespace HmmDotNet.Extentions
{
    public static class Converters
    {
        public static double[][] ToArray(this IDictionary<int, double[]> dic)
        {
            var arr = new double[dic.Count][];
            foreach (var keyValuePair in dic)
            {
                arr[keyValuePair.Key] = keyValuePair.Value;
            }
            return arr;
        }
    }
}
