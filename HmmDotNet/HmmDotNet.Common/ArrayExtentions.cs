namespace HmmDotNet.Extentions
{
    public static class ArrayExtentions
    {
        public static double[] Add(this double[] arr, double symbol)
        {
            var result = new double[arr.Length + 1];
            for (var i = 0; i < arr.Length; i++)
            {
                result[i] = arr[i];
            }
            result[arr.Length] = symbol;
            return result;
        }
    }
}
