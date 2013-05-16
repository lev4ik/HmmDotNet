namespace HmmDotNet.Extentions
{
    public static class JaggedArray
    {
        public static double[][] Trancate(this double[][] a, int length)
        {
            var N = a.Length - length;
            var result = new double[N][];
            for (var n = 0; n < N; n++)
            {
                result[n] = a[n];
            }
            return result;
        }
    }
}
