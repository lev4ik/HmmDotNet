namespace HmmDotNet.Extentions
{
    public static class ArrayExtentions
    {
        public static T[] Add<T>(this T[] arr, T symbol)
        {
            var result = new T[arr.Length + 1];
            for (var i = 0; i < arr.Length; i++)
            {
                result[i] = arr[i];
            }
            result[arr.Length] = symbol;
            return result;
        }

        public static void Swap<T>(this T[] arr, int index1, int index2)
        {
            var temp = arr[index1];
            arr[index1] = arr[index2];
            arr[index2] = temp;
        }
    }
}
