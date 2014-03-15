using System;
using System.Collections.Generic;
using System.Linq;

namespace HmmDotNet.Extentions
{
    public static class ListExtensions
    {
        public static IList<T> Clone<T>(this IList<T> listToClone) where T : ICloneable
        {
            return listToClone.Select(item => (T)item.Clone()).ToList();
        }

        public static IList<T> Copy<T>(this IList<T> list)
        {
            var copy = new List<T>();
            foreach (var item in list)
            {
                copy.Add(item);
            }

            return copy;
        }
    }
}
