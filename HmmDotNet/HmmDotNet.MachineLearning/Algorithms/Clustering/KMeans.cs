using System;
using System.Collections.Generic;
using System.Diagnostics;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Distance;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics;

namespace HmmDotNet.MachineLearning.Algorithms.Clustering
{
    public enum InitialClusterSelectionMethod
    {
        Random = 0,
        Furthest = 1
    }

    public class KMeans
    {
        public const int KMeansDefaultIterations = 50;

        public Dictionary<int, double[][]> Clusters { get; protected set; }

        public double[][] ClusterCenters { get; protected set; }

        public double[][,] ClusterCovariances { get; protected set; }

        public void CreateClusters(double[][] observations, int k, int iterations, InitialClusterSelectionMethod initMethod)
        {
            if (observations.Length < k)
            {
                throw new ApplicationException("Number of clusters must be equal or greater to observations sequence length");
            }

            ClusterCovariances = new double[k][,];
            ClusterCenters = (initMethod == InitialClusterSelectionMethod.Random) ? SetRandomInitialClusters(observations, k) : SetInitialClusters(observations, k);

            Clusters = UpdateClusterData(ClusterCenters, observations, k);

            var convirged = false;

            while (!convirged)
            {
                Clusters = UpdateClusterData(ClusterCenters, observations, k);
                var newCentroid = new double[k][];

                for (int i = 0; i < k; i++)
                {
                    newCentroid[i] = Clusters[i].Mean();
                }

                convirged = iterations-- == 0 || !IsChanged(newCentroid, ClusterCenters, k);
                ClusterCenters = newCentroid;
            }

            for (int i = 0; i < k; i++)
            {
                ClusterCovariances[i] = Utils.Covariance(observations, ClusterCenters[i], null);
            }
        }

        #region Private Methods

        private bool IsChanged(double[][] newCentroid, double[][] oldCentroid, int k)
        {
            for (int i = 0; i < k; i++)
            {
                if (!newCentroid[i].EqualsTo(oldCentroid[i]))
                {
                    return true;
                }
            }
            return false;
        }

        private double[][] SetRandomInitialClusters(double[][] observations, int k)
        {
            var selected = new int[k];
            for (int i = 0; i < k; i++)
            {
                selected[i] = -1;
            }

            var clusterCentroids = new double[k][];

            for (var i = 0; i < k; i++)
            {
                var r = new Random();
                var randomOk = false;
                var place = r.Next(0, observations.Length);

                while (!randomOk)
                {
                    if (!selected.In(place))
                    {
                        randomOk = true;
                        selected[i] = place;
                    }
                    else
                    {
                        place = r.Next(0, observations.Length);
                    }
                }
                
                clusterCentroids[i] = observations[place];

                Debug.WriteLine("Centroid : " + i + " : place " + place + " : " + (new Vector(clusterCentroids[i])).ToString());
            }

            return clusterCentroids;
        }

        private double[][] SetInitialClusters(double[][] observations, int k)
        {
            var clusterCentroids = new double[k][];
            clusterCentroids[0] = observations[0];
            for (int i = 1; i < k; i++)
            {
                clusterCentroids[i] = new double[observations[0].Length];
            }
            Debug.WriteLine("Centroid : 0 : " + (new Vector(clusterCentroids[0])).ToString());

            for (int i = 1; i < k; i++)
            {               
                var point = new double[observations[0].Length];
                for (int j = 0; j < i; j++)
                {
                    for (int l = 0; l < observations[0].Length; l++)
                    {
                        point[l] += clusterCentroids[j][l];
                    }
                }
                point.Init(i);
                
                clusterCentroids[i] = GetPointWithMaxDistance(clusterCentroids, observations, point);

                Debug.WriteLine("Centroid : " + i + " : " + (new Vector(clusterCentroids[i])).ToString());
            }

            return clusterCentroids;
        }

        private Dictionary<int, double[][]> UpdateClusterData(double[][] centroids, double[][] observations, int k)
        {
            var clusters = new Dictionary<int, double[][]>();
            for (int j = 0; j < k; j++)
            {
                clusters.Add(j, new double[0][]);
            }

            for (int n = 0; n < observations.Length; n++)
            {
                double minDistance = double.MaxValue;
                var selectedCluster = int.MinValue;

                for (int i = 0; i < k; i++)
                {
                    var distance = Euclidean.Calculate(centroids[i], observations[n]);
                    if (distance < minDistance)
                    {
                        selectedCluster = i;
                        minDistance = distance;
                    }
                }

                clusters[selectedCluster] = clusters[selectedCluster].Append(observations[n]);
            }

            return clusters;
        }

        private double[] GetPointWithMaxDistance(double[][] clusterCentroids, double[][] observations, double[] point)
        {
            double[] result = null;

            var maxDistance = double.MinValue;
            for (int i = 0; i < observations.Length; i++)
            {
                var distance = Euclidean.Calculate(point, observations[i]);
                if (distance > maxDistance && !clusterCentroids.In(observations[i]))
                {
                    maxDistance = distance;
                    result = observations[i];
                }
            }

            return result;
        }

        #endregion Private Methods
    }
}
